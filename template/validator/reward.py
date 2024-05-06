# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 KMFODA

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import asyncio
import random
import time
from typing import List

import bittensor as bt
import torch

from template.data.dataset import SubsetFalconLoader#, get_random_batches
from template.utils.misc import compute_losses
from template.utils.uids import get_random_uids


def score_metrics(self):
    """
    Scores the metrics of the model after running AllReduce.
    We are using both loss and perplexity here to get a more nuanced view of the model's performance to balance the validation of miners more fairly.
    """
    
    # Load a random set of batches
    batches = get_random_batches( n = self.config.pages_per_epoch, batch_size = self.config.bs, sequence_length = self.config.sl )
    
    # TODO Should we do add/remove delta? Or just use the model as is?
    average_loss, ppl = self.previous_loss - compute_losses(self.model, batches, device=self.config.device)
                    
    delta_loss = average_loss - self.previous_loss
    delta_ppl = ppl - self.previous_ppl
    
    # Weights are the softmax of the loss and perplexity deltas, i.e. scaled according to how much the metrics have changed.
    score = self.config.alpha * torch.softmax(delta_loss, dim=0) + (1 - self.config.alpha)
    score *= self.config.beta * torch.softmax(delta_ppl, dim=0) + (1 - self.config.beta)
    
    self.previous_loss = average_loss
    self.previous_ppl = ppl
    
    return score
    
    
def score_gradients(self, response, uid):
    
    # Create Dataloader
    dataloader = SubsetFalconLoader(
        batch_size=self.config.neuron.local_batch_size_train, sequence_length=1024, rows=response.dataset_indices
    )

    # Train data for on last indices
    for index, batch in enumerate(dataloader): continue

    inputs = batch.to(self.device)

    # Zero Gradients
    self.opt.zero_grad()
    
    # Forward pass
    outputs = self.model(input_ids=inputs, labels=inputs)

    loss = outputs.loss

    # Backward Pass
    loss.backward()

    # Accumulate Gradients
    self.grad_averager.accumulate_grads_(batch_size=len(inputs))
    
    if not self.config.neuron.dont_wandb_log:
        self.wandb.log({"loss": outputs.loss.detach().item()})

    # Store summed random gradients in the synapse
    gradients = tuple(param.grad.detach().cpu().clone() if param.grad is not None else torch.zeros_like(param) for param in self.model.parameters())
    gradients =  float(torch.sum(torch.abs(gradients[response.gradient_test_index])))
        
    bt.logging.info(f"Local Validator Sum of Layer {response.gradient_test_index}'s Gradients are: {gradients}")
    bt.logging.info(f"UID {uid} Sum of Layer {response.gradient_test_index}'s Gradients are: {response.gradients}")

    # TODO Address issue where gradient sum is negative
    score = 1-(abs(gradients-response.gradients))
    
    return score


async def score_blacklist(self, uids):
    
    scores = torch.FloatTensor([1 for _ in uids]).to(self.device)
    for i, uid in enumerate(uids):
        if self.uids_to_peerids[uid] == None:
            scores[i] = 0.0
        else:
            scores[i] = 1.0

    return scores

async def score_bandwidth(self, uids, timeout = 60):
    
    scores = torch.FloatTensor([1 for _ in uids]).to(self.device)
    for i, uid in enumerate(uids):
        peer = self.uids_to_peerids[uid]

        if peer is None:

            scores[i] = 0
        
        else:
        
            try:
                start_time = time.perf_counter()

                metadata, tensors = await asyncio.wait_for(self.load_state_from_miner(peer), timeout=timeout)
                end_time = time.perf_counter()

                if (metadata is None) or (tensors is None):
                    scores[i] = 0
                else:
                    scores[i] = 1 - ((end_time - start_time)/timeout)

                bt.logging.info(f"Reward for peer {peer} is {scores[i]}")

            except Exception as e:

                bt.logging.info(f"Failed to download state from {peer} - {repr(e)}")
                scores[i] = 0
                bt.logging.info(f"Reward for peer {peer} is {scores[i]}")

    return scores
                     
async def get_rewards(
    self,
    uids: List[int],
    responses: list,
    all_reduce: bool = False,
    scores = None,
) -> torch.FloatTensor:
    """
    Returns a tensor of rewards for the given query and responses.

    Args:
    - uids (List[int]): A list of uids that were queried.
    - responses (List): A list of all the responses from the queried uids.
    - all_reduce (bool): A boolean representing wether the all_reduce synapse was called.
    - responses (List[float]): A list of responses from the miners.

    Returns:
    - torch.FloatTensor: A tensor of rewards for the given query and responses.
    """
    if (responses != [[]]) and ([response[0] for response in responses if response[0].dendrite.status_code == 200] != []):
        
        if all_reduce:          

            # Check if peer is connected to DHT & run_id and blacklist them if they are not
            blacklist_scores = await score_blacklist(self, uids)
            bt.logging.info(f"DHT Blacklist Scores: {blacklist_scores}")
            self.event.update({f"rewards.blacklist.uid{uid}": blacklist_score for uid, blacklist_score in zip(uids, blacklist_scores)})
            scores *= blacklist_scores

            # Score miners bandwidth
            bandwidth_scores = await score_bandwidth(self, uids)
            bt.logging.info(f"Bandwidth Scores: {bandwidth_scores}")
            self.event.update({f"rewards.bandwidth_scores.uid{uid}": bandwidth_score for uid, bandwidth_score in zip(uids, bandwidth_scores)})
            scores *= bandwidth_scores
            
            # TODO Add more metrics after the all-reduce?
            #scores *= score_metrics()        
        
        else:

            scores = torch.FloatTensor([1 if response.dendrite.status_code == 200 and response.loss != [] else 0 for _, response in zip(uids, responses[0])]).to(self.device)
            bt.logging.info(f"Timeout Scores: {scores}")

            # Periodically check if peer is connected to DHT & run_id and blacklist them if they are not
            if ((self.step % 10)==0):

                # Update mapping of uids to peerids
                self.uids_to_peerids = await self.map_uid_to_peerid(range(0, self.metagraph.n))
                
                # Check if peer is connected to DHT & run_id and blacklist them if they are not
                blacklist_scores = await score_blacklist(self, uids.tolist())
                bt.logging.info(f"DHT Blacklist Scores: {blacklist_scores}")
                self.event.update({f"rewards.blacklist.uid{uid}": blacklist_score for uid, blacklist_score in zip(uids, blacklist_scores)})
                scores *= blacklist_scores

            # Re-calculate gradients for a subset of uids and score the difference between local gradients and the miner's gradients
            gradient_scores = torch.FloatTensor([score_gradients(self,response, uids.tolist()[index]) if (response.dendrite.status_code == 200) and (scores[index] != 0) else 0 for index, response in enumerate(responses[0])]).to(self.device)
            bt.logging.info(f"Gradient Scores: {gradient_scores}")
            self.event.update({f"rewards.gradient.uid{uid}": gradient_score for uid, gradient_score in zip(uids.tolist(), gradient_scores)})
            scores *= gradient_scores

            # Calculate Data Indices Scores
            steps_scores = torch.FloatTensor([len(response.dataset_indices) if (response.dendrite.status_code == 200) and (scores[index] != 0) else 0 for index, response in enumerate(responses[0])]).to(self.device)
            bt.logging.info(f"Steps Scores: {steps_scores}")
            self.event.update({f"rewards.steps.uid{uid}": steps_score for uid, steps_score in zip(uids.tolist(), steps_scores)})
            scores *= steps_scores

    return scores