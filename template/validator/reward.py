# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# TODO(developer): Set your name
# Copyright © 2023 <your name>

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

import random
from typing import List

import bittensor as bt
import torch
from template.data.dataset import SubsetFalconLoader
from template.utils.misc import get_bandwidth
from hivemind.utils.timed_storage import get_dht_time
import time
import asyncio


def get_loss(self, dataset_indices, batch_size, gradient_accumilation_steps):

    # Create Dataloader
    dataloader = SubsetFalconLoader(
        batch_size=batch_size, sequence_length=1024, rows=dataset_indices
    )

    total_loss = 0
    n_acc_steps = 0
    accumulation_steps = gradient_accumilation_steps

    # Train data for one epoch
    for step, batch in enumerate(dataloader):

        inputs = batch.to(self.device)

        # Forward pass
        outputs = self.model(input_ids=inputs, labels=inputs)
        
        loss = outputs.loss

        # Backward Pass
        loss.backward()

        bt.logging.info(f"Step {step} Loss: {outputs.loss.detach().item()}")

    average_loss = total_loss / step

    bt.logging.info(f"Final Loss:           {outputs.loss.detach().item()}")
    bt.logging.info(f"Average Loss:         {average_loss}")

    return average_loss

def get_local_score(self, synapse):

    if False: # Dummy fix need to switch to if self.tracker.global_progress.epoch != self.current_epoch:
        score = 1
    else:
        loss = get_loss(self, synapse.dataset_indices, synapse.batch_size, synapse.gradient_accumilation_steps)
        bt.logging.info(f"Calculated Loss:  {loss}")
        bt.logging.info(f"Synapse Loss:     {synapse.loss}")
        # The miner's local score is the variance between the loss it returns and the 
        # loss the validator calculates for the last batch of data sent to that miner
        score = 1-(abs(loss-synapse.loss)/loss)
        bt.logging.info(f"Local Score:      {score}")

    return score
    

def score_gradients(self, response):
    
    # Create Dataloader
    dataloader = SubsetFalconLoader(
        batch_size=response.batch_size, sequence_length=1024, rows=response.dataset_indices
    )

    # Train data for one epoch
    for step, batch in enumerate(dataloader):

        inputs = batch.to(self.device)

        # Forward pass
        outputs = self.model(input_ids=inputs, labels=inputs)

        loss = outputs.loss

        # Backward Pass
        loss.backward()

        bt.logging.info(f"Step {step} Loss: {outputs.loss.detach().item()}")
    
        if not self.config.neuron.dont_wandb_log:
            self.wandb.log({"loss": outputs.loss.detach().item()})

    gradients = []
    for layer in self.model.parameters():
        gradients.append(layer.grad)
    
    gradients = float(sum(gradients[response.gradient_test_index]))
        
    score = 1-(abs(gradients-response.gradients))
    score = score * len(response.dataset_indices)

    return score


async def score_blacklist(self, uids, scores):
    
    peer_ids = []

    for i, uid in enumerate(uids):
        peer_id = await self.map_uid_to_peerid(uid)
        if peer_id == None:
            scores[i] = 0.0
        else:
            scores[i] = 1.0
        peer_ids.append(peer_id)

    return peer_ids, scores

async def score_bandwidth(self, peer_ids, scores):

    for i, peer in enumerate(peer_ids):
        
        try:
            start_time = time.perf_counter()
            metadata, tensors = await asyncio.wait_for(self.load_state_from_miner(peer.peer_id), timeout=60)

            end_time = time.perf_counter()

            if (metadata is None) or (tensors is None):
                scores[i] = 0
            else:
                scores[i] = end_time - start_time

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
    all_reduce: bool,
) -> torch.FloatTensor:
    """
    Returns a tensor of rewards for the given query and responses.

    Args:
    - uids (List[int]): A list of uids that were queried.
    - responses (List[float]): A list of responses from the miners.

    Returns:
    - torch.FloatTensor: A tensor of rewards for the given query and responses.
    """
    # TODO Test this
    if (responses == [[]]) or ([response[0] for response in responses if response[0].dendrite.status_code == 200 and response[0].loss != []] == []):
        scores = torch.FloatTensor([0 for uid in uids]).to(self.device)
    else:
        scores = torch.FloatTensor([1 if response.dendrite.status_code == 200 and response.loss != [] else 0 for _, response in zip(uids, responses[0])]).to(self.device)
        bt.logging.info(f"Timeout Scores: {scores}")

        if (self.step != 0) and ((self.step % 10)==0):
            # Periodically check if peer is connected to DHT & run_id and blacklist them if they are not
            peer_ids, scores = await score_blacklist(self, uids, scores)
            bt.logging.info(f"DHT Blacklist Scores: {scores}")

        if all_reduce:
            # Score miners bandwidth
            scores = await score_bandwidth(self, peer_ids, scores)
            bt.logging.info(f"Bandwidth Scores: {scores}")
        else:
            # Adjust Global Score with Local Score
            test_uids_index = [uid_index for uid_index, uid in enumerate(uids) if responses[0][uid_index].dendrite.status_code == 200]
            
            # test_uids_sample_index = random.sample(test_uids_index, k = min(4, len(test_uids_index)))
            test_uids_sample_index = random.sample(test_uids_index, k = 1)
            
            scores = torch.FloatTensor([scores[uid_index] * score_gradients(self, responses[0][uid_index]) 
                                        if uid_index in test_uids_sample_index else scores[uid_index] 
                                        for uid_index,_ in enumerate(uids)]).to(self.device)
            bt.logging.info(f"Gradient Scores: {scores}")

    return scores

