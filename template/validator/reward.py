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
from hivemind.utils.timed_storage import get_dht_time
import time

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
        
        # Zero gradients
        # self.opt.zero_grad()

        # Normalize loss to account for batch accumulation
        # loss = outputs.loss / accumulation_steps 
        loss = outputs.loss

        # Accumulate Total Loss
        total_loss += outputs.loss.detach().item() 

        # Backward Pass
        loss.backward()

        # Adjust gradient
        # self.opt.step()

        # if (step + 1) % accumulation_steps == 0:
        #     n_acc_steps += 1
            
        #     bt.logging.info(f"Step {n_acc_steps} Loss: {outputs.loss.detach().item()}")
            
        #     if not self.config.neuron.dont_wandb_log:
        #         self.wandb.log({"loss": outputs.loss.detach().item(), "opt_local_epoch": self.opt.local_epoch})

        # torch.cuda.empty_cache()

        bt.logging.info(f"Step {step} Loss: {outputs.loss.detach().item()}")
    
        if not self.config.neuron.dont_wandb_log:
            self.wandb.log({"loss": outputs.loss.detach().item()})

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
    
async def score_blacklist(self, uids):

    rewards = torch.zeros(len(uids))
    peer_ids = []
    peer_list = await self._p2p.list_peers()

    for i, uid in enumerate(uids):
        peer_id = await self.get_uid_peerid(peer_list, uid)
        if peer_id == []:
            rewards[i] = 0.0
        else:
            rewards[i] = 1.0
        peer_ids.append(peer_id)

    return peer_ids, rewards

async def score_bandwith(self, peer_ids):

    rewards = torch.zeros(len(peer_ids))

    for i, peer in enumerate(peer_ids):
        start_time = time.perf_counter()
        _ = self.load_state_from_peers(peer[0])
        end_time = time.perf_counter()
        rewards[i] = end_time - start_time

    return rewards
                     
async def get_rewards(
    self,
    uids: List[int],
    responses: list
) -> torch.FloatTensor:
    """
    Returns a tensor of rewards for the given query and responses.

    Args:
    - uids (List[int]): A list of uids that were queried.
    - responses (List[float]): A list of responses from the miners.

    Returns:
    - torch.FloatTensor: A tensor of rewards for the given query and responses.
    """

    # load_state_from_peers_status = False
    # retries = 0
    # while load_state_from_peers_status is False:
    #     try:
    #         load_state_from_peers_status = self.state_averager.load_state_from_peers()
    #     except Exception as e:
    #         bt.logging.error(f"Attempt {retries + 1} to write to the load state from peers failed: {e}")
    #         retries += 1
    #         bt.logging.error(f"Retrying ...")

    peer_ids, rewards = await score_blacklist(self, uids)
    rewards = await score_bandwith(self, peer_ids)
    breakpoint()

    global_step = self.dataset_common_state.get_dht("step")
    if global_step is not None:
        self.global_step = global_step
    bt.logging.info(f"Global Step:   {self.global_step}")
    if (self.global_step % 100 == 0) and (self.global_step != 0):
        self.dataset_indices_list_test = (
            self.dataset_common_state.get_dataset_indices_test(
                self.config.neuron.local_batch_size_test
            )
        )
    # Get loss on randomly selected test dataset to be used for the Global Score
    loss = get_loss(self, self.dataset_indices_list_test, self.config.neuron.local_batch_size_test, self.config.neuron.local_gradient_accumilation_steps_test)

    if not self.config.neuron.dont_wandb_log:
        self.wandb.log({"loss": loss, "previous_loss": self.previous_loss})

    # Get latest previous loss from DHT
    # self.previous_loss = self.dataset_common_state.get_dht("loss")
    previous_loss = self.dataset_common_state.get_dht("loss")
    if previous_loss is not None:
        self.previous_loss = previous_loss
    bt.logging.info(f"Previous Global Loss:    {self.previous_loss}")
    bt.logging.info(f"Current Global Loss:     {loss}")

    # Compute Global Score
    if ((self.previous_loss is None) or ((loss - self.previous_loss) < 0)) and (responses[0] != []):
        score = 1
        # self.dataset_common_state.set_dht("loss", float(loss))
        self.dataset_common_state.set_dht("loss", loss)
    else:
        score = 0

    # Log score, previous and current loss
    bt.logging.info(f"Global Score:            {score}")

    # Set previous loss to current loss
    self.previous_loss = loss
    
    # Write loss to dht
    self.dataset_common_state.set_dht("loss", loss)

    # Get all the reward results by iteratively calling your reward() function.
    scores = torch.FloatTensor([score if response.dendrite.status_code == 200 and response.loss != [] else 0 
                                for _, response in zip(uids, responses[0])]).to(self.device)
    bt.logging.info(f"Global Scores:        {scores}")

    # Adjust Global Score with Local Score
    test_uids_index = [uid_index for uid_index, uid in enumerate(uids) 
                       if responses[0][uid_index].dendrite.status_code == 200]
    
    test_uids_sample_index = random.sample(test_uids_index, k = min(4, len(test_uids_index)))
    
    scores = torch.FloatTensor([scores[uid_index] * get_local_score(self, responses[0][uid_index]) 
                                if uid_index in test_uids_sample_index else scores[uid_index] 
                                for uid_index,_ in enumerate(uids)]).to(self.device)

    bt.logging.info(f"Adjusted Global Scores:   {scores}")
    
    return scores

