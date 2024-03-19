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

import bittensor as bt

from template.validator.reward import get_rewards
from template.utils.uids import get_random_uids
from template.utils.hivemind import load_state_from_peer
from hivemind.averaging.group_info import GroupInfo
from hivemind.dht import DHTID

import template
import asyncio
import random

from template.utils.misc import get_bandwidth
import time


async def forward(self):
    
    """
    The forward function is called by the validator every time step.

    It is responsible for querying the network and scoring the responses.

    Args:
        self (:obj:`bittensor.neuron.Neuron`): The neuron object which contains all the necessary state for the validator.

    """

    # bt.logging.info(f"Global samples: {self.tracker.global_progress.samples_accumulated} | Global epoch: {self.tracker.global_progress.epoch} | Number of Peers: {self.tracker.global_progress.num_peers}")
    # if (self.tracker.global_progress.epoch != self.tracker.local_progress.epoch):
    #     bt.logging.info("Local Epoch Behind Global Epoch Loading State From Peers")
    #     load_state_from_peer(self)

    #if ((self.config.neuron.global_batch_size_train - self.tracker.global_progress.samples_accumulated) <= 25) and (not self.step_scheduled) and (self.tracker.global_progress.epoch == self.tracker.local_progress.epoch):
    if self.next_step_time - self.dtraining_time <= 60 * 2:  
        
        sample_size = self.config.neuron.sample_size_allreduce
        self.step_scheduled = True  
        all_reduce = True
        self.event.update({"synapse_type":"all_reduce"})
          
    # else:

    #     sample_size = self.config.neuron.sample_size
    #     all_reduce = False
    #     self.event.update({"synapse_type":"train"})
    
    # Get random miners based on the respective sample size
    self.miner_uids = await get_random_uids(
        self, dendrite=self.dendrite, k=sample_size
    )
    self.event.update({"uids":self.miner_uids})
    bt.logging.info(f"UIDs:  {self.miner_uids}")

    query_tasks = []
    if all_reduce:
        group_peerids = self.map_uid_to_peerid(self.miner_uids+self.uid)
        group = template.protocol.Group(
            peer_count=len(group_peerids), # Including the local peer
            peer_ids=group_peerids,
        )
        
        queries = [template.protocol.AllReduce(
                    group=group,
            ) for _ in self.miner_uids
        ]
        bt.logging.info("Performing Gradient Averaging")
        # Define a custom group for all-reduce
        # TODO should group_peerids be ordered with validator peerid first?
        group_id = DHTID.generate().to_bytes()
        custom_group = GroupInfo(group_id, tuple(group_peerids), gathered=None)

        # Perform AllReduce step with queried miners to get averaged gradients
        gradient_averaging_step = self.grad_averager.step(group=custom_group, wait=False)

    
    # TODO intermittent gradient norm checks, to further check if miners are behaving
    # else:
    #     queries = [template.protocol.GradientCheck( 
    #                 gradient_test_index = random.choice(self.test_layer_indices),
    #         ) for _ in self.miner_uids
    #     ]

    # The dendrite client queries the network.
    query_tasks.append(
        self.dendrite_pool.async_forward(
            self.miner_uids,
            queries
        )
    )
    responses = await asyncio.gather(*query_tasks)
    if all_reduce and responses != []:
        responses = []
        sleep_counter = 1
        while (gradient_averaging_step.done() is False) and (sleep_counter <= 150):
            time.sleep(1)
            sleep_counter += 1
        if gradient_averaging_step.done():
            # Log the results for monitoring purposes.
            bt.logging.info('Model Weights Before Optimizer Step')
            bt.logging.info([layer for layer in self.model.parameters()][-1][-10:])
            #with self.tracker.pause_updates():
            with self.grad_averager.use_averaged_gradients():  # this will fill param.grads with aggregated gradients
                bt.logging.info("Performing Optimizer Step")
                self.opt.step()  # update model parameters using averaged grad
            bt.logging.info('Model Weights After Optimizer Step')
            bt.logging.info([layer for layer in self.model.parameters()][-1][-10:])
            self.grad_averager.reset_accumulated_grads_()  # prepare for next step
            #self.tracker.local_progress.epoch = self.tracker.update_epoch(self.tracker.local_progress.epoch + 1)

        else:
            bt.logging.info("Averaging Failed. Loading State From Peer")
            load_state_from_peer(self)

        #self.step_scheduled = False 
    # else:
    #     bt.logging.info(
    #         "Received responses: " + str([
    #             {
    #                 'Loss': response.loss,
    #                 'Dataset Indices': (min(response.dataset_indices), max(response.dataset_indices)),
    #                 'IP': self.metagraph.axons[uid].ip,
    #                 'Port': self.metagraph.axons[uid].port,
    #                 'Hotkey': self.metagraph.axons[uid].hotkey
    #             } for response, uid in zip(responses[0],self.miner_uids) if response.dendrite.status_code == 200
    #         ])
    #     )
    
    # Adjust the scores based on responses from miners.
    rewards = await get_rewards(self, uids=self.miner_uids, responses=responses, all_reduce=all_reduce)
    
    # Normalise Rewards
    if rewards.sum() != 0:
        rewards  = rewards / rewards.sum()
    
    bt.logging.info(f"Final Scores: {rewards}")
    
    # Update the scores based on the rewards.
    self.update_scores(rewards, self.miner_uids)

    self.event.update(self.get_validator_info())
    try:
        self.event.update(get_bandwidth())
    except:
        bt.logging.info("Error getting bandwidth metrics")


    return responses
