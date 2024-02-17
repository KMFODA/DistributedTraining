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

import template
import asyncio
import random

from template.utils.misc import get_bandwidth


async def forward(self):
    
    """
    The forward function is called by the validator every time step.

    It is responsible for querying the network and scoring the responses.

    Args:
        self (:obj:`bittensor.neuron.Neuron`): The neuron object which contains all the necessary state for the validator.

    """

    event = {}
    if ((self.config.neuron.global_batch_size_train - self.tracker.global_progress.samples_accumulated) <= 25) and (not self.step_scheduled) and (self.tracker.global_progress.epoch == self.tracker.local_progress.epoch):

        bt.logging.info("Scheduling all-reduce synapse call")
        sample_size=int(self.metagraph.n)
        next_step_control = self.grad_averager.schedule_step()
        self.step_scheduled = True  
        all_reduce = True
    else:

        if (self.tracker.global_progress.epoch != self.tracker.local_progress.epoch):
            bt.logging.info("Local Epoch Behind Global Epoch Loading State From Peers")
            self.grad_averager.load_state_from_peers()
            self.tracker.local_progress.epoch = self.tracker.global_progress.epoch

        sample_size = self.config.neuron.sample_size
        all_reduce = False
    
    # Get as many active miners as possible
    self.miner_uids = await get_random_uids(
        self, dendrite=self.dendrite, k=sample_size
    )
    event.update({"uids":self.miner_uids})
    bt.logging.info(f"UIDs:  {self.miner_uids}")

    query_tasks = []
    if all_reduce:
        with self.tracker.pause_updates():
            bt.logging.info("Gradient Averaging")
            self.grad_averager.step(control=next_step_control, wait=False)

        queries = [template.protocol.AllReduce() for _ in self.miner_uids]
    else:
        queries = [template.protocol.Train( 
                    gradient_test_index = random.choice(self.test_layer_indices),
            ) for _ in self.miner_uids
        ]

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
        # Log the results for monitoring purposes.
        with self.grad_averager.use_averaged_gradients():  # this will fill param.grads with aggregated gradients
            bt.logging.info("Performing Optimizer Step")
            self.opt.step()  # update model parameters using averaged grad
        self.grad_averager.reset_accumulated_grads_()  # prepare for next step
        # local_epoch = self.tracker.update_epoch(local_epoch + 1)
        self.step_scheduled = False 
    else:
        bt.logging.info(
            "Received responses: " + str([
                {
                    'Loss': response.loss,
                    'Dataset Indices': (min(response.dataset_indices), max(response.dataset_indices)),
                    'IP': self.metagraph.axons[uid].ip,
                    'Port': self.metagraph.axons[uid].port,
                    'Hotkey': self.metagraph.axons[uid].hotkey
                } for response, uid in zip(responses[0],self.miner_uids) if response.dendrite.status_code == 200
            ])
        )
    
    # Adjust the scores based on responses from miners.
    rewards = await get_rewards(self, uids=self.miner_uids, responses=responses, all_reduce=all_reduce)
    
    # Normalise Rewards
    if rewards.sum() != 0:
        rewards  = rewards / rewards.sum()
    
    bt.logging.info(f"Final Scores: {rewards}")
    
    # Update the scores based on the rewards.
    self.update_scores(rewards, self.miner_uids)

    event = {}
    event.update(self.get_validator_info())
    try:
        event.update(get_bandwidth())
    except:
        bt.logging.info("Error getting bandwidth metrics")

    # Log to wandb
    if not self.config.neuron.dont_wandb_log:
        self.wandb.log(event)

    return responses
