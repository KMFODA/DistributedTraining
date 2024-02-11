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

import bittensor as bt

from template.protocol import Train
from template.utils.misc import AsyncDendritePool
from template.validator.reward import get_rewards
from template.utils.uids import get_random_uids
from hivemind.utils.timed_storage import get_dht_time

import template
import asyncio
import torch


async def forward(self):
    
    """
    The forward function is called by the validator every time step.

    It is responsible for querying the network and scoring the responses.

    Args:
        self (:obj:`bittensor.neuron.Neuron`): The neuron object which contains all the necessary state for the validator.

    """
    event = {}
    self.miner_uids = await get_random_uids(
        self, dendrite=self.dendrite, k=self.config.neuron.sample_size
    )
    event.update({"uids":self.miner_uids})
    bt.logging.info(f"UIDs:  {self.miner_uids}")

    if self.opt.tracker.estimated_next_update_time - get_dht_time() <= self.opt.matchmaking_time:
        all_reduce = True

    if all_reduce:
        self.opt.grad_averager.schedule_step(timeout=self.opt.averaging_timeout)
        queries = [template.protocol.AllReduce() for uid in self.miner_uids]
    else:
        datapoints_per_group = self.config.neuron.training_examples_per_miner
        self.dataset_indices_list = self.dataset_common_state.get_dataset_indices(
                groups_count=len(self.miner_uids),
                items_per_group=datapoints_per_group,
        )

        query_tasks = []

        queries = [
            template.protocol.Train( 
                    dataset_indices = uid_dataset,
                    run_id = self.config.neuron.run_id,
                    batch_size = self.config.neuron.local_batch_size_train,
                    gradient_accumilation_steps = self.config.neuron.local_gradient_accumilation_steps_train
            ) for uid_dataset in self.dataset_indices_list
        ]

    # The dendrite client queries the network.
    query_tasks.append(
        self.dendrite_pool.async_forward(
            self.miner_uids,
            queries
        )
    )
    responses = await asyncio.gather(*query_tasks)

    if all_reduce:
        # Log the results for monitoring purposes.
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
    else:
        responses = []
    
    # Adjust the scores based on responses from miners.
    rewards = await get_rewards(self, uids=self.miner_uids, responses=responses, all_reduce=all_reduce)

    bt.logging.info(f"Scored responses: {rewards}")
    
    # Update the scores based on the rewards.
    self.update_scores(rewards, self.miner_uids)

    # Update the current_epoch
    self.current_epoch = 1 # Dummy fix need to switch to self.tracker.global_progress.epoch

    # Update global step
    step_update_status = self.dataset_common_state.update_step()
    if step_update_status is None:
        self.global_step += 1

    return responses
