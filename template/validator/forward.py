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
from hivemind.utils.timed_storage import get_dht_time

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

    # if self.opt._should_load_state_from_peers():
    #     bt.logging.info("Local state is behind global state")
    #     self.opt.load_state_from_peers()
        # self.opt.state_averager.load_state_from_peers()
    
    # self.opt.state_averager.load_state_from_peers()

    event = {}
    self.miner_uids = await get_random_uids(
        self, dendrite=self.dendrite, k=self.config.neuron.sample_size
    )
    event.update({"uids":self.miner_uids})
    bt.logging.info(f"UIDs:  {self.miner_uids}")
    # breakpoint()  

    # self.opt.use_gradient_averaging
    # self.opt._update_global_epoch
    # self.opt.grad_scaler
    # self.opt.scheduled_grads
    # self.opt.grad_averager.step(control=self.opt.scheduled_grads, reset_accumulators=True, wait=False)
    # if self.opt.tracker.estimated_next_update_time - get_dht_time() <= self.opt.matchmaking_time:
    if ((self.config.neuron.global_batch_size_train - self.tracker.global_progress.samples_accumulated) <= 25) and (not self.step_scheduled):
        next_step_control = self.grad_averager.schedule_step()
        self.step_scheduled = True  
        all_reduce = True
        bt.logging.info("Scheduling all-reduce synapse call")
    else:
        all_reduce = False

    query_tasks = []
    if all_reduce:
        # import hivemind
        # next_step_time = hivemind.get_dht_time() + 60   # runs global steps every 60 seconds
        # next_step_control = None

        # timeout = 60
        # control = self.grad_averager.schedule_step(timeout=timeout)
        # self.grad_averager.load_accumulators_into_averager_()
        # self.grad_averager._accumulators_used_in_step = True
        # self.grad_averager._new_averaged_grads = True
        # weight = None
        # control.weight = self.grad_averager.local_samples_accumulated if weight is None else weight
        # reset_accumulators = True
        # if reset_accumulators:
        #     self.grad_averager.reset_accumulated_grads_()
        # control.allow_allreduce()
        # control.result(timeout)
        # next_step_control = self.grad_averager.schedule_step(scheduled_time=next_step_time)
        # self.grad_averager.step(control=next_step_control, timeout = timeout)

        # self.opt.grad_averager.next_chunk_timeout = 15
        # self.opt.grad_averager.is_looking_for_group
        # self.opt.scheduled_grad = self.opt.grad_averager.schedule_step(timeout=self.opt.averaging_timeout)
        # self.opt.grad_scaler = None
        # self.opt._update_global_epoch(self.opt.grad_scaler)

        with self.tracker.pause_updates():
            bt.logging.info("Gradient Averaging")
            self.grad_averager.step(control=next_step_control, wait=False)
        #     with self.grad_averager.use_averaged_gradients():  # this will fill param.grads with aggregated gradients
        #         bt.logging.info("Performing Optimizer Step")
        #         self.opt.step()  # update model parameters using averaged grad
        #     self.grad_averager.reset_accumulated_grads_()  # prepare for next step
        #     # local_epoch = self.tracker.update_epoch(local_epoch + 1)
        #     self.step_scheduled = False 

        queries = [template.protocol.AllReduce() for uid in self.miner_uids]
    else:
        queries = [
            template.protocol.Train( 
                    gradient_test_index = random.choice(self.test_layer_indices),
            ) for uid in self.miner_uids
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

    # Update global step
    step_update_status = self.dataset_common_state.update_step()
    if step_update_status is None:
        self.global_step += 1

    self.step += 1

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
