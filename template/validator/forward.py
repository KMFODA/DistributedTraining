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
import torch
import base64

from hivemind.averaging.group_info import GroupInfo
from hivemind.dht import DHTID

import bittensor as bt
from huggingface_hub import create_tag, list_repo_refs

import template
from template.utils.hivemind import load_state_from_peer
from template.utils.misc import get_bandwidth, update_global_tracker_state
from template.utils.uids import get_random_uids
from template.validator.reward import get_rewards
import copy
import numpy as np


async def forward(self):
    """
    The forward function is called by the validator every time step.

    It is responsible for querying the network and scoring the responses.

    Args:
        self (:obj:`bittensor.neuron.Neuron`): The neuron object which contains all the necessary state for the validator.

    """
    # while self.tracker.global_progress.num_peers == 0:
    #     self.warmup()

    update_global_tracker_state(self)
    if (self.local_progress.epoch < self.global_progress.epoch) and (
        self.model_hf_tag < self.global_progress.epoch
    ):
        bt.logging.info("Local Epoch Behind Global Epoch Loading State From Peers")
        load_state_from_peer(self)

    if (
        (
            (
                self.config.neuron.global_batch_size_train
                - self.global_progress.samples_accumulated
            )
            <= 25
        )
        and (not self.step_scheduled)
        and (self.global_progress.epoch == self.local_progress.epoch)
    ):
        # bt.logging.info("Scheduling all-reduce synapse call")
        sample_size = int(self.metagraph.n)
        # next_step_control = self.grad_averager.schedule_step()
        # self.step_scheduled = True
        all_reduce = True
        self.event.update({"synapse_type": "all_reduce"})

    else:
        sample_size = self.config.neuron.sample_size
        all_reduce = False
        self.event.update({"synapse_type": "train"})

    # Get random available miners based on the respective sample size
    self.miner_uids = await get_random_uids(self, dendrite=self.dendrite, k=sample_size)

    self.event.update({"uids": self.miner_uids})
    bt.logging.info(f"UIDs:  {self.miner_uids}")

    query_tasks = []
    rewards = None

    if all_reduce:
        group_peerids = None

        # All-reduce synapse
        while group_peerids is None or any(
            peer_id is None for peer_id in group_peerids.values()
        ):
            group_peerids = await self.map_uid_to_peerid(self.miner_uids)

        group_id = DHTID.generate().to_bytes()
        print("DHT:", self.dht.peer_id)
        print("Peers:", list(group_peerids.values()))
        ordered_peer_ids = [self.dht.peer_id] + list(group_peerids.values())

        group = template.protocol.Group(
            peer_count=len(group_peerids) + 1,  # Including the local peer
            peer_ids=[peer_id.to_string() for peer_id in ordered_peer_ids],
            group_id=base64.b64encode(group_id),
        )

        queries = [
            template.protocol.AllReduce(
                group=group,
            )
            for _ in self.miner_uids
        ]
        
        # Define a custom group for all-reduce
        custom_group = GroupInfo(group_id, tuple(ordered_peer_ids), gathered=None)

        # The dendrite client queries the network.
        query_tasks.append(
            self.dendrite_pool.async_forward(
                self.miner_uids, queries, timeout=self.all_reduce_timeout
            )
        )

        
        try:
            
            bt.logging.info("Performing Gradient Averaging")
            
            # Perform AllReduce step with queried miners to get averaged gradients
            gradient_averaging_step = self.grad_averager.step(
                custom_group_info=custom_group, wait=False
            )
            
            # Start synapse queries - don't await so we can enter below timeout counter
            queries = asyncio.gather(*query_tasks)

            sleep_counter = 1
            while (gradient_averaging_step.done() is False) and (sleep_counter <= 300):
                time.sleep(1)
                sleep_counter += 1

            if gradient_averaging_step.done():
                # Log the results for monitoring purposes.
                bt.logging.info("Model Weights Before Optimizer Step")
                current_model_weights_sample = copy.copy(
                    [layer for layer in self.model.parameters()][-1][-10:].tolist()
                )
                bt.logging.info(current_model_weights_sample)
                with self.tracker.pause_updates():
                    with self.grad_averager.use_averaged_gradients():
                        bt.logging.info("Performing Optimizer Step")
                        self.opt.step()  # update model parameters using averaged grad
                    bt.logging.info("Model Weights After Optimizer Step")
                    new_model_weights_sample = copy.copy(
                        [layer for layer in self.model.parameters()][-1][-10:].tolist()
                    )
                    bt.logging.info(new_model_weights_sample)

                    if new_model_weights_sample == current_model_weights_sample:
                        # TODO This seems like it could be optimized furhter. Sometimes some weights might not change, no?
                        bt.logging.info("Averaging Failed. Model Weights Haven't Changed.")
                        load_state_from_peer(self, epoch = self.local_progress.epoch + 1)

                    elif np.nan in new_model_weights_sample:
                        bt.logging.info(
                            "Averaging Failed. Model Weights Corrupted With Nans After Running The Optimizer Step."
                        )
                        load_state_from_peer(self, epoch = self.local_progress.epoch + 1)

                    else:
                        self.grad_averager.reset_accumulated_grads_() 
                        self.tracker.local_progress.epoch = self.tracker.update_epoch(
                            self.tracker.local_progress.epoch + 1
                        )
                        self.local_progress.epoch += 1
                        self.local_progress.samples_accumulated = 0

                        refs = list_repo_refs(
                            self.config.neuron.model_name, repo_type="model"
                        )
                        tag_name = max([int(tag.name) for tag in refs.tags]) if refs.tags else None
                        bt.logging.info(f"Old Model Tag {tag_name}")
                        if (
                            tag_name
                            and tag_name < self.local_progress.epoch
                        ):
                            # TODO Is this awaited, if so, might need it as a background task
                            bt.logging.info("Pushing New Model Weights To HF Hub")
                            self.model.push_to_hub(self.config.neuron.model_name)
                            create_tag(
                                self.config.neuron.model_name,
                                repo_type="model",
                                tag=str(self.local_progress.epoch),
                                tag_message="Bump release version.",
                            )
                            refs = list_repo_refs(
                                self.config.neuron.model_name, repo_type="model"
                            )
                            tag_name = max([int(tag.name) for tag in refs.tags])
                            bt.logging.info(f"New Model Tag {tag_name}")
                    
                    scores = torch.FloatTensor([1 for _ in self.miner_uids]).to(self.device)

            
            elif gradient_averaging_step.cancelled():
                raise asyncio.CancelledError("Gradient averaging step was cancelled.")
                
            else:
                raise TimeoutError("Gradient averaging step timed out.")
        
        except Exception as e:
            bt.logging.info(
                f"AllReduce Failed With Error: {e}"
            )  # TODO Propogate timeout error to here + additional bad peers
            scores = torch.FloatTensor([0 for _ in self.miner_uids]).to(self.device)
            responses = [[]]
            # self.update_scores(rewards, self.miner_uids)
            load_state_from_peer(self)

        rewards = await get_rewards(
            self,
            uids=self.miner_uids,
            responses=responses,
            all_reduce=all_reduce,
            scores=scores,
        )

    else:
        # Regular training synapse
        queries = [
            template.protocol.Train(
                gradient_test_index=random.choice(self.test_layer_indices),
            )
            for _ in self.miner_uids
        ]

        query_tasks.append(self.dendrite_pool.async_forward(self.miner_uids, queries))

        responses = await asyncio.gather(*query_tasks)

        bt.logging.info(
            "Received responses: "
            + str(
                [
                    {
                        "Loss": response.loss,
                        "Dataset Indices": (
                            min(response.dataset_indices),
                            max(response.dataset_indices),
                        ),
                        "IP": self.metagraph.axons[uid].ip,
                        "Port": self.metagraph.axons[uid].port,
                        "Hotkey": self.metagraph.axons[uid].hotkey,
                    }
                    for response, uid in zip(responses[0], self.miner_uids)
                    if response.dendrite.status_code == 200
                ]
            )
        )

        rewards = await get_rewards(self, uids=self.miner_uids, responses=responses)

        # else:
        # responses = []

    if rewards is None:
        return responses

    # Normalise Rewards
    if rewards.sum() != 0:
        rewards = rewards / rewards.sum()
        
    bt.logging.info(f"Final Scores: {rewards}")

    # Update the tracker based on the rewards
    if not all_reduce:
        self.update_local_tracker_state(rewards, responses)
    self.event.update(
        {
            "local_samples_accumulated": self.local_progress.samples_accumulated,
            "local_epoch": self.local_progress.epoch,
            "global_samples_accumulated": self.global_progress.samples_accumulated,
            "global_epoch": self.global_progress.epoch,
        }
    )

    # Update the scores based on the rewards.
    self.update_scores(rewards, self.miner_uids)

    self.event.update(self.get_validator_info())
    try:
        self.event.update(get_bandwidth())
    except:
        bt.logging.info("Error getting bandwidth metrics")

    return responses
