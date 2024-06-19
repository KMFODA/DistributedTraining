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

import bittensor as bt
from huggingface_hub import create_tag, list_repo_refs

import template
from template.utils.state_loader import load_state_from_peer
from template.utils.misc import get_bandwidth
from template.utils.progress_tracker import update_global_tracker_state
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

    # Get as many active miners as possible
    self.miner_uids = await get_random_uids(self, dendrite=self.dendrite, k=sample_size)

    self.event.update({"uids": self.miner_uids})
    bt.logging.info(f"UIDs:  {self.miner_uids}")

    query_tasks = []
    if all_reduce:
        with self.tracker.pause_updates():
            bt.logging.info("Performing Gradient Averaging")
            gradient_averaging_step = self.grad_averager.step(wait=False)

        queries = [template.protocol.AllReduce() for _ in self.miner_uids]
    else:
        queries = [
            template.protocol.Train(
                gradient_test_index=random.choice(self.test_layer_indices),
            )
            for _ in self.miner_uids
        ]

    # The dendrite client queries the network.
    query_tasks.append(
        self.dendrite_pool.async_forward(
            self.miner_uids, queries, timeout=self.all_reduce_timeout
        )
    )
    bt.logging.info("Responses Sent Out")
    responses = await asyncio.gather(*query_tasks)
    bt.logging.info("Responses Received")
    if all_reduce and responses != []:
        responses = []
        sleep_counter = 1

        while (gradient_averaging_step.done() is False) and (
            sleep_counter <= self.all_reduce_timeout
        ):
            time.sleep(1)
            sleep_counter += 1

        if gradient_averaging_step.done():
            # Log the results for monitoring purposes.
            bt.logging.info("Model Weights Before Optimizer Step")
            current_model_weights = copy.deepcopy(
                [layer for layer in self.model.parameters()][-100][-10:].tolist()[0]
            )
            current_model_weights_sample = copy.copy(
                [layer for layer in self.model.parameters()][-1][-10:].tolist()
            )
            bt.logging.info(current_model_weights_sample)
            with self.tracker.pause_updates():
                with self.grad_averager.use_averaged_gradients():  # this will fill param.grads with aggregated gradients
                    # bt.logging.info({n:p.grad for n,p in self.model.named_parameters() if p.grad is not None})
                    bt.logging.info("Performing Optimizer Step")
                    self.opt.step()  # update model parameters using averaged grad
                bt.logging.info("Model Weights After Optimizer Step")
                new_model_weights = copy.deepcopy(
                    [layer for layer in self.model.parameters()][-100][-10:].tolist()[0]
                )
                new_model_weights_sample = copy.copy(
                    [layer for layer in self.model.parameters()][-1][-10:].tolist()
                )
                bt.logging.info(new_model_weights_sample)

                if new_model_weights == current_model_weights:
                    bt.logging.info("Averaging Failed. Model Weights Haven't Changed.")
                    load_state_from_peer(self, epoch=self.local_progress.epoch + 1)

                elif np.nan in new_model_weights_sample:
                    bt.logging.info(
                        "Averaging Failed. Model Weights Corrupted With Nans After Running The Optimizer Step."
                    )
                    load_state_from_peer(self, epoch=self.local_progress.epoch + 1)

                else:
                    self.grad_averager.reset_accumulated_grads_()  # prepare for next step
                    self.tracker.local_progress.epoch = self.tracker.update_epoch(
                        self.tracker.local_progress.epoch + 1
                    )
                    self.local_progress.epoch += 1
                    self.local_progress.samples_accumulated = 0

                    refs = list_repo_refs(
                        self.config.neuron.model_name, repo_type="model"
                    )
                    tag_name = (
                        max([int(tag.name) for tag in refs.tags]) if refs.tags else None
                    )
                    bt.logging.info(f"Old Model Tag {tag_name}")
                    if (tag_name is not None) and tag_name < self.local_progress.epoch:
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

        else:
            bt.logging.info("Averaging Failed. Loading State From Peer")
            gradient_averaging_step.cancel()
            load_state_from_peer(self)

        self.step_scheduled = False
    else:
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

    # Adjust the scores based on responses from miners.
    rewards = await get_rewards(
        self, uids=self.miner_uids, responses=responses, all_reduce=all_reduce
    )

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
