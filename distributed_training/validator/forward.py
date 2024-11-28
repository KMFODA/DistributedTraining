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
import copy
import random
import time

import bittensor as bt
import numpy as np
import torch
from huggingface_hub import list_repo_refs, list_repo_files
from huggingface_hub.utils import HfHubHTTPError

import distributed_training
from distributed_training.utils.misc import get_bandwidth

from distributed_training.utils.state_loader import (
    load_state_from_peer,
    save_and_upload_state,
)
from distributed_training.utils.progress_tracker import update_global_tracker_state
from distributed_training.utils.uids import get_random_uids
from distributed_training.validator.reward import get_rewards


async def forward(self):
    """
    The forward function is called by the validator every time step.

    It is responsible for querying the network and scoring the responses.

    Args:
        self (:obj:`bittensor.neuron.Neuron`): The neuron object which contains all the necessary state for the validator.

    """
    gathered, failed_peers, participating_peers = [], [], []

    update_global_tracker_state(self)
    if self.local_progress.epoch != self.global_progress.epoch:
        bt.logging.info(
            f"Local Epoch {self.local_progress.epoch} Behind Global Epoch {self.global_progress.epoch}. Loading Latest Model State."
        )
        load_state_from_peer(self, epoch=self.global_progress.epoch)

    # Evaluate wether to run an AllReduce or a Train synapse based
    # on the global samples accumulated

    if (
        (
            (
                self.config.neuron.global_batch_size_train
                - self.global_progress.samples_accumulated
            )
            <= 25
        )
        and (self.global_progress.epoch == self.local_progress.epoch)
        and (
            (self.uid != self.master_uid)
            or (self.local_progress.samples_accumulated != 0)
        )
    ):
        if self.uid == self.master_uid:
            # If running an AllReduce synapse, call as many miners as possible
            sample_size = int(self.metagraph.n)
        else:
            sample_size = self.config.neuron.sample_size

        all_reduce = True
        self.event.update({"synapse_type": "all_reduce"})

    else:
        # If running a Train synapse call, only call the sample_size
        sample_size = self.config.neuron.sample_size
        all_reduce = False
        self.event.update({"synapse_type": "train"})

    if (self.uid == self.master_uid) or (all_reduce == False):
        if all_reduce:
            # Get active miners
            while len(self.miner_uids) < (self.config.neuron.min_group_size - 1):
                bt.logging.info(
                    f"Found {len(self.miner_uids)} UIDs. Attempting to find {self.config.neuron.min_group_size-len(self.miner_uids)} more UIDs."
                )
                self.miner_uids = await get_random_uids(
                    self,
                    dendrite=self.dendrite,
                    k=sample_size,
                    epoch=self.local_progress.epoch if all_reduce else None,
                )

        else:
            self.miner_uids = await get_random_uids(
                self,
                dendrite=self.dendrite,
                k=sample_size,
                epoch=self.local_progress.epoch if all_reduce else None,
            )

        self.event.update({"uids": self.miner_uids})
        bt.logging.info(f"UIDs:  {self.miner_uids}")

        if len(self.miner_uids) == 0:
            responses = [[]]
            bt.logging.info("No Active Miners Found This Step.")
        else:
            query_tasks = []
            if all_reduce:
                bt.logging.info("Performing Gradient Averaging")
                self.peerids_to_uids = {
                    str(value[0]): key for key, value in self.uids_to_peerids.items()
                }
                gradient_averaging_step = self.grad_averager.step(
                    gather=0, wait=False, peerids_to_uids=self.peerids_to_uids
                )
                self.learning_rate = self.get_learning_rate()
                bt.logging.info(f"Current Learning Rate: {self.learning_rate}")

                queries = [
                    distributed_training.protocol.AllReduce(
                        learning_rate=self.learning_rate
                    )
                    for _ in self.miner_uids
                ]
            else:
                # Get a random layer to check gradients against
                gradient_test_index = random.choice(self.test_layer_indices)
                queries = [
                    distributed_training.protocol.Train(
                        model_name=self.model.name_or_path,
                        gradient_test_index=gradient_test_index,
                    )
                    for _ in self.miner_uids
                ]

            # # Query the network
            query_tasks.append(
                self.dendrite_pool.async_forward(
                    self.miner_uids,
                    queries,
                    timeout=(
                        self.all_reduce_timeout if all_reduce else self.train_timeout
                    ),
                )
            )

            bt.logging.info("Query Sent Out")
            start_time = time.perf_counter()
            responses = await asyncio.gather(*query_tasks)
            bt.logging.info("Query Responses Received")

            # Process the AllReduce query responses
            if all_reduce and responses != [[]]:
                failed_gradient_all_reduce = False
                # Wait for gradient averaging to finish
                while (gradient_averaging_step.done() is False) and (
                    (time.perf_counter() - start_time) <= (self.all_reduce_timeout)
                ):
                    time.sleep(1)

                if gradient_averaging_step.done():
                    (
                        gathered,
                        failed_peers,
                        participating_peers,
                    ) = gradient_averaging_step.result()

                    batch_size = sum(
                        [
                            value
                            if (value is not None) and (key not in failed_peers)
                            else 0
                            for key, value in gathered.items()
                        ]
                    )

                    participating_uids = [
                        self.peerids_to_uids.get(str(participating_peer), "'''")
                        for participating_peer in participating_peers
                    ]
                    failed_uids = [
                        self.peerids_to_uids.get(str(failed_peer), "'''")
                        for failed_peer in failed_peers
                    ]

                    all_reduce_scores = {}
                    for uid in range(int(self.metagraph.n)):
                        if (uid in participating_uids) and (uid not in failed_uids):
                            all_reduce_scores[str(uid)] = "SUCCESS"
                        elif uid in failed_peers:
                            all_reduce_scores[str(uid)] = "FAIL"
                        else:
                            all_reduce_scores[str(uid)] = "NON_PARTICIPATING"

                    self.model.config.all_reduce_scores = all_reduce_scores
                    bt.logging.info(f"Gathered {gathered} gradients")
                    bt.logging.info(f"Failed allreduce: {failed_peers}")
                    bt.logging.info(f"Participating peers: {participating_peers}")
                    bt.logging.info(f"Batch Size: {batch_size}")
                    bt.logging.info(f"Failed UIDs: {failed_uids}")
                    bt.logging.info(f"Participating UIDs: {participating_uids}")
                    bt.logging.info(f"AllReduce UID Scores: {all_reduce_scores}")

                    self.event.update(
                        {
                            "batch_size": batch_size,
                            "failed_peers_count": len(failed_peers),
                            "participating_peers_count": len(participating_peers),
                            "succesfull_peers_count": len(participating_peers)
                            - len(failed_peers),
                        }
                    )

                    # Optimizer Step
                    with self.grad_averager.use_averaged_gradients():
                        # Log Model Weight Before Optimizer Step
                        bt.logging.info("Model Weights Before Optimizer Step")
                        current_model_weights_sample = copy.copy(
                            [layer for layer in self.model.parameters()][-2][
                                -10:
                            ].tolist()
                        )
                        bt.logging.info(current_model_weights_sample)

                        bt.logging.info(
                            f"Updating Optimizer Learning Rate To: {self.learning_rate}"
                        )
                        for param_group in self.opt.param_groups:
                            param_group["lr"] = self.learning_rate

                        # Update model parameters using averaged gradients
                        bt.logging.info("Performing Optimizer Step")
                        self.opt.step()
                        # Reset gradient buffers
                        self.grad_averager.reset_accumulated_grads_()

                    # Log Model Weight After Optimizer Step
                    bt.logging.info("Model Weights After Optimizer Step")
                    new_model_weights_sample = copy.copy(
                        [layer for layer in self.model.parameters()][-2][-10:].tolist()
                    )
                    bt.logging.info(new_model_weights_sample)

                    if new_model_weights_sample == current_model_weights_sample:
                        bt.logging.info(
                            "Averaging Failed. Model Weights Haven't Changed. Loading Latest Model State."
                        )
                        failed_gradient_all_reduce = True
                        load_state_from_peer(self, epoch=self.local_progress.epoch + 1)

                    elif sum(np.isnan(new_model_weights_sample)) > 1:
                        bt.logging.info(
                            "Averaging Failed. Model Weights Corrupted With NaNs After Running The Optimizer Step. Loading Latest Model State."
                        )
                        failed_gradient_all_reduce = True
                        state_loaded = load_state_from_peer(
                            self, epoch=self.local_progress.epoch + 1
                        )
                        if not state_loaded:
                            state_loaded = load_state_from_peer(
                                self, epoch=self.local_progress.epoch
                            )

                    else:
                        # Update local progress
                        self.local_progress.epoch += 1
                        self.local_progress.samples_accumulated = 0

                        bt.logging.info(f"Old Model Tag {self.global_progress.epoch}")

                        attempt = 0
                        while attempt < self.model_upload_retry_limit:
                            try:
                                bt.logging.info(
                                    f"Pushing new model and optimizer state to HF Hub with tag {self.local_progress.epoch}"
                                )

                                # Save and upload both model and optimizer state
                                upload_success = save_and_upload_state(
                                    self,
                                    epoch=self.local_progress.epoch,
                                    batch_size=batch_size,
                                    participating_peers=participating_peers,
                                    failed_peers=failed_peers,
                                )

                                if upload_success:
                                    # Cast back to float32 outside of upload context:
                                    self.model.to(dtype=torch.float32)

                                    # Verify the upload
                                    updated_refs = list_repo_refs(
                                        self.config.neuron.model_name,
                                        repo_type="model",
                                    )
                                    new_tag = max(
                                        [int(tag.name) for tag in updated_refs.tags]
                                    )
                                    bt.logging.info(
                                        f"Successfully pushed new model with tag {new_tag}"
                                    )
                                    # Wait to allow out of sync miners to donwload new model state
                                    time.sleep(self.load_state_timeout)
                                    break

                            except HfHubHTTPError as e:
                                attempt += 1
                                bt.logging.info(f"{e}. Loading State from Peer.")
                                state_loaded = load_state_from_peer(
                                    self, epoch=self.global_progress.epoch
                                )
                                if state_loaded:
                                    break
                            except Exception as e:
                                attempt += 1
                                bt.logging.warning(
                                    f"Failed To Upload Model To HF hub, Retrying. Attempt {attempt}/{self.model_upload_retry_limit}."
                                )
                                if attempt < self.model_upload_retry_limit:
                                    time.sleep(self.model_upload_retry_delay)
                                else:
                                    bt.logging.error(
                                        "Maximum Retry Limit Reached. Unable To Upload Model To HF Hub."
                                    )
                                    raise

                else:
                    bt.logging.info("Averaging Failed. Loading Latest Model State.")
                    failed_gradient_all_reduce = True
                    load_state_from_peer(self)

                if failed_gradient_all_reduce:
                    gradient_averaging_step.cancel()
                    bt.logging.info("Gradient Step Cancelled")
                    with self.grad_averager.use_averaged_gradients():
                        self.opt.zero_grad()
                    bt.logging.info("Optimizer Gradients Zeroed")

            # Process the Train query responses
            else:
                bt.logging.info(
                    "Received Responses: "
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
                            if (
                                (response.dendrite.status_code == 200)
                                and (response.dataset_indices is not None)
                                and (type(response.dataset_indices) == list)
                            )
                        ]
                    )
                )
                self.average_loss = np.array(
                    [
                        response.loss
                        for response, uid in zip(responses[0], self.miner_uids)
                        if response.dendrite.status_code == 200
                        and (response.dataset_indices is not None)
                    ]
                ).mean()
                bt.logging.info(f"Current Average Miner Loss: {self.average_loss}")

    else:
        bt.logging.info(
            f"Waiting {self.all_reduce_timeout + 30} seconds whilst master UID completes all reduce."
        )
        time.sleep(self.all_reduce_timeout + 30)
        self.miner_uids = []
        responses = [[]]

    # Adjust the scores based on responses from miners.
    rewards = await get_rewards(
        self,
        uids=self.miner_uids,
        responses=responses,
        all_reduce=all_reduce,
    )

    # Normalise Rewards
    if rewards.sum() != 0:
        rewards = rewards / rewards.sum()
    bt.logging.info(f"Final Scores: {rewards}")

    if not all_reduce:
        # Update the tracker based on the rewards
        self.update_local_tracker_state(rewards, responses)

    self.event.update(
        {
            "local_samples_accumulated": self.local_progress.samples_accumulated,
            "local_epoch": self.local_progress.epoch,
            "global_samples_accumulated": self.global_progress.samples_accumulated,
            "global_epoch": self.global_progress.epoch,
            "average_miner_loss": self.average_loss,
            "learning_rate": self.learning_rate,
        }
    )

    # Update the scores based on the rewards.
    self.update_scores(rewards.detach().cpu().numpy(), self.miner_uids)

    self.event.update(self.get_validator_info())

    try:
        self.event.update(get_bandwidth())
    except:
        bt.logging.info("Error getting bandwidth metrics")

    return responses
