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
import torch
from huggingface_hub import list_repo_refs
from huggingface_hub.utils import HfHubHTTPError

from distributed_training.utils.misc import get_bandwidth
from distributed_training.utils.progress_tracker import update_global_tracker_state
from distributed_training.utils.state_loader import (
    load_state_from_peer,
)
from distributed_training.utils.uids import get_random_uids
from distributed_training.validator.reward import get_rewards


async def forward(self):
    """
    The forward function is called by the validator every time step.

    It is responsible for querying the network and scoring the responses.

    Args:
        self (:obj:`bittensor.neuron.Neuron`): The neuron object which contains all the necessary state for the validator.

    """

    update_global_tracker_state(self)
    if self.local_progress.epoch != self.global_progress.epoch:
        bt.logging.info(
            f"Local Epoch {self.local_progress.epoch} Behind Global Epoch {self.global_progress.epoch}. Loading Latest Model State."
        )
        load_state_from_peer(self, epoch=self.global_progress.epoch)

    # Evaluate wether to run an AllReduce or validate HF miner states
    blocks_since_allreduce = self.block - self.last_allreduce_block
    should_allreduce = blocks_since_allreduce >= self.config.neuron.blocks_per_allreduce

    responses = [[]]
    rewards = torch.zeros(len(self.miner_uids)) if self.miner_uids else torch.zeros(0)
    hf_miner_states = {}

    if should_allreduce:
        self.event.update({"synapse_type": "all_reduce"})
        all_reduce = True

        if self.uid == self.master_uid:
            # Master validator coordinates AllReduce and queries miners
            try:
                sample_size = int(self.metagraph.n)

                # Get active miners
                while len(self.miner_uids) < (self.config.neuron.min_group_size - 1):
                    bt.logging.info(
                        f"Found {len(self.miner_uids)} UIDs. Attempting to find {self.config.neuron.min_group_size - len(self.miner_uids)} more UIDs."
                    )
                    self.miner_uids = await get_random_uids(
                        self,
                        dendrite=self.dendrite,
                        k=sample_size,
                        epoch=self.local_progress.epoch if all_reduce else None,
                    )

                # Run AllReduce as master
                results = await self.avg_handler.run_validator_allreduce(
                    timeout=self.config.neuron.allreduce_timeout,
                    learning_rate=self.learning_rate,
                    peerids_to_uids=self.peerids_to_uids,
                )

                if results:
                    # Update scoring based on allreduce participation
                    self.allreduce_scores = (
                        self.avg_handler.calculate_allreduce_scores(
                            results["participating_peers"],
                            results["failed_peers"],
                            self.peerids_to_uids,
                        )
                    )

                    # Update state after successful allreduce
                    self.local_progress.epoch += 1
                    self.local_progress.samples_accumulated = 0

                    # Upload new state
                    await self._upload_new_state(
                        epoch=self.local_progress.epoch,
                        batch_size=sum(results["gathered"].values()),
                        results=results,
                    )

            except Exception as e:
                bt.logging.error(f"AllReduce failed: {e}")
                await load_state_from_peer(self)

        else:
            # Non-master validators participate in AllReduce to help spread the load and update local model
            try:
                (
                    results,
                ) = await self.gradient_processor.run_validator_allreduce(
                    timeout=self.config.neuron.allreduce_timeout,
                    learning_rate=self.learning_rate,
                    peerids_to_uids=self.peerids_to_uids,
                )

                if results:
                    # Calculate scores even as non-master
                    self.allreduce_scores = self.scoring.calculate_allreduce_scores(
                        results["participating_peers"],
                        results["failed_peers"],
                        self.peerids_to_uids,
                    )

            except Exception as e:
                bt.logging.error(f"AllReduce failed: {e}")
                await load_state_from_peer(self)

    else:
        # If running HF validation round, only call the sample_size
        sample_size = self.config.neuron.sample_size
        all_reduce = False
        self.event.update({"synapse_type": "train"})

        self.miner_uids = await get_random_uids(
            self,
            dendrite=self.dendrite,
            k=sample_size,
            epoch=self.local_progress.epoch if all_reduce else None,
        )
        if len(self.miner_uids) == 0:
            bt.logging.info("No Active Miners Found This Step.")
            return responses  # Early return if no active miners found

        self.event.update({"UIDs": self.miner_uids})
        bt.logging.info(f"UIDs:  {self.miner_uids}")

        # Check if miners uploaded new state since last N blocks
        # TODO: Implement HF state check
        # * Below is placeholder for now 
        
        try:
            repo_refs = list_repo_refs(self.config.neuron.model_name, repo_type="model")
            for uid in range(int(self.metagraph.n)):
                hf_miner_states[str(uid)] = await self._check_miner_hf_state(
                    uid, repo_refs
                )
        except HfHubHTTPError as e:
            bt.logging.error(f"Error checking HF states: {e}")
            hf_miner_states = {
                str(uid): {"updated": False} for uid in range(int(self.metagraph.n))
            }

    # Adjust the scores based on responses from miners.
    rewards = await get_rewards(
        self,
        uids=self.miner_uids,
        miner_states=hf_miner_states,
        all_reduce=all_reduce,
    )

    # Normalize rewards if any exist
    if len(rewards) > 0 and rewards.sum() != 0:
        rewards = rewards / rewards.sum()
    bt.logging.info(f"Final Scores: {rewards}")

    if not all_reduce:
        # Update the tracker based on the rewards
        self.update_local_tracker_state(rewards, responses)

    self.event.update(
        {
            "learning_rate": self.learning_rate,
            "average_miner_loss": self.average_loss,
            "local_epoch": self.local_progress.epoch,
            "global_epoch": self.global_progress.epoch,
            "local_samples_accumulated": self.local_progress.samples_accumulated,
            "global_samples_accumulated": self.global_progress.samples_accumulated,
        }
    )

    # Update scores
    if len(rewards) > 0:
        self.update_scores(rewards.detach().cpu().numpy(), self.miner_uids)

    self.event.update(self.get_validator_info())

    try:
        self.event.update(get_bandwidth())
    except Exception:
        bt.logging.info("Error getting bandwidth metrics")

    return responses
