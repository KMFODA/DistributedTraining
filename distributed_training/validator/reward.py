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

from typing import List

import bittensor as bt
import torch
from distributed_training.data.dataset import DataLoader
from distributed_training.utils.uids import get_random_uids, map_uid_to_peerid
import time
import asyncio
import random
import numpy as np

# GPU optimizations.
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Seeds
torch.manual_seed(42)
torch.cuda.manual_seed(42)


def score_gradients(self, response, uid, threshold=0.85):
    try:
        if "gradient_sums" not in response.__dict__:
            bt.logging.info(
                f"Gradient score = 0. Gradient sums not found in UID{uid}'s response"
            )

            score_sum = 0

            return score_sum

        else:
            # Create DataLoader
            dataloader = DataLoader(
                batch_size=response.batch_size
                if "batch_size" in response.__dict__
                else self.config.neuron.local_batch_size_train,
                sequence_length=1024,
                rows=response.dataset_indices,
            )

            num_checks = 10
            seed = random.randint(0, 2**32 - 1)
            checkpoint_rng = random.Random(seed)
            checkpoint_indices = sorted(
                checkpoint_rng.sample(range(len(dataloader)), num_checks)
            )

            if len(dataloader) != len(response.gradient_sums):
                bt.logging.info(
                    f"Gradient score = 0. Local dataloader length is {len(dataloader)}. UID{uid}'s gradinet_sums list length is {len(response.gradient_sums)}."
                )

                score_sum = 0

                return score_sum

            checkpoint_indices_set = set(checkpoint_indices)
            validator_gradient_sums = []
            collected_indices = set()

            target_param = list(self.model.parameters())[response.gradient_test_index]

            # Process data at checkpoint indices
            for index, batch in enumerate(dataloader):
                if index in checkpoint_indices_set:
                    # Extract inputs and labels
                    inputs = batch[0].to(self.device)
                    labels = batch[1].to(self.device)

                    # Zero Gradients
                    self.opt.zero_grad()

                    # Forward pass
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        outputs = self.model(input_ids=inputs, labels=labels)
                        loss = outputs[1]

                    # Backward Pass
                    loss.backward()

                    # Extract gradient for the test_layer_index
                    gradient = target_param.grad.detach()

                    validator_gradient_sums.append(
                        torch.sum(torch.abs(gradient)).item()
                    )

                    collected_indices.add(index)
                    if len(collected_indices) == len(checkpoint_indices):
                        break  # All required checkpoints have been processed

            if response.gradient_test_index >= len(gradient):
                bt.logging.info(
                    f"UID {uid} running incorrect model. Assigning it a gradient score of 0."
                )
                score = 0
                return score

            # Extract miner's projected gradients and gradient sums at checkpoint indices
            miner_gradient_sums = [
                response.gradient_sums[idx] for idx in checkpoint_indices
            ]

            bt.logging.info(
                f"Local Validator Gradient Sums at checkpoints: {validator_gradient_sums}"
            )
            bt.logging.info(
                f"UID {uid} Gradient Sums at checkpoints: {miner_gradient_sums}"
            )

            # Compute the differences between the miner's and validator's gradient sums
            differences = [
                abs(m - v) for m, v in zip(miner_gradient_sums, validator_gradient_sums)
            ]

            # Compute relative differences
            relative_diffs = [
                diff / max(abs(m), abs(v), 1e-8)
                for diff, m, v in zip(
                    differences, miner_gradient_sums, validator_gradient_sums
                )
            ]

            # Compute average relative difference
            average_relative_diff = sum(relative_diffs) / len(relative_diffs)

            # Normalize score between 0 and 1 for gradient sum method
            score_sum = max(0.0, 1.0 - average_relative_diff)

            # Zero out scores below threshold
            if score_sum < threshold:
                bt.logging.info(
                    f"Gradient score = {score_sum} for UID {uid} below threshold of {threshold}. Setting to score to 0."
                )
                score_sum = 0

            return score_sum
    except Exception as e:
        bt.logging.info(
            f"Gradient score = 0. Gradient scoring failed for UID {uid} with error {e}."
        )
        score_sum = 0

        return score_sum


async def score_blacklist(self, uids):
    scores = torch.FloatTensor([1 for _ in uids]).to(self.device)
    for i, uid in enumerate(uids):
        if self.uids_to_peerids[uid] == None:
            scores[i] = 0.0
        else:
            scores[i] = 1.0

    return scores


async def score_bandwidth(self, uids, timeout=30):
    scores = torch.FloatTensor([1 for _ in uids]).to(self.device)
    for i, uid in enumerate(uids):
        peer = self.uids_to_peerids[uid]

        if peer is None:
            scores[i] = 0

        else:
            try:
                start_time = time.perf_counter()

                metadata, tensors = await asyncio.wait_for(
                    self.load_state_from_miner(peer), timeout=timeout
                )
                end_time = time.perf_counter()

                if (metadata is None) or (tensors is None):
                    scores[i] = 0
                else:
                    scores[i] = 1 - ((end_time - start_time) / timeout)

                bt.logging.info(f"Reward for peer {peer} is {scores[i]}")

            except Exception as e:
                bt.logging.info(f"Failed to download state from {peer} - {repr(e)}")
                scores[i] = 0
                bt.logging.info(f"Reward for peer {peer} is {scores[i]}")

    return scores


def score_failed_senders(self, uids, failed_peers, participating_peers):
    scores = torch.FloatTensor([0.0 for _ in uids]).to(self.device)
    for i, uid in enumerate(uids):
        peer_id = self.uids_to_peerids.get(uid)

        if peer_id in participating_peers:
            if peer_id in failed_peers:
                bt.logging.info(f"UID:{uid} - Failed participating peer")
                scores[i] = 0.0
            else:
                bt.logging.info(f"UID:{uid} - Successful participating peer")
                scores[i] = 1.0
        else:
            bt.logging.info(f"UID:{uid} - Non participating peer")
            scores[i] = 0.0

    return scores


async def get_rewards(
    self,
    uids: List[int],
    responses: list,
    all_reduce: bool,
    failed_peers=None,
    participating_peers=None,
) -> torch.FloatTensor:
    """
    Returns a tensor of rewards for the given query and responses.

    Args:
    - uids (List[int]): A list of uids that were queried.
    - responses (List): A list of all the responses from the queried uids.
    - all_reduce (bool): A boolean representing wether the all_reduce synapse was called.
    - responses (List[float]): A list of responses from the miners.

    Returns:
    - torch.FloatTensor: A tensor of rewards for the given query and responses.
    """
    # Score a non-empty AllReduce response
    if all_reduce and ((responses != [[]]) or (self.uid != self.master_uid)):
        if self.uid != self.master_uid:
            # Now that we've called all_reduce on all available UIDs, only score a sample of them to spread
            # the scoring burden across all validators
            self.miner_uids = await get_random_uids(
                self, dendrite=self.dendrite, k=self.config.neuron.sample_size
            )
            self.event.update({"uids": self.miner_uids})
            bt.logging.info(f"UIDs:  {self.miner_uids}")

        # Set up the scores tensor
        scores = torch.FloatTensor([1 for _ in self.miner_uids]).to(self.device)

        # Update mapping of uids to peerids
        self.uids_to_peerids = await map_uid_to_peerid(self, range(0, self.metagraph.n))
        self.uids_to_peerids[self.uid] = self.dht.peer_id
        bt.logging.info(f"UID To PeerID Mapping: {self.uids_to_peerids}")

        # Check if peer is connected to DHT & run_id and blacklist them if they are not
        blacklist_scores = await score_blacklist(self, self.miner_uids.tolist())
        bt.logging.info(f"DHT Blacklist Scores: {blacklist_scores}")
        self.event.update(
            {
                f"rewards.blacklist.uid{uid}": blacklist_score
                for uid, blacklist_score in zip(uids, blacklist_scores)
            }
        )
        scores *= blacklist_scores

        if self.uid == self.master_uid:
            # Apply penalty to failed senders if any
            failed_sender_scores = score_failed_senders(
                self, self.miner_uids.tolist(), failed_peers, participating_peers
            )
            bt.logging.info(f"Failed Sender Scores: {failed_sender_scores}")
            self.event.update(
                {
                    f"rewards.failed_sender_score.uid{uid}": failed_sender_score
                    for uid, failed_sender_score in zip(uids, failed_sender_scores)
                }
            )
            scores *= failed_sender_scores
        else:
            # Score miners bandwidth
            bandwidth_scores = await score_bandwidth(
                self, self.miner_uids.tolist(), self.load_state_timeout
            )
            bt.logging.info(f"Bandwidth Scores: {bandwidth_scores}")
            self.event.update(
                {
                    f"rewards.bandwidth_scores.uid{uid}": bandwidth_score
                    for uid, bandwidth_score in zip(
                        self.miner_uids.tolist(), bandwidth_scores
                    )
                }
            )
            scores *= bandwidth_scores

    # Score an empty responses
    elif (responses == [[]]) or (
        [
            response.gradient_sums
            for response in responses[0]
            if (response.dendrite.status_code == 200)
            and (response.gradient_sums is not None)
        ]
        == []
    ):
        scores = torch.FloatTensor([0 for _ in uids]).to(self.device)

    # Score a non-empty Train response
    else:
        scores = torch.FloatTensor(
            [
                1
                if response.dendrite.status_code == 200 and response.loss != 0.0
                else 0
                for _, response in zip(uids, responses[0])
            ]
        ).to(self.device)
        bt.logging.info(f"Timeout Scores: {scores}")

        # Periodically check if peer is connected to DHT & run_id and blacklist them if they are not
        if (self.step % 1) == 0:
            # Update mapping of uids to peerids
            self.uids_to_peerids = await map_uid_to_peerid(
                self, range(0, self.metagraph.n)
            )
            self.uids_to_peerids[self.uid] = self.dht.peer_id
            bt.logging.info(f"UID To PeerID Mapping: {self.uids_to_peerids}")

            # Check if peer is connected to DHT & run_id and blacklist them if they are not
            blacklist_scores = await score_blacklist(self, uids.tolist())
            bt.logging.info(f"DHT Blacklist Scores: {blacklist_scores}")
            self.event.update(
                {
                    f"rewards.blacklist.uid{uid}": blacklist_score
                    for uid, blacklist_score in zip(uids, blacklist_scores)
                }
            )
            scores *= blacklist_scores

        # Re-calculate gradients and score the difference between local gradients and the miner's gradients
        gradient_scores = torch.FloatTensor(
            [
                (
                    score_gradients(self, response, uids.tolist()[index])
                    if (response.dendrite.status_code == 200)
                    and (response.gradient_sums is not None)
                    else 0
                )
                for index, response in enumerate(responses[0])
            ]
        ).to(self.device)
        bt.logging.info(f"Gradient Scores: {gradient_scores}")
        self.event.update(
            {
                f"rewards.gradient.uid{uid}": gradient_score
                for uid, gradient_score in zip(uids.tolist(), gradient_scores)
            }
        )
        scores *= gradient_scores

        # Score miners based off the size of the data they have trained on this step
        steps_scores = torch.FloatTensor(
            [
                (
                    len(set(response.dataset_indices))
                    if (
                        (response.dendrite.status_code == 200)
                        and (response.dataset_indices is not None)
                        and (type(response.dataset_indices) == list)
                    )
                    else 0
                )
                for index, response in enumerate(responses[0])
            ]
        ).to(self.device)
        bt.logging.info(f"Steps Scores: {steps_scores}")
        self.event.update(
            {
                f"rewards.steps.uid{uid}": steps_score
                for uid, steps_score in zip(uids.tolist(), steps_scores)
            }
        )
        scores *= steps_scores

    return scores
