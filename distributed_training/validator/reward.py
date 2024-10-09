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


def score_gradients(self, response, uid):
    # Create Dataloader
    dataloader = DataLoader(
        batch_size=self.config.neuron.local_batch_size_train,
        sequence_length=1024,
        rows=response.dataset_indices,
    )

    index = 0
    # Train data for on last indices
    for index, batch in enumerate(dataloader):
        continue

    if index == 0:
        score = 0
        return score

    # Extract inputs and labels
    inputs = batch[0].to(self.device)
    labels = batch[1].to(self.device)

    # Zero Gradients
    self.opt.zero_grad()

    # Forward pass
    outputs = self.model(input_ids=inputs, labels=labels)

    loss = outputs[1]

    # Backward Pass
    loss.backward()

    # Accumulate Gradients
    self.grad_averager.accumulate_grads_(batch_size=len(inputs))

    # Copy gradients
    gradients = tuple(
        (
            param.grad.detach().cpu().clone()
            if param.grad is not None
            else torch.zeros_like(param)
        )
        for param in self.model.parameters()
    )

    if response.gradient_test_index > len(gradients):
        bt.logging.info(
            f"UID {uid} running incorrect model. Assigning it a gradients core of 0."
        )
        score = 0
        return score
    else:
        # Store summed random gradients in the synapse
        gradients = float(torch.sum(torch.abs(gradients[response.gradient_test_index])))

        bt.logging.info(
            f"Local Validator Sum of Layer {response.gradient_test_index}'s Gradients are: {gradients}"
        )
        bt.logging.info(
            f"UID {uid} Sum of Layer {response.gradient_test_index}'s Gradients are: {response.gradients}"
        )

        # TODO Address issue where gradient sum is negative
        score = 1 - (abs(gradients - response.gradients))

        return score


async def score_blacklist(self, uids):
    scores = torch.FloatTensor([1 for _ in uids]).to(self.device)
    for i, uid in enumerate(uids):
        if self.uids_to_peerids[uid] == None:
            scores[i] = 0.0
        else:
            scores[i] = 1.0

    return scores


async def score_bandwidth(self, uids, timeout=90):
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

def score_failed_senders(self, uids, failed_senders, participating_peers):
    scores = torch.FloatTensor([0.0 for _ in uids]).to(self.device)
    for i, uid in enumerate(uids):
        peer_id = self.uids_to_peerids.get(uid)
        
        if peer_id in participating_peers:
            if peer_id in failed_senders:
                bt.logging.info(f"- Scoring UID {uid} 0.0 (Failed sender)")
                scores[i] = 0.0
            else:
                bt.logging.info(f"- Scoring UID {uid} 1.0 (Successful participating peer)")
                scores[i] = 1.0
        else:
            bt.logging.info(f"- Scoring UID {uid} 0.0 (Not a participating peer)")
            scores[i] = 0.0
    
    return scores

async def get_rewards(
    self,
    uids: List[int],
    responses: list,
    all_reduce: bool,
    failed_senders=None,
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
        # Now that we've called all_reduce on all available UIDs, only score a sample of them to spread
        # the scoring burden across all validators
        self.miner_uids = await get_random_uids(
            self, dendrite=self.dendrite, k=self.config.neuron.sample_size
        )

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

        # Score miners bandwidth
        bandwidth_scores = await score_bandwidth(self, self.miner_uids.tolist())
        bt.logging.info(f"Bandwidth Scores: {bandwidth_scores}")
        self.event.update(
            {
                f"rewards.bandwidth_scores.uid{uid}": bandwidth_score
                for uid, bandwidth_score in zip(uids, bandwidth_scores)
            }
        )
        scores *= bandwidth_scores

        # Apply penalty to failed senders if any
        failed_sender_scores = score_failed_senders(self, uids.tolist(), failed_senders, participating_peers)
        bt.logging.info(f"Failed Sender Scores: {failed_sender_scores}")
        self.event.update(
            {
                f"rewards.failed_sender_score.uid{uid}": failed_sender_score
                for uid, failed_sender_score in zip(uids, failed_sender_scores)
            }
        )
        scores *= failed_sender_scores

    # Score an empty responses
    elif (responses == [[]]) or (
        [
            response.gradients
            for response in responses[0]
            if (response.dendrite.status_code == 200)
            and (response.gradients is not None)
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
                    and (response.gradients is not None)
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
                    len(response.dataset_indices)
                    if (response.dendrite.status_code == 200)
                    and (response.dataset_indices is not None)
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
