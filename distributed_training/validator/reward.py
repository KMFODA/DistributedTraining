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
from typing import List

import base58
import bittensor as bt
import numpy as np
import torch
import torch.nn.functional as F
from distributed_training.data.dataset import DatasetLoader
from distributed_training.utils.state_loader import cleanup_old_cache
from distributed_training.utils.uids import (
    get_random_uids,
    map_uid_to_peerid,
    update_run_peerid_list,
)
from hivemind.p2p import PeerID
from huggingface_hub import list_repo_commits
from transformers import AutoModelForCausalLM

# GPU optimizations.
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Seeds
torch.manual_seed(42)
torch.cuda.manual_seed(42)


async def score_blacklist(self, uids):
    scores = torch.FloatTensor([1 for _ in uids]).to(self.device)
    for i, uid in enumerate(uids):
        if self.uids_to_peerids[uid][0] == None:
            scores[i] = 0.0
        elif self.uids_to_peerids[uid][0] in self.run_peer_id_list:
            scores[i] = 1.0
        else:
            scores[i] = 0.0

    return scores


async def score_bandwidth(self, uids, timeout=30):
    scores = torch.FloatTensor([1 for _ in uids]).to(self.device)
    for i, uid in enumerate(uids):
        peer_id = self.uids_to_peerids[uid][0]

        if peer_id is None:
            peer = None
        else:
            peer = PeerID(base58.b58decode(peer_id))

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
        peer_id = self.uids_to_peerids.get(uid)[0]

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

# TODO clean up this file
async def get_rewards(
    self,
    uids: List[int],
    responses: list,
    all_reduce: bool,
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
    # Score an AllReduce response
    if all_reduce:
        # Now that we've called all_reduce on all available UIDs, only score a sample of them to spread
        # the scoring burden across all validators
        self.miner_uids = await get_random_uids(
            self, dendrite=self.dendrite, k=self.config.neuron.sample_size
        )
        self.event.update({"uids": self.miner_uids})
        bt.logging.info(f"UIDs:  {self.miner_uids}")

        # Set up the scores tensor
        scores = torch.FloatTensor([1 for _ in self.miner_uids]).to(self.device)

        # Check if peer is connected to DHT & run_id and blacklist them if they are not
        bt.logging.info(f"UID To PeerID Mapping: {self.uids_to_peerids}")

        # Update UID to PeerID mapping
        map_uid_to_peerid(self, uids)

        # Update PeerIDs list
        update_run_peerid_list(self)

        blacklist_scores = await score_blacklist(self, self.miner_uids)
        bt.logging.info(f"DHT Blacklist Scores: {blacklist_scores}")
        self.event.update(
            {
                f"rewards.blacklist.uid{uid}": blacklist_score
                for uid, blacklist_score in zip(uids, blacklist_scores)
            }
        )
        scores *= blacklist_scores

        # This is done via the all_reduce instead
        # # Score miners bandwidth
        # bandwidth_scores = await score_bandwidth(
        #     self,
        #     self.miner_uids,
        # )
        # bt.logging.info(f"Bandwidth Scores: {bandwidth_scores}")
        # self.event.update(
        #     {
        #         f"rewards.bandwidth_scores.uid{uid}": bandwidth_score
        #         for uid, bandwidth_score in zip(
        #             self.miner_uids.tolist(), bandwidth_scores
        #         )
        #     }
        # )
        # scores *= bandwidth_scores

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
                (
                    1
                    if response.dendrite.status_code == 200 and response.loss != 0.0
                    else 0
                )
                for _, response in zip(uids, responses[0])
            ]
        ).to(self.device)
        bt.logging.info(f"Timeout Scores: {scores}")

        # Check if peer is connected to DHT & run_id and blacklist them if they are not
        bt.logging.info(f"UID To PeerID Mapping: {self.uids_to_peerids}")

        if (self.uid == self.master_uid) and (
            self.local_progress.samples_accumulated == 0
        ):
            indices = random.sample(range(len(uids)), self.config.neuron.sample_size)
            uids = np.array([uids[i] for i in indices])
            responses = [[responses[0][i] for i in indices]]
            self.miner_uids = uids

        # Update UID to PeerID mapping
        map_uid_to_peerid(self, uids)

        # Update PeerIDs list
        update_run_peerid_list(self)

        blacklist_scores = await score_blacklist(self, uids)
        bt.logging.info(f"DHT Blacklist Scores: {blacklist_scores}")
        self.event.update(
            {
                f"rewards.blacklist.uid{uid}": blacklist_score
                for uid, blacklist_score in zip(uids, blacklist_scores)
            }
        )

        # Re-calculate gradients and score the difference between local gradients and the miner's gradients
        gradient_scores = torch.FloatTensor(
            [
                (
                    score_gradients(self, response, uids[index])
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
                for uid, gradient_score in zip(uids, gradient_scores)
            }
        )

        # Score miners based off the size of the data they have trained on this step
        steps_scores = torch.FloatTensor(
            [
                (
                    len(set(response.dataset_indices))
                    if (
                        (response.dendrite.status_code == 200)
                        and (response.dataset_indices is not None)
                        and (type(response.dataset_indices) is list)
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
                for uid, steps_score in zip(uids, steps_scores)
            }
        )
        steps_scores = torch.nn.functional.normalize(steps_scores, dim=0)

        # Score miners based off wether they where succesfull or not in the all_reduce round
        if hasattr(self, "all_reduce_scores"):
            all_reduce_scores = torch.FloatTensor(
                [
                    (
                        1
                        if str(uid) in self.all_reduce_scores
                        and self.model.config.all_reduce_scores[str(uid)] == "SUCCESS"
                        else 0
                    )
                    for uid in uids.tolist()
                ]
            ).to(self.device)
            bt.logging.info(f"All Reduce Scores: {all_reduce_scores}")

            self.event.update(
                {
                    f"rewards.all_reduce.uid{uid}": all_reduce_score
                    for uid, all_reduce_score in zip(uids, all_reduce_scores)
                }
            )

            # Final balanced score calculation with all_reduce
            scores = blacklist_scores * (
                (0.5 * gradient_scores * steps_scores) + (0.5 * all_reduce_scores)
            )

        else:
            # Final balanced score calculation without all_reduce
            scores = blacklist_scores * (gradient_scores * steps_scores)

    return scores


async def fetch_training_data(self, block):
    """Async function to fetch training data"""

    try:
        pages = await DatasetLoader.next_pages(
            offset=block,
            n_pages=5,
            seed=self.uid if not self.config.random else random.randint(0, 1000),
        )
        random.shuffle(pages)

        dataset = await DatasetLoader.create(
            batch_size=self.config.neuron.local_batch_size_train,
            sequence_length=1024,
            pages_info=pages,
            tokenizer=self.tokenizer,
        )

        return dataset
    except Exception as e:
        bt.logging.error(f"Error fetching training data: {str(e)}")
        raise


async def score_uid(self, uid):
    """Score a single UID"""

    if self.uid_metadata_tracker[uid]["model_huggingface_id"] is None:
        return 0

    cleanup_old_cache(
        self,
        repo_id=self.uid_metadata_tracker[uid]["model_huggingface_id"],
        current_revision=None,
    )

    commits = (
        list_repo_commits(
            self.uid_metadata_tracker[uid]["model_huggingface_id"], repo_type="model"
        )[0].commit_id,
        list_repo_commits(
            self.uid_metadata_tracker[uid]["model_huggingface_id"], repo_type="model"
        )[1].commit_id,
    )
    model_huggingface_id = self.uid_metadata_tracker[uid]["model_huggingface_id"]

    self.model = AutoModelForCausalLM.from_pretrained(
        model_huggingface_id, revision=commits[0], trust_remote_code=True
    )
    # Move the model to the appropriate device
    self.model = self.model.to(self.device)

    model_final = AutoModelForCausalLM.from_pretrained(
        model_huggingface_id, revision=commits[1], trust_remote_code=True
    )

    blocks = model_final.config.block_list
    try:
        for block in blocks:
            dataset = await fetch_training_data(self, block)
            total_loss = 0
            batch_count = 0
            inner_step_counter = 0

            for inputs, labels in dataset:
                # Move to device
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs = self.model(input_ids=inputs, labels=labels)
                    loss = outputs[1]

                loss.backward()

                self.local_progress.samples_accumulated += inputs.size(0)
                total_loss += loss.detach().item()
                batch_count += 1
                inner_step_counter += 1

                if batch_count % 5 == 0:
                    bt.logging.info(
                        f":training: Inner Step: {inner_step_counter} | Average Loss: {total_loss / batch_count:.4f}"
                    )

                self.inner_optimizer.step()
                self.inner_optimizer.zero_grad()
                
    except Exception:
        bt.logging.error("Forward Loop Failed, Falling Back To Full Reward")
        return torch.tensor([1.0])


    cleanup_old_cache(
        self,
        repo_id=model_huggingface_id,
        current_revision=None,
    )

    try:
        return score_models(self.model, model_final)
    except Exception as e:
        bt.logging.error(f"Error calculating final score: {str(e)}")
        return torch.tensor([1.0])


def score_models(model_1, model_2):
    score = 0
    index = 0

    for param_1, param_2 in zip(model_1.parameters(), model_2.parameters()):
        score += (
            F.cosine_similarity(param_1.to("cpu"), param_2.to("cpu"), dim=0)
            .mean()
            .item()
        )
        index += 1

    average_score = torch.tensor([score / index])
    return average_score
