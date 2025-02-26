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
from hivemind.p2p import PeerID
from huggingface_hub import list_repo_commits
from transformers import AutoModelForCausalLM

from distributed_training.data.dataset import DatasetLoader
from distributed_training.utils.state_loader import (
    check_model_exists,
    cleanup_old_cache,
)
from distributed_training.utils.progress_tracker import get_local_epoch

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


async def fetch_training_data(self, block, miner_uid):
    """Async function to fetch training data"""

    try:
        pages = await DatasetLoader.next_pages(
            offset=block,
            n_pages=5,
            seed=miner_uid,
        )
        random.seed(miner_uid)
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


async def score_uid(self, uid: int):
    """Score a single UID"""
    target_blocks = self.config.neuron.target_n_blocks
    latest_commit = None
    blocks = []
    time_delta = 0

    if self.uid_tracker[uid]["model_huggingface_id"] is None:
        bt.logging.info(f"Score 0 for UID {uid}: HuggingFace Repo Id is None")
        scores = 0

    elif not check_model_exists(self.uid_tracker[uid]["model_huggingface_id"]):
        bt.logging.info(
            f"Score 0 for UID {uid}: HuggingFace Repo Id {self.uid_tracker[uid]['model_huggingface_id']} Doesn't Exist"
        )
        scores = 0

    else:
        cleanup_old_cache(
            self,
            repo_id=self.uid_tracker[uid]["model_huggingface_id"],
            current_revision=None,
        )

        commits = list_repo_commits(
            self.uid_tracker[uid]["model_huggingface_id"], repo_type="model"
        )[:2]
        latest_commit = commits[0].commit_id
        time_delta = (commits[0].created_at - commits[1].created_at).seconds

        model_huggingface_id = self.uid_tracker[uid]["model_huggingface_id"]

        self.model = AutoModelForCausalLM.from_pretrained(
            model_huggingface_id, revision=commits[0].commit_id, trust_remote_code=True
        )
        # Move the model to the appropriate device
        self.model = self.model.to(self.device)

        model_final = AutoModelForCausalLM.from_pretrained(
            model_huggingface_id, revision=commits[1].commit_id, trust_remote_code=True
        )

        self.local_progress.samples_accumulated = 0
        self.local_progress.inner_step = (
            self.model.config.inner_step
            if "inner_step" in self.model.config.__dict__
            else 0
        )
        self.local_progress.epoch = get_local_epoch(self, model_huggingface_id)

        if self.local_progress.epoch != self.global_progress.epoch:
            bt.logging.info(
                f"Score 0 for UID {uid}: Local Epoch {self.local_progress.epoch} != Global Epoch {self.global_progress.epoch}"
            )
            scores = 0
        elif ("block_list" in self.model.config.__dict__) and (
            len(self.model.config.block_list) > target_blocks
        ):
            scores = 0
            bt.logging.info(
                f"Score 0 for UID {uid}: Block List Length {len(self.model.config.block_list)} > Target Blocks {target_blocks}"
            )
        else:
            blocks = model_final.config.block_list
            try:
                for block in blocks:
                    bt.logging.info(":pages: Fetching fineweb-edu pages")
                    dataset = await fetch_training_data(self, block, uid)

                    for inputs, labels in dataset:
                        # Move to device
                        inputs, labels = inputs.to(self.device), labels.to(self.device)

                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            _, loss = self.model(input_ids=inputs, labels=labels)
                            scaled_loss = loss / self.number_of_local_steps

                        scaled_loss.backward()

                        self.running_loss += loss.item()
                        self.batch_count += 1
                        self.local_progress.loss = self.running_loss / self.batch_count

                        self.local_progress.samples_accumulated += self.local_batch_size_train
                        
                        if (
                            self.local_progress.samples_accumulated
                            % (self.logging_interval * self.local_batch_size_train)
                            == 0
                        ):
                            bt.logging.info(
                                f":training:  Outer Step: {self.local_progress.epoch} | "
                                f"Inner Step: {self.local_progress.inner_step} | "
                                f"Average Loss: {self.local_progress.loss:.4f} | "
                                f"Micro Batches: [{self.local_progress.samples_accumulated}/{self.local_batch_size_train_effective}]"
                            )

                        # Check if we've accumulated enough samples for a step
                        if (
                            self.local_progress.samples_accumulated
                            >= self.local_batch_size_train_effective
                        ):
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                            self.inner_optimizer.step()
                            self.inner_optimizer.zero_grad()
                            self.local_progress.inner_step += 1
                            self.local_progress.samples_accumulated = 0
                            self.running_loss = 0.0
                            self.batch_count = 0

            except Exception:
                bt.logging.error("Forward Loop Failed, Falling Back To Full Reward")
                return torch.tensor([1.0])

        cleanup_old_cache(
            self,
            repo_id=model_huggingface_id,
            current_revision=None,
        )

        try:
            scores = score_models(self.model, model_final)
        except Exception as e:
            bt.logging.error(f"Error calculating final score: {str(e)}")
            scores = 1.0

    self.uid_tracker[uid]["last_commit"] = latest_commit
    self.uid_tracker[uid]["train_number_of_blocks"] += len(blocks)
    self.uid_tracker[uid]["train_duration"] += time_delta
    self.uid_tracker[uid]["train_similarity_score_last_updated"] = time.time()
    self.uid_tracker[uid]["train_similarity_score"] += scores
    self.uid_tracker[uid]["train_validation_count"] += 1

    rewards = get_normalised_score(self, uid, target_blocks)
    bt.logging.info(f"UID {uid} Train Score: {rewards}")

    return rewards


def get_normalised_score(self, uid, target_blocks):
    miner_uid_tracker = self.uid_tracker[uid]

    if miner_uid_tracker["train_duration"] != 0:
        train_score = (
            miner_uid_tracker["train_similarity_score"]
            * min(miner_uid_tracker["train_number_of_blocks"], target_blocks)
        ) / miner_uid_tracker["train_duration"]
        miner_uid_tracker["train_score"] = train_score

    if miner_uid_tracker["all_reduce_counts"] != 0:
        miner_uid_tracker["all_reduce_score"] = (
            miner_uid_tracker["all_reduce_successes"]
            / miner_uid_tracker["all_reduce_counts"]
        )

    miner_uid_tracker["total_score"] = (
        0.5 * miner_uid_tracker["train_score"]
        + 0.5 * miner_uid_tracker["all_reduce_score"]
    )

    # Normalise all total_scores after updating uid specific total_score
    all_scores = [self.uid_tracker[u]["total_score"] for u in self.uid_tracker]
    scores = torch.nn.functional.normalize(torch.tensor(all_scores), dim=0)

    return torch.tensor([scores[uid]])


def score_models(model_1, model_2):
    """Calculate the cosine similarity score between two model states"""
    score = 0
    index = 0

    for param_1, param_2 in zip(model_1.parameters(), model_2.parameters()):
        score += (
            F.cosine_similarity(param_1.to("cpu"), param_2.to("cpu"), dim=0)
            .mean()
            .item()
        )
        index += 1

    average_score = score / index
    return average_score
