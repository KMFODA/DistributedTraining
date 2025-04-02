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
import errno
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

from datetime import datetime
import pytz
from huggingface_hub import list_repo_commits
from transformers import AutoConfig
from huggingface_hub import hf_hub_download, list_repo_files
import json

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
    model_huggingface_id = self.uid_tracker[uid]["model_huggingface_id"]
    local_epoch = get_local_epoch(self, model_huggingface_id)
    accepted_files = [
        ".gitattributes",
        "config.json",
        "model.safetensors",
        "inner_optimizer.pt",
        "inner_optimizer.npz",
        "outer_optimizer.pt",
        "outer_optimizer.npz",
    ]
    blocks = []
    time_delta = 0

    try:
        if model_huggingface_id is None:
            scores = 0
            raise Exception(f"Score 0 for UID {uid}: HuggingFace Repo Id is None")
        elif not check_model_exists(model_huggingface_id):
            scores = 0
            raise Exception(
                f"Score 0 for UID {uid}: HuggingFace Repo Id {self.uid_tracker[uid]['model_huggingface_id']} Doesn't Exist"
            )
        elif (local_epoch is None) or (local_epoch != self.global_progress.epoch):
            scores = 0
            raise Exception(
                f"Score 0 for UID {uid}: Local Epoch {local_epoch} != Global Epoch {self.global_progress.epoch}"
            )

        cleanup_old_cache(
            self,
            repo_id=model_huggingface_id,
            current_revision=None,
        )

        commits = list_repo_commits(model_huggingface_id, repo_type="model")[:2]
        latest_commit = commits[0].commit_id
        time_delta = (commits[0].created_at - commits[1].created_at).seconds

        # Check current model configs haven't been altered
        config_path = hf_hub_download(
            repo_id=model_huggingface_id,
            filename="config.json",
            revision=commits[0].commit_id,
        )
        with open(config_path) as config_data:
            config = json.load(config_data)

        if config["auto_map"] != self.global_model_config.auto_map:
            raise Exception(
                f"Score 0 for UID {uid}: Commit {commits[0].commit_id} config differs from the global model config"
            )

        for file in list_repo_files(
            repo_id=model_huggingface_id, revision=commits[0].commit_id
        ):
            if file not in accepted_files:
                raise Exception(
                    f"Score 0 for UID {uid}: File {file} for commi {commits[0].commit_id} not in list of accepted files {accepted_files}"
                )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_huggingface_id, revision=commits[0].commit_id, trust_remote_code=True
        )
        # Move the model to the appropriate device
        self.model = self.model.to(self.device)

        config_final_path = hf_hub_download(
            repo_id=model_huggingface_id,
            filename="config.json",
            revision=commits[1].commit_id,
        )
        with open(config_final_path) as config_final_data:
            config_final = json.load(config_final_data)

        if config_final["auto_map"] != self.global_model_config.auto_map:
            raise Exception(
                f"Score 0 for UID {uid}: Commit {commits[1].commit_id} config differs from the global model config"
            )

        for file in list_repo_files(
            repo_id=model_huggingface_id, revision=commits[1].commit_id
        ):
            if file not in accepted_files:
                raise Exception(
                    f"Score 0 for UID {uid}: File {file} for commi {commits[1].commit_id} not in list of accepted files {accepted_files}"
                )

        model_final = AutoModelForCausalLM.from_pretrained(
            model_huggingface_id, revision=commits[1].commit_id, trust_remote_code=True
        )

        self.local_progress.samples_accumulated = 0
        self.local_progress.inner_step = (
            self.model.config.inner_step
            if "inner_step" in self.model.config.__dict__
            else 0
        )
        # Only set self.local_progress.epoch if model is correct format
        self.local_progress.epoch = get_local_epoch(self, model_huggingface_id)

        if ("block_list" in self.model.config.__dict__) and (
            len(self.model.config.block_list) > target_blocks
        ):
            scores = 0
            bt.logging.info(
                f"Score 0 for UID {uid}: Block List Length {len(self.model.config.block_list)} > Target Blocks {target_blocks}"
            )
        else:
            blocks = model_final.config.block_list
            for block in blocks:
                bt.logging.debug(":pages: Fetching fineweb-edu pages")
                dataset = await fetch_training_data(self, block, uid)

                for inputs, labels in dataset:
                    # Move to device
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        outputs = self.model(input_ids=inputs, labels=labels)
                        loss = outputs[1] / self.number_of_local_steps

                    self.scaler.scale(loss).backward()

                    self.running_loss += loss.item() * self.number_of_local_steps
                    self.batch_count += 1
                    self.local_progress.loss = self.running_loss / self.batch_count

                    self.local_progress.samples_accumulated += (
                        self.local_batch_size_train
                    )

                    # Check if we've accumulated enough samples for a step
                    if (
                        self.local_progress.samples_accumulated
                        >= self.local_batch_size_train_effective
                    ):
                        bt.logging.info(
                            f":training:  Outer Step: {self.local_progress.epoch} | "
                            f"Inner Step: {self.local_progress.inner_step} | "
                            f"Learning Rate: {self.inner_optimizer.param_groups[0]['lr']:.8f} | "
                            f"Average Loss: {self.local_progress.loss:.2f}"
                        )

                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.scaler.unscale_(optimizer=self.inner_optimizer)
                        self.scaler.step(self.inner_optimizer)
                        self.scaler.update()

                        self.scheduler.step()

                        self.inner_optimizer.zero_grad()

                        self.local_progress.inner_step += 1

                        self.running_loss = 0.0
                        self.batch_count = 0

                        self.local_progress.samples_accumulated = 0

            scores = score_models(self.model, model_final)
    except Exception as e:
        # TODO if OSError and e.rrno == errno.ENOSPC score as 1
        scores = 0.0
        bt.logging.info(
            f"Score {int(scores)} for UID {uid}: Forward Loop Failed With Error: {e}"
        )

    finally:
        if model_huggingface_id is not None:
            cleanup_old_cache(
                self,
                repo_id=model_huggingface_id,
                current_revision=None,
            )

    self.uid_tracker[uid]["last_commit"] = latest_commit
    self.uid_tracker[uid]["train_number_of_blocks"] += len(blocks)
    self.uid_tracker[uid]["train_duration"] += time_delta
    self.uid_tracker[uid]["train_similarity_score_last_updated"] = time.time()
    self.uid_tracker[uid]["train_similarity_score"] += scores
    self.uid_tracker[uid]["train_validation_count"] += 1
    self.uid_tracker[uid]["loss"] = self.local_progress.loss

    bt.logging.info(f"UID {uid} Current Train Score: {scores}")
    update_total_score(self, uid, target_blocks)


def update_total_score(self, uid, target_blocks):
    miner_uid_tracker = self.uid_tracker[uid]

    if miner_uid_tracker["train_duration"] != 0:
        miner_uid_tracker["train_score"] = (
            miner_uid_tracker["train_similarity_score"]
            * min(miner_uid_tracker["train_number_of_blocks"], target_blocks)
        ) / miner_uid_tracker["train_duration"]

    if miner_uid_tracker["all_reduce_counts"] != 0:
        miner_uid_tracker["all_reduce_score"] = (
            miner_uid_tracker["all_reduce_successes"]
            / miner_uid_tracker["all_reduce_counts"]
        )

    miner_uid_tracker["total_score"] = (
        0.5 * miner_uid_tracker["train_score"]
        + 0.5 * miner_uid_tracker["all_reduce_score"]
    )

    self.uid_tracker = dict(sorted(self.uid_tracker.items()))


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


def score_repo(self, repo_id: str) -> bool:
    try:
        local_config = AutoConfig.from_pretrained(repo_id, trust_remote_code=True)
        if (
            (self.global_model_config.n_embd != local_config.n_embd)
            or (self.global_model_config.n_head != local_config.n_head)
            or (self.global_model_config.n_layer != local_config.n_layer)
        ):
            return False
        latest_commit = list_repo_commits(repo_id)[0]

        if (datetime.now(pytz.utc) - latest_commit.created_at).seconds > (
            self.config.neuron.target_n_blocks * 60 * 3
        ):
            return False
        return True
    except Exception as e:
        return False


def benchmark_untested_uids(self):
    for uid in self.uid_tracker:
        try:
            if (self.uid_tracker[uid]["train_validation_count"] == 0) and (
                self.uid_tracker[uid]["all_reduce_counts"] == 0
            ):
                self.uid_tracker[uid]["repo_valid_sum"] += score_repo(
                    self, self.uid_tracker[uid]["model_huggingface_id"]
                )
                self.uid_tracker[uid]["repo_valid_count"] += 1
                self.uid_tracker[uid]["repo_valid_score"] = (
                    self.uid_tracker[uid]["repo_valid_sum"]
                    / self.uid_tracker[uid]["repo_valid_count"]
                )
                bt.logging.info(
                    f"UID {uid} identifies as untested. Benchmarking with score {self.uid_tracker[uid]['repo_valid_score']}"
                )
        except Exception as e:
            bt.logging.info(
                f"UID {uid} identifies as untested. Benchmarking failed with error {e}"
            )


def update_all_reduce_scores(self):
    try:
        if self.model.config.all_reduce_scores != {}:
            for uid in self.model.config.all_reduce_scores.keys():
                if self.model.config.all_reduce_scores[uid] == "SUCCESS":
                    self.uid_tracker[int(uid)]["all_reduce_score"] = 1
                else:
                    self.uid_tracker[int(uid)]["all_reduce_score"] = 0

    except Exception as e:
        bt.logging.info(f"Error {e} updating all_reduce scores")


def update_total_scores(self):
    update_all_reduce_scores(self)
    # Sort uid tracker
    self.uid_tracker = dict(sorted(self.uid_tracker.items()))

    # Normalise each type of reward
    train_score_normalised = np.linalg.norm(
        [self.uid_tracker[uid]["train_score"] for uid in self.uid_tracker],
        ord=1,
        axis=0,
        keepdims=True,
    ).item()
    all_reduce_score_normalised = np.linalg.norm(
        [self.uid_tracker[uid]["all_reduce_score"] for uid in self.uid_tracker],
        ord=1,
        axis=0,
        keepdims=True,
    ).item()
    repo_valid_score_normalised = np.linalg.norm(
        [self.uid_tracker[uid]["repo_valid_score"] for uid in self.uid_tracker],
        ord=1,
        axis=0,
        keepdims=True,
    ).item()

    # Check if any norms are zero or contains NaN values
    if np.any(train_score_normalised == 0) or np.isnan(train_score_normalised).any():
        train_score_normalised = np.ones_like(train_score_normalised).item()

    if (
        np.any(all_reduce_score_normalised == 0)
        or np.isnan(all_reduce_score_normalised).any()
    ):
        all_reduce_score_normalised = np.ones_like(all_reduce_score_normalised).item()

    if (
        np.any(repo_valid_score_normalised == 0)
        or np.isnan(repo_valid_score_normalised).any()
    ):
        repo_valid_score_normalised = np.ones_like(repo_valid_score_normalised).item()

    # Update total scores with repo_valid_score if train_score or all_reduce_score are 0
    # Otherwise score using weighted train_score and all_reduce_score
    for uid in self.uid_tracker:
        if (self.uid_tracker[uid]["all_reduce_score"] == 0) and (
            self.uid_tracker[uid]["train_score"] == 0
        ):
            self.uid_tracker[uid]["total_score"] = (
                self.uid_tracker[uid]["repo_valid_score"] / repo_valid_score_normalised
            )
        else:
            self.uid_tracker[uid]["total_score"] = (
                (0.5 * self.uid_tracker[uid]["train_score"]) / train_score_normalised
            ) + (
                (0.5 * self.uid_tracker[uid]["all_reduce_score"])
                / all_reduce_score_normalised
            )
