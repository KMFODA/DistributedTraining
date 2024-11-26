import os
import re
import copy
import time
import random
import logging
import tempfile
import threading
from itertools import chain
from typing import Any, Dict, Optional, Sequence, Tuple

import bittensor as bt
import hivemind
import torch
from bitsandbytes.optim import LAMB8bit
from hivemind.compression import deserialize_torch_tensor
from hivemind.p2p import PeerID
from hivemind.proto import averaging_pb2
from hivemind.utils import MPFuture, get_logger, nested_pack
from hivemind.utils.asyncio import aiter_with_timeout
from hivemind.utils.streaming import combine_from_streaming
from hivemind.utils.timed_storage import ValueWithExpiration
from huggingface_hub import upload_file, hf_hub_download, scan_cache_dir, create_tag, upload_folder


from transformers import AutoModelForCausalLM

from distributed_training.utils.progress_tracker import (
    LocalTrainingProgress,
    get_global_epoch,
)

logger = get_logger(__name__)
logger.setLevel(logging.INFO)


class DTStateAverager(hivemind.optim.state_averager.TrainingStateAverager):
    def load_state_from_peers_with_latest_state(
        self, global_epoch, wait: bool = True, timeout: Optional[float] = None
    ) -> Optional[Tuple[Any, Sequence[torch.Tensor]]]:
        """
        Try to download the latest optimizer state one of the existing peer.
        :returns: on success, return a 2-tuple with (metadata, tensors), where

        - metadata is a small object containing metadata (e.g. hyperparameters, scalars, etc)
        - tensors is a sequence of pytorch tensors meant to contain peer's model weights and optimizer statistics

        The exact contents of both metadata and tensors are determined by get_current_state method
        """
        future = MPFuture()
        self._outer_pipe.send(
            (
                "_load_state_from_peers_with_latest_state",
                [],
                dict(timeout=timeout, global_epoch=global_epoch, future=future),
            )
        )
        return future.result(timeout=timeout) if wait else future

    async def _load_state_from_peers_with_latest_state(
        self, global_epoch, future: MPFuture, timeout: Optional[float] = None
    ):
        bt.logging.info(f"Timeout = {timeout}")
        if timeout is not None:
            timeout = (
                self.next_chunk_timeout
                if self.next_chunk_timeout is not None
                else self.request_timeout
            )
        try:
            # Extract training progress metadata
            training_progress_metadata, _ = self.dht.get(
                f"{re.sub('_state_averager', '_progress', self.prefix)}", latest=True
            ) or (None, -float("inf"))
            # Get only peers where local_epoch == global_epoch
            if training_progress_metadata is None:
                logger.info(f"Averager could not load metadata from the tracker")
                future.set_result(None)
                return
            else:
                valid_peer_entries = [
                    PeerID(LocalTrainingProgress.parse_obj(peer_state.value).peer_id)
                    for peer_state in training_progress_metadata.values()
                    if (peer_state.value is not None)
                    and (
                        LocalTrainingProgress.parse_obj(peer_state.value).epoch
                        == global_epoch
                    )
                ]

            key_manager = self._matchmaking.group_key_manager
            # prefix = self.state_averager.matchmaking_kwargs['prefix']
            peer_priority, _ = self.dht.get(
                f"{key_manager.prefix}.all_averagers", latest=True
            ) or ({}, None)
            peer_priority = {
                PeerID(peer_id): (
                    float(info.value),
                    random.random(),
                )  # using randomness as a tie breaker
                for peer_id, info in peer_priority.items()
                if isinstance(info, ValueWithExpiration)
                and isinstance(info.value, (float, int))
                and (PeerID(peer_id) in valid_peer_entries)
            }

            if not isinstance(peer_priority, dict) or len(peer_priority) == 0:
                logger.info(
                    f"Averager could not load state from peers: peer dict empty or corrupted {peer_priority}"
                )
                future.set_result(None)
                return

            metadata = None
            for peer in sorted(
                peer_priority.keys(), key=peer_priority.get, reverse=True
            ):
                if peer != self.peer_id:
                    logger.info(f"Downloading parameters from peer {peer}")
                    try:
                        stub = self.get_stub(
                            self._p2p, peer, namespace=key_manager.prefix
                        )
                        stream = await stub.rpc_download_state(
                            averaging_pb2.DownloadRequest()
                        )
                        current_tensor_parts, tensors = [], []

                        # TODO merge this with hivemind.compression.deserialize_tensor_stream
                        async for message in aiter_with_timeout(
                            stream, timeout=timeout
                        ):
                            if message.metadata:
                                metadata = self.serializer.loads(message.metadata)
                            if message.tensor_part.dtype and current_tensor_parts:
                                # tensor_part.dtype indicates the start of the new tensor, so we should wrap up this one
                                tensors.append(
                                    deserialize_torch_tensor(
                                        combine_from_streaming(current_tensor_parts)
                                    )
                                )
                                current_tensor_parts = []
                            current_tensor_parts.append(message.tensor_part)
                        if current_tensor_parts:
                            tensors.append(
                                deserialize_torch_tensor(
                                    combine_from_streaming(current_tensor_parts)
                                )
                            )

                        if not metadata:
                            logger.debug(f"Peer {peer} did not send its state")
                            continue

                        logger.info(f"Finished downloading state from {peer}")
                        future.set_result((metadata, tensors))
                        return
                    except Exception as e:
                        logger.exception(
                            f"Failed to download state from {peer} - {repr(e)}"
                        )

        finally:
            if not future.done():
                future.set_result(None)

    def load_final_state_from_peers(self, global_epoch, **kwargs):
        """
        Attempt to download the latest optimizer state from peers and update trainer parameters/statistics.
        :returns: whether or the averager succeeded in loading parameters
        """
        opt_parameters = tuple(
            param
            for param_group in self.optimizer.param_groups
            for param in param_group["params"]
        )
        main_parameters_and_extras = tuple(chain(opt_parameters, self.extra_tensors))
        num_parameters_and_extras = len(main_parameters_and_extras)

        loaded_state = self.load_state_from_peers_with_latest_state(
            global_epoch, **kwargs
        )
        if loaded_state is None:
            return False

        metadata, flat_tensors = loaded_state
        if (not isinstance(metadata.get("epoch"), int)) or metadata[
            "epoch"
        ] < self.local_epoch:
            logger.warning(
                "Cowardly refusing to load state from peer: peer's epoch is behind our local epoch"
            )
            return False

        loaded_parameters_and_extras = flat_tensors[:num_parameters_and_extras]
        loaded_opt_tensors = flat_tensors[num_parameters_and_extras:]
        if num_parameters_and_extras != len(loaded_parameters_and_extras):
            logger.error(
                "Failed to load state from peer, received parameters, extras or metadata"
            )
            return False

        with torch.no_grad(), self.lock_averaged_tensors:
            try:
                load_optimizer_state(
                    self.optimizer, metadata["optimizer_metadata"], loaded_opt_tensors
                )
            except StopIteration:
                logger.warning(
                    "Failed to load state from peer, received inconsistent number of optimizer statistics"
                )
                return False

            for local_param, loaded_param in zip(
                main_parameters_and_extras, loaded_parameters_and_extras
            ):
                local_param.copy_(loaded_param, non_blocking=True)

        if self.offload_optimizer:
            self._apply_optimizer_parameters_()
        if not self.reuse_tensors:
            self._load_local_tensors_into_averager_()

        self.local_epoch = metadata["epoch"]
        self._update_scheduler()
        return True


def load_optimizer_state(
    optimizer: torch.optim.Optimizer,
    flat_metadata: Dict,
    flat_tensors: Sequence[torch.Tensor],
):
    """Load a state obtained by dump_optimizer_state back into the optimizer"""
    flat_optimizer_state = []
    for elem in flat_metadata:
        if elem.get("type") == "tensor" and isinstance(elem.get("index"), int):
            flat_optimizer_state.append(flat_tensors[elem["index"]])
        elif elem.get("type") == "value" and "value" in elem:
            flat_optimizer_state.append(elem["value"])
    return optimizer.load_state_dict(
        nested_pack(flat_optimizer_state, structure=optimizer.state_dict())
    )


class ModelLoadingManager:
    def __init__(self):
        self.loading_lock = threading.Lock()
        self._is_loading = False
        self._last_loaded_epoch = None

    @property
    def is_loading(self):
        with self.loading_lock:
            return self._is_loading

    @property
    def last_loaded_epoch(self):
        with self.loading_lock:
            return self._last_loaded_epoch

    def set_loading_state(self, is_loading, epoch=None):
        with self.loading_lock:
            self._is_loading = is_loading
            if not is_loading and epoch is not None:
                self._last_loaded_epoch = epoch


def load_state_from_peer(self, epoch=None, keep_recent=5):
    # Skip if we're already loading or if we've already loaded this epoch
    if self.loading_manager.is_loading:
        bt.logging.info("Model loading already in progress, skipping...")
        return False

    if epoch == self.loading_manager.last_loaded_epoch:
        bt.logging.info(f"Already loaded epoch {epoch}, skipping...")
        return False

    # Set loading state
    self.loading_manager.set_loading_state(True, epoch)

    try:
        state_loaded = False
        if epoch == None:
            self.global_progress.epoch = get_global_epoch(self)
            epoch = self.global_progress.epoch

        bt.logging.info("Model Weights Before Loading State")
        current_model_weights_sample = copy.copy(
            [layer for layer in self.model.parameters()][-2][-10:].tolist()
        )
        bt.logging.info(current_model_weights_sample)

        bt.logging.info(f"Old Model Tag: {self.local_progress.epoch}")

        if self.global_progress.epoch is not None:
            bt.logging.info(
                f"Latest Model State Found On The HF Hub With The Tag: {self.global_progress.epoch}. Loading That Model State."
            )

            # Load model state with max retries
            MAX_ATTEMPTS = 3
            attempt = 0
            while attempt < MAX_ATTEMPTS:
                try:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.config.neuron.model_name,
                        revision=str(self.global_progress.epoch),
                        trust_remote_code=True,
                    )
                    # Convert to back to fp32
                    self.model.to(dtype=torch.float32)
                    self.model.to(self.device)

                    # Initialize optimizer with model parameters
                    param_dict = {pn: p for pn, p in self.model.named_parameters()}
                    param_dict = {
                        pn: p for pn, p in param_dict.items() if p.requires_grad
                    }
                    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
                    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
                    optim_groups = [
                        {"params": decay_params, "weight_decay": self.weight_decay},
                        {"params": nodecay_params, "weight_decay": 0.0},
                    ]

                    # Try to load optimizer state if it exists
                    try:
                        optimizer_state = torch.load(
                            hf_hub_download(
                                repo_id=self.config.neuron.model_name,
                                filename="optimizer.pt",
                                revision=str(epoch)
                            )
                        )
                        
                        self.opt = LAMB8bit(
                            optim_groups,
                            lr=optimizer_state["learning_rate"],
                            betas=(0.9, 0.95),
                            eps=1e-8
                        )
                        self.opt.load_state_dict(optimizer_state["optimizer_state_dict"])
                        bt.logging.info(f"Successfully loaded optimizer state for epoch {epoch}")

                    except Exception as e:
                        bt.logging.warning(
                            f"No optimizer state found or failed to load: {str(e)}. Initializing fresh optimizer."
                        )
                        # Initialize fresh optimizer
                        self.opt = LAMB8bit(
                            optim_groups,
                            lr=self.learning_rate_maximum,
                            betas=(0.9, 0.95),
                            eps=1e-8,
                        )

                    break  # Successfully loaded model and created optimizer

                except Exception as e:
                    attempt += 1
                    if attempt == MAX_ATTEMPTS:
                        raise Exception(
                            f"Failed to load model after {MAX_ATTEMPTS} attempts: {str(e)}"
                        )
                    bt.logging.warning(
                        f"Failed to fetch data, retrying. Attempt {attempt}/{MAX_ATTEMPTS}"
                    )

            self.grad_averager.parameters = tuple(self.model.parameters())
            # Reset gradient buffers
            self.grad_averager.reset_accumulated_grads_()
            state_loaded = True

            bt.logging.info("Model Weights After Loading State")
            new_model_weights_sample = copy.copy(
                [layer for layer in self.model.parameters()][-2][-10:].tolist()
            )
            bt.logging.info(new_model_weights_sample)

            self.local_progress.epoch = self.global_progress.epoch
            self.local_progress.samples_accumulated = 0
            bt.logging.info(f"New Model Tag: {self.global_progress.epoch}")

            # Clean up old cache
            try:
                cleanup_old_cache(self, keep_recent)
            except Exception as e:
                bt.logging.warning(f"Failed to cleanup cache: {str(e)}")

        else:
            bt.logging.info(f"Model With Tag: {epoch} Does Not Exist")

        if state_loaded:
            self.loading_manager.set_loading_state(False, epoch)
        else:
            self.loading_manager.set_loading_state(False, None)

        return state_loaded

    except Exception as e:
        bt.logging.error(f"Error loading state: {str(e)}")
        self.loading_manager.set_loading_state(False, None)
        return False


def cleanup_old_cache(self, keep_recent):
    """Helper method to clean up old cache files"""
    current_revision = self.model.config._commit_hash
    cache_info = scan_cache_dir()
    for repo in cache_info.repos:
        if repo.repo_id == self.config.neuron.model_name:
            revisions = sorted(
                repo.revisions, key=lambda r: r.last_modified, reverse=True
            )
            current_index = next(
                (
                    i
                    for i, r in enumerate(revisions)
                    if r.commit_hash == current_revision
                ),
                None,
            )
            if current_index is not None:
                for revision in revisions[max(current_index + 1, keep_recent) :]:
                    cache_info.delete_revisions(revision.commit_hash).execute()
            break


def save_and_upload_state(self, epoch, batch_size, participating_peers, failed_peers):
    """Unified function to save and upload both model and optimizer state"""
    attempt = 0
    while attempt < self.model_upload_retry_limit:
        try:
            with tempfile.TemporaryDirectory() as tmp_folder:
                bt.logging.info(f"Preparing model and optimizer state for epoch {epoch}")
                
                # Save model in fp16 for efficiency
                self.model.to(dtype=torch.float16)
                self.model.save_pretrained(os.path.join(tmp_folder, "model"))
                self.model.to(dtype=torch.float32)
                
                # Save optimizer state
                optimizer_state = {
                    "optimizer_state_dict": self.opt.state_dict(),
                    "learning_rate": self.learning_rate_maximum,
                    "epoch": epoch
                }
                torch.save(optimizer_state, os.path.join(tmp_folder, "optimizer.pt"))
                
                # Upload everything in one go
                commit_message = f"Epoch {epoch}. Batch Size {batch_size}. Peers {len(participating_peers)-len(failed_peers)}."
                upload_folder(
                    folder_path=tmp_folder,
                    repo_id=self.config.neuron.model_name,
                    repo_type="model",
                    commit_message=commit_message
                )
                
                # Create a tag for this version
                create_tag(
                    self.config.neuron.model_name,
                    repo_type="model",
                    tag=str(epoch),
                    tag_message=commit_message
                )
                
                bt.logging.info(f"Successfully pushed new model and optimizer state with tag {epoch}")
                return True
                
        except Exception as e:
            attempt += 1
            bt.logging.warning(
                f"Failed to upload state to HF hub, Retrying. Attempt {attempt}/{self.model_upload_retry_limit}. Error: {str(e)}"
            )
            if attempt < self.model_upload_retry_limit:
                time.sleep(self.model_upload_retry_delay)
            else:
                bt.logging.error("Maximum retry limit reached. Unable to upload state to HF Hub.")
                raise
    return False