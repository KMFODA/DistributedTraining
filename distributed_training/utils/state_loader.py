import copy
import logging
import random
import re
from itertools import chain
from typing import Any, Dict, Optional, Sequence, Tuple

import bittensor as bt
import hivemind
import torch
from hivemind.compression import deserialize_torch_tensor
from hivemind.p2p import PeerID
from hivemind.proto import averaging_pb2
from hivemind.utils import MPFuture, get_logger, nested_pack
from hivemind.utils.asyncio import aiter_with_timeout
from hivemind.utils.streaming import combine_from_streaming
from hivemind.utils.timed_storage import ValueWithExpiration
from huggingface_hub import create_tag, list_repo_refs, scan_cache_dir
from transformers import AutoModelForCausalLM
from bitsandbytes.optim import LAMB

from distributed_training.utils.progress_tracker import (
    LocalTrainingProgress,
    update_global_tracker_state,
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


def load_state_from_peer(self, epoch=None, keep_recent=5):
    state_loaded = False
    if epoch == None:
        update_global_tracker_state(self)
        epoch = self.global_progress.epoch

    bt.logging.info("Model Weights Before Loading State")
    current_model_weights_sample = copy.copy(
        [layer for layer in self.model.parameters()][-2][-10:].tolist()
    )
    bt.logging.info(current_model_weights_sample)

    bt.logging.info(f"Old Model Tag: {self.local_progress.epoch}")
    # if (self.global_progress.epoch is not None) and (tag_name >= epoch):
    if self.global_progress.epoch is not None:
        bt.logging.info(
            f"Latest Model State Found On The HF Hub With The Tag: {self.global_progress.epoch}. Loading That Model State."
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.neuron.model_name,
            revision=str(self.global_progress.epoch),
            trust_remote_code=True,
        )
        self.model.to(self.device)
        param_dict = {pn: p for pn, p in self.model.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": self.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        self.opt = LAMB(
            optim_groups, lr=self.learning_rate_maximum, betas=(0.9, 0.95), eps=1e-8
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

        # Delete one model from the chace to maintain disk space
        current_revision = self.model.config._commit_hash
        try:
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
                        for revision in revisions[
                            max(current_index + 1, keep_recent) :
                        ]:
                            cache_info.delete_revisions(revision.commit_hash).execute()
                    break
        except:
            bt.logging.warning(
                "Failed to delete previous model version from cache. This might lead to 100% disk space utlisation in the future."
            )

    else:
        bt.logging.info(f"Model With Tag: {epoch} Does Not Exist")

    return state_loaded
