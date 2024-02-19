from typing import Any, Dict, Optional, Sequence, Tuple
import random
from hivemind.utils import MPFuture
from hivemind.p2p import PeerID
from hivemind.utils.timed_storage import DHTExpiration, ValueWithExpiration, get_dht_time
from hivemind.utils import MPFuture, get_logger
from hivemind.optim.progress_tracker import LocalTrainingProgress
import logging
import hivemind
import torch
import re
from contextlib import contextmanager

from hivemind.proto import averaging_pb2
from hivemind.utils.asyncio import aiter_with_timeout
from hivemind.utils.streaming import combine_from_streaming
from hivemind.compression import deserialize_torch_tensor

logger = get_logger(__name__)
logger.setLevel(logging.INFO)

class DTGradientAverager(hivemind.optim.grad_averager.GradientAverager):
    '''
    Needs this wrapper class to ensure device is set properly when averaging gradients
    See: https://github.com/learning-at-home/hivemind/blob/d20e81017481aa2028efc33217522248aabd7d95/hivemind/optim/grad_averager.py#L224
    '''
    @contextmanager
    @torch.no_grad()
    def use_averaged_gradients(self):
        """Substitute model's main gradients with averaged gradients"""
        self._new_averaged_grads = False
        with self.get_tensors() as averaged_grads:
            assert len(averaged_grads) == len(self.parameters)
            try:
                old_grads = [param.grad for param in self.parameters]
                for param, new_grad in zip(self.parameters, averaged_grads):
                    # move new_grad to the same device as param before assigning
                    param.grad = new_grad.to(param.device)
                yield averaged_grads
            finally:
                for param, old_grad in zip(self.parameters, old_grads):
                    param.grad = old_grad

    def load_state_from_peers_with_latest_state(
        self, global_epoch, wait: bool = True, timeout: Optional[float] = None, peer_id = None,
    ) -> Optional[Tuple[Any, Sequence[torch.Tensor]]]:
        """
        Try to download the latest optimizer state one of the existing peer.
        :returns: on success, return a 2-tuple with (metadata, tensors), where

        - metadata is a small object containing metadata (e.g. hyperparameters, scalars, etc)
        - tensors is a sequence of pytorch tensors meant to contain peer's model weights and optimizer statistics

        The exact contents of both metadata and tensors are determined by get_current_state method
        """
        future = MPFuture()
        self._outer_pipe.send(("_load_state_from_peers_with_latest_state", [], dict(timeout=timeout, global_epoch=global_epoch, future=future, peer_id = peer_id)))
        return future.result(timeout=timeout) if wait else future

    async def _load_state_from_peers_with_latest_state(self, future: MPFuture, global_epoch, timeout: Optional[float] = None, peer_id = None, max_retries = 3):
        if timeout is not None:
            timeout = self.next_chunk_timeout if self.next_chunk_timeout is not None else self.request_timeout
        try:
            # Extract training progress metadata
            training_progress_metadata, _ = self.dht.get(f"{re.sub('_grad_averager', '_progress', self.prefix)}", latest=True) or (None, -float("inf"))
            
            # Get only peers where local_epoch == global_epoch
            if training_progress_metadata is None:
                logger.info(f"Averager could not load metadata from the tracker")
                future.set_result(None)
                return
            else:
                valid_peer_entries = [PeerID(LocalTrainingProgress.parse_obj(peer_state.value).peer_id) for peer_state in training_progress_metadata.values() if (peer_state.value is not None) and (LocalTrainingProgress.parse_obj(peer_state.value).epoch == global_epoch)]

            # Get all the peers connected to the correct gradient averager and filter for peers with the right local epoch
            key_manager = self._matchmaking.group_key_manager
            peer_priority, _ = self.dht.get(f"{key_manager.prefix}.all_averagers", latest=True) or ({}, None)
            peer_priority = {
                PeerID(peer_id): (float(info.value), random.random())  # using randomness as a tie breaker
                for peer_id, info in peer_priority.items()
                if isinstance(info, ValueWithExpiration) and isinstance(info.value, (float, int)) and (PeerID(peer_id) in valid_peer_entries)
            }

            if not isinstance(peer_priority, dict) or len(peer_priority) == 0:
                logger.info(f"Averager could not load state from peers: peer dict empty or corrupted {peer_priority}")
                future.set_result(None)
                return
            
            metadata = None
            for peer in sorted(peer_priority.keys(), key=peer_priority.get, reverse=True):
                if peer != self.peer_id:
                    logger.info(f"Downloading parameters from peer {peer}")
                    try:
                        stub = self.get_stub(self._p2p, peer, namespace=self.prefix)
                        stream = await stub.rpc_download_state(averaging_pb2.DownloadRequest())
                        current_tensor_parts, tensors = [], []

                        # TODO merge this with hivemind.compression.deserialize_tensor_stream
                        async for message in aiter_with_timeout(stream, timeout=timeout):
                            if message.metadata:
                                metadata = self.serializer.loads(message.metadata)
                            if message.tensor_part.dtype and current_tensor_parts:
                                # tensor_part.dtype indicates the start of the new tensor, so we should wrap up this one
                                tensors.append(deserialize_torch_tensor(combine_from_streaming(current_tensor_parts)))
                                current_tensor_parts = []
                            current_tensor_parts.append(message.tensor_part)
                        if current_tensor_parts:
                            tensors.append(deserialize_torch_tensor(combine_from_streaming(current_tensor_parts)))

                        if not metadata:
                            logger.debug(f"Peer {peer} did not send its state")
                            continue

                        logger.info(f"Finished downloading state from {peer}")
                        future.set_result((metadata, tensors))
                        return
                    except Exception as e:
                        logger.exception(f"Failed to download state from {peer} - {repr(e)}")

        finally:
            if not future.done():
                future.set_result(None)
