import asyncio
import logging
import random
import re
from contextlib import contextmanager
from itertools import chain
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import bittensor as bt
import hivemind
import torch
from hivemind.averaging.allreduce import AveragingMode
from hivemind.averaging.control import AveragingStage, StepControl
from hivemind.averaging.group_info import GroupInfo
from hivemind.averaging.matchmaking import MatchmakingException
from hivemind.compression import deserialize_torch_tensor
from hivemind.optim.progress_tracker import LocalTrainingProgress
from hivemind.p2p import PeerID
from hivemind.proto import averaging_pb2
from hivemind.utils import MPFuture, get_logger, nested_pack
from hivemind.utils.asyncio import aiter_with_timeout, enter_asynchronously
from hivemind.utils.streaming import combine_from_streaming
from hivemind.utils.timed_storage import (DHTExpiration, ValueWithExpiration,
                                          get_dht_time)

GatheredData = Any
logger = get_logger(__name__)
logger.setLevel(logging.DEBUG)

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
    
    def step(
        self,
        gather: Optional[GatheredData] = None,
        scheduled_time: Optional[DHTExpiration] = None,
        weight: Optional[float] = None,
        timeout: Optional[float] = None,
        allow_retries: bool = True,
        require_trigger: bool = False,
        wait: bool = True,
        custom_group_info: Optional[GroupInfo] = None,  # New parameter to accept custom GroupInfo
    ) -> Union[Optional[Dict[PeerID, GatheredData]], StepControl]:
        """
        Set up the averager to look for a group and run one round of averaging, return True on success, False on failure.
        If custom_group_info is provided, it directly uses this group for averaging without original hivemind matchmaking.
        
        Example:
        
        >>> group_id = DHTID.generate().to_bytes()
        >>> ordered_peer_ids = [PeerID.generate() for _ in range(4)]
        >>> gathered = None
        >>> group = GroupInfo(group_id, tuple(ordered_peer_ids), gathered)
        >>> DTGradientAverager.step(custom_group_info = group)
        
        """
        
        if self.mode == AveragingMode.AUX and weight is not None:
            logger.warning("Averager is running in auxiliary mode, weight is unused")
        if scheduled_time is None:
            scheduled_time = get_dht_time() + self.matchmaking_kwargs["min_matchmaking_time"]
        if weight is None:
            weight = float(self.mode != AveragingMode.AUX)
        deadline = get_dht_time() + timeout if timeout is not None else float("inf")
        assert isinstance(weight, (int, float)) and weight >= 0, f"Expected a positive int/float, got {type(weight)}"
        assert not (wait and require_trigger), "Non-asynchronous step cannot wait for trigger (use wait=False)"
        assert scheduled_time < deadline, "Scheduled start time does not fit within timeout"

        user_data_for_gather = self.serializer.dumps(gather)  # serialize here to avoid imports in the averager process
        data_for_gather = self.serializer.dumps([self.bandwidth, self.mode.value, user_data_for_gather])
        step = StepControl(
            scheduled_time=scheduled_time,
            deadline=deadline,
            allow_retries=allow_retries,
            weight=weight,
            data_for_gather=data_for_gather,
        )

        future_for_init = MPFuture()
        
        self._pending_groups_registered = asyncio.Event()
        self._pending_groups_registered.set()
        
        # When custom_group_info is provided, bypass matchmaking and proceed directly
        if custom_group_info is not None:
            
            
            # with self._register_allreduce_group(custom_group_info):
            #     step_control.stage = AveragingStage.RUNNING_ALLREDUCE
            #     self.loop.create_task(
            #         self._aggregate_with_group(
            #                                 custom_group_info,
            #                                 weight=weight, 
            #                                 **self.allreduce_kwargs,)
            #         )
            self._outer_pipe.send(("_step_custom", [], dict(step=step, future_for_init=future_for_init, custom_group_info=custom_group_info)))
            step.attach(*future_for_init.result())
            # future_for_init.set_result(None)  # No need for the result from matchmaking
        else:
            # Default behavior: initiate matchmaking and proceed as originally designed
            self._outer_pipe.send(("_step", [], dict(step=step, future_for_init=future_for_init)))
            step.attach(*future_for_init.result())

        if not require_trigger:
            step.allow_allreduce()
        return step.result() if wait else step
    
    async def _step_custom(self, *, step: StepControl, future_for_init: MPFuture, custom_group_info: GroupInfo):
        try:
            trigger, cancel = MPFuture(), MPFuture()
            step.attach(trigger, cancel)
            future_for_init.set_result((trigger, cancel))
            
            self._pending_groups_registered.clear()
            step.stage = AveragingStage.LOOKING_FOR_GROUP
                        
            with self._register_allreduce_group(custom_group_info):
                step.stage = AveragingStage.RUNNING_ALLREDUCE
                step.set_result(
                    await asyncio.wait_for(
                        self._aggregate_with_group(
                            custom_group_info,
                            tensor_infos=self.tensor_infos,
                            weight=step.weight,
                            **self.allreduce_kwargs,
                            ),
                        timeout=self._allreduce_timeout,
                        )
                    )
                
        except BaseException as e:
            if not step.done():
                step.set_exception(e)
            raise
        finally:
            step.stage = AveragingStage.FINISHED
            if not step.done():
                step.set_exception(
                    RuntimeError(
                        "Internal sanity check failed: averager.step left future pending."
                        " Please report this to hivemind issues."
                    )
                )
            
    async def _aggregate_with_group(self, group_info: GroupInfo, min_vector_size: int, **kwargs) -> GatheredData:
        """Run aggregation in a given group and update tensors in place, return gathered metadata"""
        try:
            # bandwidths, mode_ids, user_gathered_bytes = zip(*map(self.serializer.loads, group_info.gathered))
            # user_gathered = dict(zip(group_info.peer_ids, map(self.serializer.loads, user_gathered_bytes)))
            # modes = tuple(map(AveragingMode, mode_ids))

            # # compute optimal part sizes from peer bandwidths; TODO: replace with proper load balancing
            # download_bandwidths = [
            #     thr if mode != AveragingMode.CLIENT else 0.0 for thr, mode in zip(bandwidths, modes)
            # ]
            # peer_fractions = await asyncio.get_event_loop().run_in_executor(
            #     None, load_balance_peers, self.total_size, download_bandwidths, min_vector_size
            # )
            
            # compute equal part sizes for all peers instead of load balancing
            num_peers = len(group_info.peer_ids)
            peer_fractions = [1.0 / num_peers] * num_peers

            async with enter_asynchronously(self.get_tensors()) as local_tensors:
                await self._run_allreduce_inplace_(
                                                local_tensors, 
                                                group_info, 
                                                peer_fractions=peer_fractions, 
                                                **kwargs)
                return group_info
        except BaseException as e:
            if isinstance(e, Exception):
                logger.exception(e)
            raise MatchmakingException(f"Unable to run All-Reduce: {e}")


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
        self._outer_pipe.send(("_load_state_from_peers_with_latest_state", [], dict(timeout=timeout, global_epoch = global_epoch, future=future)))
        return future.result(timeout=timeout) if wait else future

    async def _load_state_from_peers_with_latest_state(self, global_epoch, future: MPFuture, timeout: Optional[float] = None):
        bt.logging.info(f"Timeout = {timeout}")
        if timeout is not None:
            timeout = self.next_chunk_timeout if self.next_chunk_timeout is not None else self.request_timeout
        try:
            # Extract training progress metadata
            training_progress_metadata, _ = self.dht.get(f"{re.sub('_state_averager', '_progress', self.prefix)}", latest=True) or (None, -float("inf"))
            
            # Get only peers where local_epoch == global_epoch
            if training_progress_metadata is None:
                logger.info(f"Averager could not load metadata from the tracker")
                future.set_result(None)
                return
            else:
                valid_peer_entries = [PeerID(LocalTrainingProgress.parse_obj(peer_state.value).peer_id) for peer_state in training_progress_metadata.values() if (peer_state.value is not None) and (LocalTrainingProgress.parse_obj(peer_state.value).epoch == global_epoch)]

            key_manager = self._matchmaking.group_key_manager
            # prefix = self.state_averager.matchmaking_kwargs['prefix']
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
                        stub = self.get_stub(self._p2p, peer, namespace=key_manager.prefix)
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

    def load_final_state_from_peers(self, global_epoch, **kwargs):
        """
        Attempt to download the latest optimizer state from peers and update trainer parameters/statistics.
        :returns: whether or the averager succeeded in loading parameters
        """
        opt_parameters = tuple(param for param_group in self.optimizer.param_groups for param in param_group["params"])
        main_parameters_and_extras = tuple(chain(opt_parameters, self.extra_tensors))
        num_parameters_and_extras = len(main_parameters_and_extras)

        loaded_state = self.load_state_from_peers_with_latest_state(global_epoch, **kwargs)
        if loaded_state is None:
            return

        metadata, flat_tensors = loaded_state
        if (not isinstance(metadata.get("epoch"), int)) or metadata["epoch"] < self.local_epoch:
            logger.warning("Cowardly refusing to load state from peer: peer's epoch is behind our local epoch")
            return

        loaded_parameters_and_extras = flat_tensors[:num_parameters_and_extras]
        loaded_opt_tensors = flat_tensors[num_parameters_and_extras:]
        if num_parameters_and_extras != len(loaded_parameters_and_extras):
            logger.error("Failed to load state from peer, received parameters, extras or metadata")
            return

        with torch.no_grad(), self.lock_averaged_tensors:
            try:
                load_optimizer_state(self.optimizer, metadata["optimizer_metadata"], loaded_opt_tensors)
            except StopIteration:
                logger.warning("Failed to load state from peer, received inconsistent number of optimizer statistics")
                return

            for local_param, loaded_param in zip(main_parameters_and_extras, loaded_parameters_and_extras):
                local_param.copy_(loaded_param, non_blocking=True)

        if self.offload_optimizer:
            self._apply_optimizer_parameters_()
        if not self.reuse_tensors:
            self._load_local_tensors_into_averager_()

        self.local_epoch = metadata["epoch"]
        self._update_scheduler()

def load_optimizer_state(optimizer: torch.optim.Optimizer, flat_metadata: Dict, flat_tensors: Sequence[torch.Tensor]):
    """Load a state obtained by dump_optimizer_state back into the optimizer"""
    flat_optimizer_state = []
    for elem in flat_metadata:
        if elem.get("type") == "tensor" and isinstance(elem.get("index"), int):
            flat_optimizer_state.append(flat_tensors[elem["index"]])
        elif elem.get("type") == "value" and "value" in elem:
            flat_optimizer_state.append(elem["value"])
    return optimizer.load_state_dict(nested_pack(flat_optimizer_state, structure=optimizer.state_dict()))

def load_state_from_peer(self):

    bt.logging.info('Model Weights Before Loading State')
    bt.logging.info([layer for layer in self.model.parameters()][-1][-10:])
    self.state_averager.load_final_state_from_peers(self.tracker.global_progress.epoch, timeout = self.state_averager.next_chunk_timeout)
    bt.logging.info('Model Weights After Loading State')
    bt.logging.info([layer for layer in self.model.parameters()][-1][-10:])

    self.tracker.local_progress.epoch = self.tracker.global_progress.epoch
    self.local_epoch = self.tracker.local_progress.epoch