import asyncio
import logging
import random
import re
from contextlib import contextmanager
from itertools import chain
from typing import Any, AsyncIterator, Dict, Optional, Sequence, Tuple, Union

import bittensor as bt
import hivemind
import hivemind.averaging
import hivemind.averaging.averager
import numpy as np
import torch
from hivemind.averaging.allreduce import (AllreduceException, AllReduceRunner,
                                          AveragingMode)
from hivemind.averaging.control import AveragingStage, StepControl
from hivemind.averaging.group_info import GroupInfo
from hivemind.averaging.matchmaking import MatchmakingException
from hivemind.compression import (deserialize_torch_tensor,
                                  serialize_torch_tensor)
from hivemind.optim.progress_tracker import LocalTrainingProgress
from hivemind.p2p import (P2PDaemonError, P2PHandlerError,
                          PeerID)
from hivemind.proto import averaging_pb2
from hivemind.utils import MPFuture, get_logger, nested_pack
from hivemind.utils.asyncio import (aenumerate, aiter_with_timeout,
                                    amap_in_executor, as_aiter,
                                    attach_event_on_finished, azip,
                                    enter_asynchronously)
from hivemind.utils.streaming import combine_from_streaming
from hivemind.utils.timed_storage import (DHTExpiration, ValueWithExpiration,
                                          get_dht_time)

GatheredData = Any
logger = get_logger(__name__)
logger.setLevel(logging.DEBUG)


class DTAllReduceRunner(AllReduceRunner):
    
    async def _communicate_with_peer(self, peer_id: PeerID):
        """Send a part of local tensors and metadata to a single peer, receive the average for that part of tensors"""
        peer_index = self.ordered_peer_ids.index(peer_id)
        if peer_id == self.peer_id:
            sender_index = self.sender_peer_ids.index(peer_id)
            for part_index, tensor_part in enumerate(self.parts_for_local_averaging):
                averaged_part = await self.tensor_part_reducer.accumulate_part(
                    sender_index, part_index, tensor_part, weight=self.weight
                )
                self.tensor_part_container.register_processed_part(peer_index, part_index, averaged_part - tensor_part)

        else:
            try:
                done_sending = asyncio.Event()
                inputs_aiter = attach_event_on_finished(self._generate_input_for_peer(peer_index), done_sending)
                stream = await self._get_peer_stub(peer_id).rpc_aggregate_part(inputs_aiter)

                if self.should_delay_results(self.peer_id):
                    await done_sending.wait()

                part_index = 0

                def _try_deserialize(msg):
                    if msg.code != averaging_pb2.AVERAGED_PART:
                        raise AllreduceException(f"{peer_id} sent {averaging_pb2.MessageCode.Name(msg.code)}")
                    return deserialize_torch_tensor(msg.tensor_part), msg

                async for delta, msg in amap_in_executor(
                    _try_deserialize,
                    aiter_with_timeout(stream, self.reducer_timeout),
                    max_prefetch=self.tensor_part_container.prefetch,
                ):
                    self.tensor_part_container.register_processed_part(peer_index, part_index, delta)
                    part_index += 1

                if part_index != self.tensor_part_container.num_parts_by_peer[peer_index]:
                    raise AllreduceException(
                        f"peer {peer_id} sent {part_index} parts, but we expected "
                        f"{self.tensor_part_container.num_parts_by_peer[peer_index]}"
                    )
            except BaseException as e:
                if isinstance(e, Exception):
                    logger.debug(f"Caught {repr(e)} when communicating to {peer_id}", exc_info=True)
                #? Remove fault-tolerant method here
                #self.tensor_part_container.register_failed_reducer(peer_index) 
                raise

    async def rpc_aggregate_part(self, stream, context) -> AsyncIterator[averaging_pb2.AveragingData]:
        """
        Handles the aggregation of tensor parts sent by peers. If an error is encountered, such as a timeout
        or failure in communication, it directly raises an exception.
        """
        try:
            #! Test fault-tolerance here:
            # condition = np.random.choice(["FAIL_SENDING", "SLOW_REDUCE", "CANCEL"])  
            test_fault = True
            if test_fault:
                condition = None
                
                async for message in super().rpc_aggregate_part(stream, context):
                    self.count+=1
                    yield message
                    if self.count == 2:
                        if condition == "FAIL_SENDING":
                            yield averaging_pb2.AveragingData(code=averaging_pb2.INTERNAL_ERROR)
                            break
                        elif condition == "SLOW_REDUCE":
                            await asyncio.sleep(10)
                        elif condition == "CANCEL":  
                            yield averaging_pb2.AveragingData(code=averaging_pb2.CANCELLED)
            else:
                async for message in super().rpc_aggregate_part(stream, context):
                    yield message

        except Exception as e:
            logger.error(f"RPC aggregation error with peer {context.remote_id}: {e}")
            raise e

    async def _generate_input_for_peer(self, peer_index: int) -> AsyncIterator[averaging_pb2.AveragingData]:
        """
        Prepares and sends tensor parts to a peer for aggregation. If a fault condition like failing to send
        or delays are detected, it raises an exception.
        """
        try:
            parts_aiter = self.tensor_part_container.iterate_input_parts_for(peer_index)
            first_part = await anext(parts_aiter)
            yield averaging_pb2.AveragingData(
                code=averaging_pb2.PART_FOR_AVERAGING,
                group_id=self.group_id,
                tensor_part=first_part,
                weight=self.weight,
            )
            #! Test Fault-tolerance here:
            # last_reducer_index = self.group_size - 1 - (self.tensor_part_container.num_parts_by_peer[-1] == 0)
            # if peer_index == last_reducer_index:
            #     # Create random condition:
            #     condition = np.random.choice(["FAIL_SENDING", "SLOW_REDUCE"])    
            #     if condition == "FAIL_SENDING":
            #         raise Exception("Oops, I failed!")
            #     else:
            #         await asyncio.sleep(10)
            
            async for part in parts_aiter:
                yield averaging_pb2.AveragingData(tensor_part=part, weight=self.weight)
            
            
        except Exception as e:
            logger.error(f"Error preparing input for peer {self.ordered_peer_ids[peer_index]}: {e}")
            raise e
    
    #TODO This is hit if the rpc_aggregate_with_peer is failing - so maybe we are using double Exceptions atm    
    async def _ban_sender(self, peer_id: PeerID):
        async with self.banlock:
            if peer_id not in self.banned_senders:
                self.banned_senders.add(peer_id)
                #? Remove fault-tolerant method here:
                #self.tensor_part_reducer.on_sender_failed(self.sender_peer_ids.index(peer_id))
                error_message = f"Banning peer {peer_id} due to a failure."
                logger.error(error_message)
                raise Exception(error_message)
            
        
# class DTGradientAverager(hivemind.optim.grad_averager.GradientAverager):
class DTGradientAverager(hivemind.averaging.averager.DecentralizedAverager):
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
            
            #while not step.done():
            try:
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
            except (
                AllreduceException,
                MatchmakingException,
                AssertionError,
                StopAsyncIteration,
                asyncio.CancelledError,
                asyncio.InvalidStateError,
                P2PHandlerError,
                P2PDaemonError,
            ) as e:
                if step.done() or not step.allow_retries or get_dht_time() >= step.deadline:
                    if not step.cancelled():
                        logger.exception(e)
                    if not step.done():
                        step.set_exception(e)
                else:
                    logger.warning(f"{self.__class__.__name__} caught {repr(e)}, retrying")
            
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

            # async with enter_asynchronously(self.get_tensors()) as local_tensors:
            #     await self._run_allreduce_inplace_(
            #                                     local_tensors, 
            #                                     group_info, 
            #                                     peer_fractions=peer_fractions, 
            #                                     **kwargs)
                
                
            async with enter_asynchronously(self.get_tensors()) as local_tensors:
                runner = DTAllReduceRunner(
                    p2p=self._p2p,
                    servicer_type=type(self),
                    prefix=self.prefix,
                    group_id=group_info.group_id,
                    tensors=local_tensors,
                    ordered_peer_ids=group_info.peer_ids,
                    peer_fractions=peer_fractions,
                    **kwargs,
                )

                self._running_groups[group_info.group_id].set_result(runner)

                if runner.modes[group_info.peer_ids.index(self.peer_id)] != AveragingMode.AUX:
                    iter_results = runner.run()
                    async for tensor, update in azip(as_aiter(*local_tensors), iter_results):
                        # all-reduce is performed asynchronously while iterating
                        tensor.add_(update, alpha=self._averaging_alpha)
                        self.last_updated = get_dht_time()
                        self._state_updated.set()
                else:
                    async for _ in runner:
                        raise ValueError("aux peers should not receive averaged tensors")
                
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