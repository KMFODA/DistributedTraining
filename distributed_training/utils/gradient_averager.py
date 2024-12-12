import asyncio
import logging
from contextlib import contextmanager
from typing import (
    Any,
    AsyncIterator,
    Dict,
    Iterable,
    Iterator,
    Optional,
    Sequence,
    Union,
)

import bittensor as bt
import torch

import hivemind
import hivemind.averaging.averager
from hivemind.averaging.allreduce import (
    AllreduceException,
    AllReduceRunner,
    AveragingMode,
)
from hivemind.averaging.control import AveragingStage, StepControl
from hivemind.averaging.group_info import GroupInfo
from hivemind.averaging.load_balancing import load_balance_peers
from hivemind.averaging.matchmaking import MatchmakingException
from hivemind.compression import CompressionInfo, deserialize_torch_tensor
from hivemind.dht import DHT
from hivemind.p2p import P2PContext, P2PDaemonError, P2PHandlerError, PeerID
from hivemind.proto import averaging_pb2
from hivemind.utils import MPFuture, get_logger
from hivemind.utils.asyncio import (
    aiter_with_timeout,
    amap_in_executor,
    as_aiter,
    attach_event_on_finished,
    azip,
    enter_asynchronously,
)
from hivemind.utils.streaming import split_for_streaming
from hivemind.utils.timed_storage import DHTExpiration, get_dht_time

GatheredData = Any

logger = get_logger(__name__)
logger.setLevel(logging.INFO)


class DTAllReduceRunner(AllReduceRunner):
    def __init__(self, peerids_to_uids, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.count = 0
        self.peerids_to_uids = peerids_to_uids
        bt.logging.info(f"PeerID to UID mapping: {self.peerids_to_uids}")
        self.nan_count = 0
        self.max_nan_tolerance = 3  # Max number of NaN tensors before banning peer
    
    @staticmethod
    @torch.jit.script
    def _fast_nan_check(tensor: torch.Tensor) -> bool:
        """JIT-optimized NaN check"""
        return bool(torch.isnan(tensor).any().item())

    def _check_tensor_for_nan(self, tensor: torch.Tensor, peer_id: str, part_index: int) -> bool:
        """Check tensor and handle logging"""
        has_nan = self._fast_nan_check(tensor)
        if has_nan:
            uid = self.peerids_to_uids.get(str(peer_id), "'''")
            peer_id_abbreviated = str(peer_id)[-3:]
            bt.logging.warning(
                f"UID:{uid} - PeerID:{peer_id_abbreviated} - NaN detected in tensor part {part_index}"
            )
        return has_nan

    @staticmethod
    @torch.jit.script
    def _handle_nan_tensor(tensor: torch.Tensor) -> torch.Tensor:
        """Replace NaN values with zeros in-place"""
        nan_mask = torch.isnan(tensor)
        if bool(nan_mask.any().item()):
            tensor.masked_fill_(nan_mask, 0.0)
        return tensor

    async def _communicate_with_peer(self, peer_id: PeerID):
        """Send a part of local tensors and metadata to a single peer, receive the average for that part of tensors"""
        uid = self.peerids_to_uids.get(str(peer_id), "'''")
        peer_id_abreviated = str(peer_id)[-3:]

        bt.logging.info(
            f"UID:{uid} - PeerID:{peer_id_abreviated} - communicate_with_peer started",
        )
        peer_index = self.ordered_peer_ids.index(peer_id)
        if peer_id == self.peer_id:
            sender_index = self.sender_peer_ids.index(peer_id)
            for part_index, tensor_part in enumerate(self.parts_for_local_averaging):
                # Check local tensor for NaN before averaging
                if self._check_tensor_for_nan(tensor_part, peer_id, part_index):
                    tensor_part = self._handle_nan_tensor(tensor_part)
                    
                averaged_part = await self.tensor_part_reducer.accumulate_part(
                    sender_index, part_index, tensor_part, weight=self.weight
                )
                # Check averaged result for NaN
                if self._check_tensor_for_nan(averaged_part, peer_id, part_index):
                    averaged_part = self._handle_nan_tensor(averaged_part)
                    
                delta = averaged_part - tensor_part
                if self._check_tensor_for_nan(delta, peer_id, part_index):
                    delta = torch.zeros_like(delta)
                    
                self.tensor_part_container.register_processed_part(
                    peer_index,
                    part_index,
                    delta,
                )

        else:
            try:
                done_sending = asyncio.Event()
                inputs_aiter = attach_event_on_finished(
                    self._generate_input_for_peer(peer_index, uid, peer_id),
                    done_sending,
                )
                bt.logging.info(
                    f"UID:{uid} - PeerID:{peer_id_abreviated} - generate_input_for_peer started"
                )
                stream = await self._get_peer_stub(peer_id).rpc_aggregate_part(
                    inputs_aiter
                )
                bt.logging.info(
                    f"UID:{uid} - PeerID:{peer_id_abreviated} - get_peer_stub finished"
                )

                if self.should_delay_results(self.peer_id):
                    await done_sending.wait()

                bt.logging.info(
                    f"UID:{uid} - PeerID:{peer_id_abreviated} - sending tensors finished"
                )
                part_index = 0

                def _try_deserialize(msg):
                    if msg.code != averaging_pb2.AVERAGED_PART:
                        raise AllreduceException(
                            f"{peer_id_abreviated} sent {averaging_pb2.MessageCode.Name(msg.code)}"
                        )
                    tensor, msg = deserialize_torch_tensor(msg.tensor_part), msg
                    
                    # Check received tensor for NaN
                    if self._check_tensor_for_nan(tensor, peer_id, part_index):
                        self.nan_count += 1
                        if self.nan_count >= self.max_nan_tolerance:
                            bt.logging.warning(f"NaN count exceeded for peer {peer_id_abbreviated}, banning")
                            raise AllreduceException(f"Too many NaN tensors from peer {peer_id_abbreviated}")
                        tensor = self._handle_nan_tensor(tensor)
                    
                    return tensor, msg

                async for delta, msg in amap_in_executor(
                    _try_deserialize,
                    aiter_with_timeout(stream, self.reducer_timeout),
                    max_prefetch=self.tensor_part_container.prefetch,
                ):
                    self.tensor_part_container.register_processed_part(
                        peer_index,
                        part_index,
                        delta,
                    )
                    part_index += 1
                bt.logging.info(
                    f"UID:{uid} - PeerID:{peer_id_abreviated} - register_processed_part finished"
                )
                if (
                    part_index
                    != self.tensor_part_container.num_parts_by_peer[peer_index]
                ):
                    bt.logging.info(
                        f"part_index != self.tensor_part_container.num_parts_by_peer[peer_index]"
                    )
                    raise AllreduceException(
                        f"peer {peer_id_abreviated} sent {part_index} parts, but we expected "
                        f"{self.tensor_part_container.num_parts_by_peer[peer_index]}"
                    )
            except BaseException as e:
                if isinstance(e, Exception):
                    logger.debug(
                        f"Caught {repr(e)} when communicating to {peer_id_abreviated}",
                        exc_info=True,
                    )
                bt.logging.info(
                    f"UID:{uid} - PeerID:{peer_id_abreviated} - Failed to communicate with peers due to error - {e}"
                )
                self.tensor_part_container.register_failed_reducer(peer_index)
                await self._ban_sender(peer_id)
                raise

    def __aiter__(self):
        return self.run()

    async def run(self) -> AsyncIterator[torch.Tensor]:
        """Run all-reduce, return differences between averaged and original tensors as they are computed"""
        pending_tasks = set()
        bt.logging.info("Running AllReducerRunner")
        bt.logging.info(
            f"self.tensor_part_container.num_parts_by_peer {self.tensor_part_container.num_parts_by_peer}"
        )
        if (
            self.tensor_part_container.num_parts_by_peer[
                self.ordered_peer_ids.index(self.peer_id)
            ]
            != 0
        ):
            pending_tasks.add(asyncio.create_task(self._handle_missing_senders()))

        try:
            if len(self.sender_peer_ids) == 0:
                logger.debug(
                    f"{self} - finished all-reduce early: all peers are auxiliaries ({self.modes})"
                )
                self.finalize()

            elif self.peer_id in self.sender_peer_ids:
                uid = self.peerids_to_uids.get(str(self.peer_id), "'''")
                peer_id_abreviated = str(self.peer_id)[-3:]

                bt.logging.info(
                    f"UID:{uid} - PeerID:{peer_id_abreviated} peer_id in sender_peer_ids"
                )

                for peer_id, parts in zip(
                    self.ordered_peer_ids, self.tensor_part_container.num_parts_by_peer
                ):
                    if parts != 0:
                        pending_tasks.add(
                            asyncio.create_task(self._communicate_with_peer(peer_id))
                        )

                bt.logging.info(f"Succesfully Communicated With All Peers")

                async for (
                    averaged_tensor_delta
                ) in self.tensor_part_container.iterate_output_tensors():
                    yield averaged_tensor_delta  # delta = averaged_tensor - original_tensor

                bt.logging.info(f"Iterate Output Tensors Finished")

                self.finalize()

                bt.logging.info(f"Finalize Finished")

            else:  # auxiliary peer
                await self.tensor_part_reducer.finished.wait()
                self.finalize()

        except BaseException as e:
            bt.logging.info(f"All Reduce Runner failed with error {e}")

            self.finalize(exception=e)
            for task in pending_tasks:
                task.cancel()
            raise

        finally:
            for task in pending_tasks:
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                except Exception as inner_exc:
                    logger.debug(f"Task {task} failed with {inner_exc}", exc_info=True)

    async def _generate_input_for_peer(
        self, peer_index: int, uid: str, peer_id: PeerID
    ) -> AsyncIterator[averaging_pb2.AveragingData]:
        peer_id_abreviated = str(self.peer_id)[-3:]
        try:
            parts_aiter = self.tensor_part_container.iterate_input_parts_for(peer_index)
            first_part = await anext(parts_aiter)
            
            # Check first part for NaN before sending
            if self._check_tensor_for_nan(first_part, peer_id, 0):
                first_part = self._handle_nan_tensor(first_part)
                
            yield averaging_pb2.AveragingData(
                code=averaging_pb2.PART_FOR_AVERAGING,
                group_id=self.group_id,
                tensor_part=first_part,
                weight=self.weight,
            )
            bt.logging.info(
                f"UID:{uid} - PeerID:{peer_id_abreviated} - generate_input_for_peer finished"
            )

            async for part in parts_aiter:
                if self._check_tensor_for_nan(part, peer_id, -1):
                    part = self._handle_nan_tensor(part)
                yield averaging_pb2.AveragingData(tensor_part=part, weight=self.weight)

        except Exception as e:
            logger.error(
                f"Error preparing input for peer {self.ordered_peer_ids[peer_index]}: {e}"
            )

            raise e

    async def _ban_sender(self, peer_id: PeerID):
        uid = self.peerids_to_uids.get(str(peer_id), "'''")
        peer_id_abreviated = str(self.peer_id)[-3:]
        bt.logging.info(f"UID:{uid} - PeerID:{peer_id_abreviated} - Banning Peer")

        async with self.banlock:
            if peer_id not in self.banned_senders:
                self.banned_senders.add(peer_id)
                self.tensor_part_reducer.on_sender_failed(
                    self.sender_peer_ids.index(peer_id)
                )


class DTAverager(hivemind.DecentralizedAverager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def step(
        self,
        gather: Optional[GatheredData] = None,
        scheduled_time: Optional[DHTExpiration] = None,
        weight: Optional[float] = None,
        timeout: Optional[float] = None,
        allow_retries: bool = True,
        require_trigger: bool = False,
        wait: bool = True,
        **kwargs,
    ) -> Union[Optional[Dict[PeerID, GatheredData]], StepControl]:
        if self.mode == AveragingMode.AUX and weight is not None:
            logger.warning("Averager is running in auxiliary mode, weight is unused")
        if scheduled_time is None:
            scheduled_time = (
                get_dht_time() + self.matchmaking_kwargs["min_matchmaking_time"]
            )
        if weight is None:
            weight = float(self.mode != AveragingMode.AUX)
        deadline = get_dht_time() + timeout if timeout is not None else float("inf")
        assert (
            isinstance(weight, (int, float)) and weight >= 0
        ), f"Expected a positive int/float, got {type(weight)}"
        assert not (
            wait and require_trigger
        ), "Non-asynchronous step cannot wait for trigger (use wait=False)"
        assert (
            scheduled_time < deadline
        ), "Scheduled start time does not fit within timeout"

        user_data_for_gather = self.serializer.dumps(
            gather
        )  # serialize here to avoid imports in the averager process
        data_for_gather = self.serializer.dumps(
            [self.bandwidth, self.mode.value, user_data_for_gather]
        )
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

        peerids_to_uids = kwargs.get("peerids_to_uids", {})

        # Default behavior: initiate matchmaking and proceed as originally designed
        self._outer_pipe.send(
            (
                "_step",
                [],
                dict(
                    step=step,
                    future_for_init=future_for_init,
                    peerids_to_uids=peerids_to_uids,
                ),
            )
        )
        step.attach(*future_for_init.result())

        if not require_trigger:
            step.allow_allreduce()
        return step.result() if wait else step

    async def _step(
        self,
        *,
        step: StepControl,
        future_for_init: MPFuture,
        peerids_to_uids: Dict = {},
    ):
        try:
            trigger, cancel = MPFuture(), MPFuture()
            step.attach(trigger, cancel)
            future_for_init.set_result((trigger, cancel))

            async def find_peers_or_notify_cancel():
                group_info = await self._matchmaking.look_for_group(step)
                if not step.triggered:
                    step.stage = AveragingStage.AWAITING_TRIGGER
                    await step.wait_for_trigger()
                return group_info

            while not step.done():
                try:
                    self._pending_groups_registered.clear()
                    step.stage = AveragingStage.LOOKING_FOR_GROUP
                    matchmaking_task = asyncio.create_task(
                        find_peers_or_notify_cancel()
                    )
                    check_cancel_task = asyncio.create_task(step.wait_for_cancel())

                    await asyncio.wait(
                        {matchmaking_task, check_cancel_task},
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if step.cancelled():
                        matchmaking_task.cancel()
                        raise asyncio.CancelledError()
                    else:
                        check_cancel_task.cancel()

                    group_info = await matchmaking_task

                    if group_info is None:
                        raise AllreduceException(
                            "Averaging step failed: could not find a group"
                        )

                    with self._register_allreduce_group(group_info):
                        step.stage = AveragingStage.RUNNING_ALLREDUCE
                        step.set_result(
                            await asyncio.wait_for(
                                self._aggregate_with_group(
                                    group_info,
                                    tensor_infos=self.tensor_infos,
                                    weight=step.weight,
                                    peerids_to_uids=peerids_to_uids,
                                    **self.allreduce_kwargs,
                                ),
                                timeout=self._allreduce_timeout,
                            )
                        )
                        # averaging is finished, loop will now exit

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
                    if (
                        step.done()
                        or not step.allow_retries
                        or get_dht_time() >= step.deadline
                    ):
                        if not step.cancelled():
                            logger.exception(e)
                        if not step.done():
                            step.set_exception(e)
                    else:
                        logger.warning(
                            f"{self.__class__.__name__} caught {repr(e)}, retrying"
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

    async def _aggregate_with_group(
        self,
        group_info: GroupInfo,
        min_vector_size: int,
        peerids_to_uids: Dict,
        **kwargs,
    ) -> GatheredData:
        """Run aggregation in a given group and update tensors in place, return gathered metadata"""
        try:
            bandwidths, mode_ids, user_gathered_bytes = zip(
                *map(self.serializer.loads, group_info.gathered)
            )
            user_gathered = dict(
                zip(
                    group_info.peer_ids, map(self.serializer.loads, user_gathered_bytes)
                )
            )
            modes = tuple(map(AveragingMode, mode_ids))

            # compute optimal part sizes from peer bandwidths; TODO: replace with proper load balancing
            download_bandwidths = [
                thr if mode != AveragingMode.CLIENT else 0.0
                for thr, mode in zip(bandwidths, modes)
            ]
            # bt.logging.info("Donwloaded bandwidths")
            # bt.logging.info(download_bandwidths)
            peer_fractions = await asyncio.get_event_loop().run_in_executor(
                None,
                load_balance_peers,
                self.total_size,
                download_bandwidths,
                min_vector_size,
            )
            bt.logging.info(group_info.peer_ids)
            bt.logging.info(peer_fractions)
            async with enter_asynchronously(self.get_tensors()) as local_tensors:
                runner = DTAllReduceRunner(
                    peerids_to_uids=peerids_to_uids,
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

                if (
                    runner.modes[group_info.peer_ids.index(self.peer_id)]
                    != AveragingMode.AUX
                ):
                    async for tensor, update in azip(as_aiter(*local_tensors), runner):
                        # all-reduce is performed asynchronously while iterating
                        tensor.add_(update, alpha=self._averaging_alpha)
                        self.last_updated = get_dht_time()
                        self._state_updated.set()
                else:
                    async for _ in runner:
                        raise ValueError(
                            "aux peers should not receive averaged tensors"
                        )

                return user_gathered, runner.banned_senders, group_info.peer_ids
        except BaseException as e:
            if isinstance(e, Exception):
                logger.exception(e)
            raise MatchmakingException(f"Unable to run All-Reduce: {e}")

    async def rpc_download_state_partial(
        self, _request: averaging_pb2.DownloadRequest, _context: P2PContext
    ) -> AsyncIterator[averaging_pb2.DownloadData]:
        """
        Get the up-to-date trainer state from a peer.
        The state consists of two parts: (serialized_metadata, tensors)

         - serialized_metadata is a small serialized bytestring meant to store scalars and hyperparameters
         - tensors is a sequence of pytorch tensors that represent model parameters or optimizer statistics
        """
        logger.info("rpc_download_state_partial")
        if not self.allow_state_sharing:
            return  # deny request and direct peer to the next prospective averager
        metadata, tensors, infos = await self._get_current_state_from_host_process()
        logger.info(len(tensors))
        if infos is None:
            infos = [
                CompressionInfo.from_tensor(tensor, key=i)
                for i, tensor in enumerate(tensors)
            ]
        assert len(tensors) == len(infos)

        # for tensor, info in zip([tensors[0]], infos):
        for tensor, info in zip([tensors[0]], infos):
            for part in split_for_streaming(
                self.state_compression.compress(tensor, info, allow_inplace=False)
            ):
                if metadata is not None:
                    yield averaging_pb2.DownloadData(
                        tensor_part=part, metadata=metadata
                    )
                    metadata = None
                else:
                    yield averaging_pb2.DownloadData(tensor_part=part)
            break


class DTGradientAverager(DTAverager):
    def __init__(
        self,
        parameters: Iterable[torch.nn.Parameter],
        *,
        dht: DHT,
        prefix: str,
        reuse_grad_buffers: bool = False,
        accumulate_grads_on: Optional[torch.device] = None,
        client_mode: bool = None,
        warn: bool = True,
        averaged_grads: Sequence[torch.Tensor] = (),
        **kwargs,
    ):
        if reuse_grad_buffers and accumulate_grads_on is not None:
            logger.warning(
                "Setting 'accumulate_grads_on' has no effect if reuse_grad_buffers=True"
            )
        client_mode = client_mode if client_mode is not None else dht.client_mode
        self.parameters = tuple(parameters)
        self.reuse_grad_buffers = reuse_grad_buffers
        self.warn = warn
        self.local_samples_accumulated = 0
        self.local_times_accumulated = 0
        self._anchor_batch_size = None
        self._local_accumulators = None
        if not reuse_grad_buffers:
            self._local_accumulators = tuple(
                torch.zeros_like(grad, device=accumulate_grads_on)
                for grad in self._grads_from_parameters()
            )
        self._accumulators_used_in_step = False
        self._new_averaged_grads = False

        with torch.no_grad():
            if not averaged_grads:
                averaged_grads = tuple(
                    grad.detach().cpu().clone().share_memory_()
                    for grad in self._grads_from_parameters()
                )
            else:
                if any(
                    param_grad.size() != grad.size()
                    for param_grad, grad in zip(
                        self._grads_from_parameters(), averaged_grads
                    )
                ):
                    raise ValueError(
                        "Averaged gradients don't have same shape as gradients from parameters"
                    )
        super().__init__(
            averaged_tensors=averaged_grads,
            dht=dht,
            prefix=prefix,
            client_mode=client_mode,
            **kwargs,
        )

    def _grads_from_parameters(self) -> Iterator[torch.Tensor]:
        """gradient buffers associated with parameters"""
        for param in self.parameters:
            if param.grad is None:
                param.grad = torch.zeros_like(param)
            yield param.grad

    @torch.no_grad()
    def _grad_accumulators(self) -> Iterator[torch.Tensor]:
        """averager-based gradient accumulators"""
        assert (self._local_accumulators is None) == self.reuse_grad_buffers
        yield from (
            self._grads_from_parameters()
            if self.reuse_grad_buffers
            else self._local_accumulators
        )

    @torch.no_grad()
    def accumulate_grads_(self, batch_size: int):
        """add current gradients to local grad accumulators (if used)"""
        if self._accumulators_used_in_step and self.warn:
            logger.warning(
                "[warn=True] Gradient accumulators were not reset since the last averaging round. Please "
                "call .reset_accumulated_grads_ after every step or use .step(reset_accumulators=True)"
            )
            self._accumulators_used_in_step = False  # warn once per round
        if self._anchor_batch_size is None:
            # remember the first batch size to correctly re-scale gradients if subsequent batches have a different size
            self._anchor_batch_size = batch_size
        self.local_samples_accumulated += batch_size
        self.local_times_accumulated += 1
        if self.reuse_grad_buffers:
            pass  # user is responsible for accumulating gradients in .grad buffers
        else:
            alpha = float(batch_size) / self._anchor_batch_size
            for grad_buf, grad_acc in zip(
                self._grads_from_parameters(), self._grad_accumulators()
            ):
                grad_acc.add_(grad_buf.to(grad_acc.device), alpha=alpha)

    def schedule_step(
        self, scheduled_time: Optional[DHTExpiration] = None, **kwargs
    ) -> StepControl:
        """
        Begin matchmaking: look for a group of peers and prepare for averaging gradients at a specified time.

        :param scheduled_time: expected time when to perform all-reduce. Can be changed using control.scheduled_time
        :param kwargs: any additional keyword args from DecentralizedAverager.step, such as gather, allow_retries, etc
        :note: setting weight at this stage is not supported, please leave this parameter as None
        :returns: step_control - a handle that can be passed into GradientAverager.step to use the pre-scheduled group
        :note: in the current implementation, each step_control can only be used in one step.
        """
        assert (
            kwargs.get("weight") is None
        ), "setting weight in schedule_step is not supported"
        return super().step(
            scheduled_time=scheduled_time, wait=False, require_trigger=True, **kwargs
        )

    def step(
        self,
        weight: Optional[float] = None,
        reset_accumulators: bool = True,
        control: Optional[StepControl] = None,
        timeout: Optional[float] = None,
        wait: bool = True,
        **kwargs,
    ):
        """
        Average accumulated gradients with peers, optionally load averaged gradients and reset accumulators

        :param weight: overrides the averaging weight; by default, weight equals the number of accumulated samples
        :param reset_accumulators: by default, set local gradient accumulators to zeros after averaging succeeds
        :param control: reuse a pre-arranged group of peers (or a matchmaking in progress) from averager.schedule_step
        :param timeout: if specified, await for averaging round for at most this number of seconds (if wait=True)
        :param wait: if True, await for the step to finish (or fail), otherwise run all-reduce in background
        """
        if control is None:
            control = self.schedule_step(timeout=timeout, **kwargs)
        elif len(kwargs) > 0:
            raise RuntimeError(
                f"Averaging with a pre-scheduled group, parameters {kwargs} will have no effect"
            )
        assert not control.triggered, f"This {type(control)} instance was already used"
        if self._new_averaged_grads and self.warn:
            logger.warning(
                "[warn=True] Starting new averaging round, but previous round results were not used. "
                "This may be a sign of incorrect optimizer behavior"
            )

        self.load_accumulators_into_averager_()
        self._accumulators_used_in_step = True
        self._new_averaged_grads = True

        control.weight = self.local_samples_accumulated if weight is None else weight
        if reset_accumulators:
            self.reset_accumulated_grads_()
        control.allow_allreduce()

        return control.result(timeout) if wait else control

    @torch.no_grad()
    def load_accumulators_into_averager_(self):
        """load locally accumulated gradients into the averager for aggregation"""
        # divide locally accumulated gradients by the number of times they were accumulated
        grad_scale = (
            (1.0 / self.local_times_accumulated)
            if self.local_times_accumulated != 0
            else 0.0
        )
        with self.get_tensors() as averaged_grads:
            for grad_acc, averaged_grad in zip(
                self._grad_accumulators(), averaged_grads
            ):
                averaged_grad.copy_(grad_acc, non_blocking=True).mul_(grad_scale)

    @torch.no_grad()
    def reset_accumulated_grads_(self):
        """reset averager-internal gradient accumulators and the denominator"""
        self._accumulators_used_in_step = False
        self.local_samples_accumulated = self.local_times_accumulated = 0
        self._anchor_batch_size = None
        for grad_buf in self._grad_accumulators():
            grad_buf.zero_()

    """
    Needs this wrapper class to ensure device is set properly when averaging gradients
    See: https://github.com/learning-at-home/hivemind/blob/d20e81017481aa2028efc33217522248aabd7d95/hivemind/optim/grad_averager.py#L224
    """

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

    def notify_used_averaged_gradients(self):
        """Notify averager that the results of a previous averaging round are accounted for"""
        self._new_averaged_grads = False
