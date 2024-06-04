import asyncio
import base64
import contextlib
import logging
import random
import re
from contextlib import contextmanager
from itertools import chain
from typing import (Any, AsyncIterator, Dict, Iterable, Iterator, Optional,
                    Sequence, Tuple, Union)

import bittensor as bt
import hivemind
import hivemind.averaging
import hivemind.averaging.averager
import torch
from hivemind.averaging.allreduce import (AllreduceException, AllReduceRunner,
                                          AveragingMode)
from hivemind.averaging.control import AveragingStage, StepControl
from hivemind.averaging.group_info import GroupInfo
from hivemind.averaging.load_balancing import load_balance_peers
from hivemind.averaging.matchmaking import MatchmakingException
from hivemind.compression import (deserialize_torch_tensor,
                                  serialize_torch_tensor)
from hivemind.dht import DHT
from hivemind.optim.progress_tracker import LocalTrainingProgress
from hivemind.p2p import P2PDaemonError, P2PHandlerError, PeerID
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
logger.setLevel(logging.INFO)


class DTAllReduceRunner(AllReduceRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    async def _communicate_with_peer(self, peer_id: PeerID):
        print("WE ARE HERE NOW!")
        print("DTAllReduceRunner sender + reducer timeout", self.sender_timeout, self.reducer_timeout)
        """Send a part of local tensors and metadata to a single peer, receive the average for that part of tensors"""
        peer_index = self.ordered_peer_ids.index(peer_id)
        if peer_id == self.peer_id:
            print("self.peer_id..")
            sender_index = self.sender_peer_ids.index(peer_id)
            for part_index, tensor_part in enumerate(self.parts_for_local_averaging):
                averaged_part = await self.tensor_part_reducer.accumulate_part(
                    sender_index, part_index, tensor_part, weight=self.weight
                )
                self.tensor_part_container.register_processed_part(
                    peer_index, part_index, averaged_part - tensor_part
                )

        else:
            try:
                print("DTAllReduceRunner..")
                done_sending = asyncio.Event()
                inputs_aiter = attach_event_on_finished(
                    self._generate_input_for_peer(peer_index), done_sending
                )
                stream = await self._get_peer_stub(peer_id).rpc_aggregate_part(
                    inputs_aiter
                )
                print("_get_peer_stub done..")

                if self.should_delay_results(self.peer_id):
                    await done_sending.wait()

                part_index = 0

                def _try_deserialize(msg):
                    print("try_deserialize..")

                    if msg.code != averaging_pb2.AVERAGED_PART:
                        raise AllreduceException(
                            f"{peer_id} sent {averaging_pb2.MessageCode.Name(msg.code)}"
                        )
                    return deserialize_torch_tensor(msg.tensor_part), msg

                async for delta, msg in amap_in_executor(
                    _try_deserialize,
                    aiter_with_timeout(stream, self.reducer_timeout),
                    max_prefetch=self.tensor_part_container.prefetch,
                ):
                    self.tensor_part_container.register_processed_part(
                        peer_index, part_index, delta
                    )
                    part_index += 1
                    print("amap_in_executor..")

                if (
                    part_index
                    != self.tensor_part_container.num_parts_by_peer[peer_index]
                ):
                    raise AllreduceException(
                        f"peer {peer_id} sent {part_index} parts, but we expected "
                        f"{self.tensor_part_container.num_parts_by_peer[peer_index]}"
                    )
            except BaseException as e:
                if isinstance(e, Exception):
                    logger.debug(
                        f"Caught {repr(e)} when communicating to {peer_id}",
                        exc_info=True,
                    )
                # self.finalize(exception=e)
                # ? Remove fault-tolerant method here
                self.tensor_part_container.register_failed_reducer(peer_index)
                raise

    #! Test fault-tolerance here:
    async def rpc_aggregate_part(
        self, stream, context
    ) -> AsyncIterator[averaging_pb2.AveragingData]:
        """
        Handles the aggregation of tensor parts sent by peers. If an error is encountered, such as a timeout
        or failure in communication, it directly raises an exception.
        """
        try:
            #
            # # condition = np.random.choice(["FAIL_SENDING", "SLOW_REDUCE", "CANCEL"])
            # test_fault = True
            # if test_fault:
            #     condition = "FAIL_SENDING"

            #     async for message in super().rpc_aggregate_part(stream, context):
            #         self.count+=1
            #         yield message
            #         if self.count == 2:
            #             if condition == "FAIL_SENDING":
            #                 yield averaging_pb2.AveragingData(code=averaging_pb2.INTERNAL_ERROR)
            #                 break
            #             elif condition == "SLOW_REDUCE":
            #                 await asyncio.sleep(10)
            #             elif condition == "CANCEL":
            #                 yield averaging_pb2.AveragingData(code=averaging_pb2.CANCELLED)
            # else:
            async for message in super().rpc_aggregate_part(stream, context):
                print("rpc_aggregate_part..")
                yield message

        except Exception as e:
            logger.error(f"RPC aggregation error with peer {context.remote_id}: {e}")
            raise e

    async def _generate_input_for_peer(
        self, peer_index: int
    ) -> AsyncIterator[averaging_pb2.AveragingData]:
        # try:
        parts_aiter = self.tensor_part_container.iterate_input_parts_for(peer_index)
        first_part = await anext(parts_aiter)
        yield averaging_pb2.AveragingData(
            code=averaging_pb2.PART_FOR_AVERAGING,
            group_id=self.group_id,
            tensor_part=first_part,
            weight=self.weight,
        )
        print("_generate_input_for_peer..")
        async for part in parts_aiter:
            print("_generate_input_for_peer for loop")
            yield averaging_pb2.AveragingData(tensor_part=part, weight=self.weight)

        # except Exception as e:
        #     logger.error(f"Error preparing input for peer {self.ordered_peer_ids[peer_index]}: {e}")
        #     self.finalize(exception=e)
        #     raise e

    async def _ban_sender(self, peer_id: PeerID):
        print("ban sender..")
        async with self.banlock:
            if peer_id not in self.banned_senders:
                self.banned_senders.add(peer_id)
                # ? Remove fault-tolerant method here:
                self.tensor_part_reducer.on_sender_failed(
                    self.sender_peer_ids.index(peer_id)
                )
                error_message = f"Banning peer {peer_id} due to a failure."
                logger.error(error_message)
                # self.finalize(exception=error_message)
                # raise Exception(error_message)


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
        print("DTAverager.. self.next_chunk_timeout", self.next_chunk_timeout)
        print("DTAverager.. timeout", timeout)
        # When custom_group_info is provided, bypass matchmaking and proceed directly
        custom_group_info = kwargs.get("custom_group_info", None)

        if custom_group_info is not None:
            self._outer_pipe.send(
                (
                    "_step_custom",
                    [],
                    dict(
                        step=step,
                        future_for_init=future_for_init,
                        custom_group_info=custom_group_info,
                    ),
                )
            )
            step.attach(*future_for_init.result())
        else:
            # Default behavior: initiate matchmaking and proceed as originally designed
            self._outer_pipe.send(
                ("_step", [], dict(step=step, future_for_init=future_for_init))
            )
            step.attach(*future_for_init.result())

        if not require_trigger:
            step.allow_allreduce()
        if wait:
            print("wait step")
            print(step.result())
        else:
            print("non wait step")
            print(step)
        return step.result() if wait else step

    async def _step_custom(
        self,
        *,
        step: StepControl,
        future_for_init: MPFuture,
        custom_group_info: GroupInfo,
    ):
        try:
            trigger, cancel = MPFuture(), MPFuture()
            step.attach(trigger, cancel)
            future_for_init.set_result((trigger, cancel))

            while not step.done():
                try:
                    self._pending_groups_registered.clear()
                    step.stage = AveragingStage.LOOKING_FOR_GROUP

                    async def distributed_barrier():
                        key = f"{base64.b64encode(custom_group_info.group_id).decode('utf-8')}.barrier"
                        print(key)
                        peer_id_strs = [
                            peer_id.to_string()
                            for peer_id in custom_group_info.peer_ids
                        ]
                        print("HERE YO..")
                        expiration_time = (
                            get_dht_time() + 300 # TODO propogate timeout to here??
                        )  
                        print("NOT HERE YO..")
                        print(expiration_time)

                        # Register this peer
                        store_result = self.dht.store(
                            key,
                            subkey=self.peer_id.to_string(),
                            value=True,
                            expiration_time=expiration_time,
                        )
                        if not store_result:
                            raise Exception(
                                f"Failed to store peer {self.peer_id} in DHT"
                            )

                        print("later:", get_dht_time, expiration_time)
                        while get_dht_time() < expiration_time:
                            # Check if all peers have registered
                            gathered = self.dht.get(key, latest=True)
                            if gathered:
                                registered_peers = gathered.value.keys()
                                if all(
                                    peer_id in registered_peers
                                    for peer_id in peer_id_strs
                                ):
                                    if not step.triggered:
                                        step.stage = AveragingStage.AWAITING_TRIGGER
                                        await step.wait_for_trigger()
                                    print(registered_peers)
                                    return  # All peers are ready
                            await asyncio.sleep(0.5)

                        raise TimeoutError(
                            "Distributed barrier timed out waiting for peers"
                        )

                    # Concurrently handle matchmaking and cancellation
                    matchmaking_task = asyncio.create_task(distributed_barrier())
                    check_cancel_task = asyncio.create_task(step.wait_for_cancel())

                    print("Here1")
                    await asyncio.wait(
                        {matchmaking_task, check_cancel_task},
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    print("Here2")

                    if step.cancelled():
                        print("CANCELLING APPARENTLY..")
                        matchmaking_task.cancel()
                        raise asyncio.CancelledError()
                    else:
                        print("checking CANCELLING APPARENTLY..")
                        check_cancel_task.cancel()

                    await matchmaking_task
                    print("Finished waiting for group to assemble")

                    with self._register_allreduce_group(custom_group_info):
                        print("Running AllReduce..")
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
            print("Finally..")
            step.stage = AveragingStage.FINISHED
            if not step.done():
                step.set_exception(
                    RuntimeError(
                        "Internal sanity check failed: averager.step left future pending."
                        " Please report this to hivemind issues."
                    )
                )

    async def _aggregate_with_group(
        self, group_info: GroupInfo, min_vector_size: int, **kwargs
    ) -> GatheredData:
        """Run aggregation in a given group and update tensors in place, return gathered metadata"""
        try:
            num_peers = len(group_info.peer_ids)
            peer_fractions = [1.0 / num_peers] * num_peers
            group_id = group_info.group_id

            async with enter_asynchronously(self.get_tensors()) as local_tensors:
                runner = DTAllReduceRunner(
                    p2p=self._p2p,
                    servicer_type=type(self),
                    prefix=self.prefix,
                    group_id=group_id,
                    tensors=local_tensors,
                    ordered_peer_ids=group_info.peer_ids,
                    peer_fractions=peer_fractions,
                    **kwargs,
                )
                assert (
                    group_id in self._running_groups
                ), f"Group id {group_id} was not registered in _register_allreduce_group"
                self._running_groups[group_info.group_id].set_result(runner)

                if (
                    runner.modes[group_info.peer_ids.index(self.peer_id)]
                    != AveragingMode.AUX
                ):
                    async for tensor, update in azip(as_aiter(*local_tensors), runner):
                        tensor.add_(update, alpha=self._averaging_alpha)
                        self.last_updated = get_dht_time()
                        self._state_updated.set()

                else:
                    async for _ in runner:
                        raise ValueError(
                            "aux peers should not receive averaged tensors"
                        )

                return group_info
        except BaseException as e:
            if isinstance(e, Exception):
                logger.exception(e)
            raise MatchmakingException(f"Unable to run All-Reduce: {e}")


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

    @torch.no_grad()
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
        yield from self._grads_from_parameters() if self.reuse_grad_buffers else self._local_accumulators

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
        print(control.stage)

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

    def notify_used_averaged_gradients(self):
        """Notify averager that the results of a previous averaging round are accounted for"""
        self._new_averaged_grads = False


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
            return

        metadata, flat_tensors = loaded_state
        if (not isinstance(metadata.get("epoch"), int)) or metadata[
            "epoch"
        ] < self.local_epoch:
            logger.warning(
                "Cowardly refusing to load state from peer: peer's epoch is behind our local epoch"
            )
            return

        loaded_parameters_and_extras = flat_tensors[:num_parameters_and_extras]
        loaded_opt_tensors = flat_tensors[num_parameters_and_extras:]
        if num_parameters_and_extras != len(loaded_parameters_and_extras):
            logger.error(
                "Failed to load state from peer, received parameters, extras or metadata"
            )
            return

        with torch.no_grad(), self.lock_averaged_tensors:
            try:
                load_optimizer_state(
                    self.optimizer, metadata["optimizer_metadata"], loaded_opt_tensors
                )
            except StopIteration:
                logger.warning(
                    "Failed to load state from peer, received inconsistent number of optimizer statistics"
                )
                return

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


def load_state_from_peer(self, epoch=None):
    if epoch == None:
        epoch = self.tracker.global_progress.epoch

    bt.logging.info("Model Weights Before Loading State")
    bt.logging.info([layer for layer in self.model.parameters()][-1][-10:])
    self.state_averager.load_final_state_from_peers(epoch)
    bt.logging.info("Model Weights After Loading State")
    bt.logging.info([layer for layer in self.model.parameters()][-1][-10:])

    with self.tracker.pause_updates():
        self.tracker.local_progress.epoch = self.tracker.global_progress.epoch
        self.local_epoch = self.tracker.local_progress.epoch
