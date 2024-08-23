import asyncio
import logging
from contextlib import contextmanager
from enum import Enum
from typing import (Any, AsyncIterator, Dict, Iterable, Iterator, Optional,
                    Sequence, Union)

import bittensor as bt
import hivemind
import hivemind.averaging.averager
import torch
from hivemind.averaging.allreduce import (AllreduceException, AllReduceRunner,
                                          AveragingMode)
from hivemind.averaging.control import AveragingStage, StepControl
from hivemind.averaging.group_info import GroupInfo
from hivemind.averaging.load_balancing import load_balance_peers
from hivemind.averaging.matchmaking import MatchmakingException
from hivemind.compression import deserialize_torch_tensor
from hivemind.dht import DHT
from hivemind.p2p import (P2PContext, P2PDaemonError, P2PHandlerError, PeerID)
from hivemind.proto import averaging_pb2
from hivemind.utils import MPFuture, get_logger
from hivemind.utils.asyncio import (aiter_with_timeout, amap_in_executor,
                                    as_aiter, attach_event_on_finished, azip,
                                    enter_asynchronously)
from hivemind.utils.timed_storage import DHTExpiration, get_dht_time

from template.proto import custom_averaging_pb2

GatheredData = Any

logger = get_logger(__name__)
logger.setLevel(logging.INFO)


class DTAveragerError(Exception):
    """Base exception for DTAverager"""


class BarrierError(DTAveragerError):
    """Exception raised for errors in the barrier process"""


class AllReduceError(DTAveragerError):
    """Exception raised for errors in the AllReduce process"""


class BarrierState(Enum):
    IDLE = 0
    LOOKING_FOR_GROUP = 1
    JOINED_GROUP = 2
    ASSEMBLING_GROUP = 3
    BARRIER_COMPLETE = 4


class DTAllReduceRunner(AllReduceRunner):
    def __init__(self, peerids_to_uids, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.count = 0
        self.peerids_to_uids = peerids_to_uids
        bt.logging.info(f"PeerID to UID mapping: {self.peerids_to_uids}")

    async def _communicate_with_peer(self, peer_id: PeerID):
        """Send a part of local tensors and metadata to a single peer, receive the average for that part of tensors"""
        uid = (
            self.peerids_to_uids[str(peer_id)]
            if str(peer_id) in self.peerids_to_uids.keys()
            else ""
        )
        bt.logging.info(
            f"UID:{uid} - PeerID:{peer_id} - communicate_with_peer started",
        )
        """Send a part of local tensors and metadata to a single peer, receive the average for that part of tensors"""
        peer_index = self.ordered_peer_ids.index(peer_id)
        if peer_id == self.peer_id:
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
                bt.logging.info("DTAllReduceRunner..")
                done_sending = asyncio.Event()
                inputs_aiter = attach_event_on_finished(
                    self._generate_input_for_peer(peer_index), done_sending,
                )

                bt.logging.info(
                    f"UID:{uid} - PeerID:{peer_id} - generate_input_for_peer started"
                )
                stream = await self._get_peer_stub(peer_id).rpc_aggregate_part(
                    inputs_aiter
                )
                bt.logging.info(
                    f"UID:{uid} - PeerID:{peer_id} - get_peer_stub finished"
                )

                if self.should_delay_results(self.peer_id):
                    bt.logging.info(
                        f"self.should_delay_results(self.peer_id) {self.should_delay_results(self.peer_id)}"
                    )
                    await done_sending.wait()

                bt.logging.info(
                    f"UID:{uid} - PeerID:{peer_id} - sending tensors finished"
                )
                part_index = 0

                def _try_deserialize(msg):
                    # bt.logging.info("try_deserialize..")

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
                bt.logging.info(
                    f"UID:{uid} - PeerID:{peer_id} - register_processed_part finished"
                )
                if (
                    part_index
                    != self.tensor_part_container.num_parts_by_peer[peer_index]
                ):
                    bt.logging.info(
                        f"part_index != self.tensor_part_container.num_parts_by_peer[peer_index]"
                    )
                    raise AllreduceException(
                        f"peer {peer_id} sent {part_index} parts, but we expected "
                        f"{self.tensor_part_container.num_parts_by_peer[peer_index]}"
                    )
            
            # except ConnectionResetError:
            #     logger.error(f"Connection reset by peer: {peer_id}")
            #     if peer_id == self.ordered_peer_ids[0]:
            #         bt.logging.error("Validator connection faulting")
            #     self.tensor_part_container.register_failed_reducer(peer_index)                    
            #     raise AllreduceException(f"Connection reset by PeerID: {peer_id} - UID:{uid}")

            except BaseException as e:
                if isinstance(
                    e, Exception
                ):  
                    logger.debug(
                        f"Caught {repr(e)} when communicating to {peer_id}",
                        exc_info=True,
                    )
                bt.logging.info(f"UID:{uid} - PeerID:{peer_id} - Failed reducer")
                self.tensor_part_container.register_failed_reducer(peer_index)
                raise

    def __aiter__(self):
        return self.run()

    async def _handle_missing_senders(self):
        """Detect senders that should have sent tensors for averaging, but did not send anything within timeout"""
        try:
            await asyncio.wait_for(self.all_senders_started.wait(), self.sender_timeout)
        except asyncio.TimeoutError:
            for peer_id in self.sender_peer_ids:
                if (
                    peer_id not in self.active_senders
                    and peer_id not in self.banned_senders
                ):
                    await self._ban_sender(peer_id)

    # async def run(self) -> AsyncIterator[torch.Tensor]:
    #     """Run all-reduce, return differences between averaged and original tensors as they are computed"""
    #     pending_tasks = set()
    #     bt.logging.info("Running AllReducerRunner")
    #     bt.logging.info(
    #         f"self.tensor_part_container.num_parts_by_peer {self.tensor_part_container.num_parts_by_peer}"
    #     )
    #     if (
    #         self.tensor_part_container.num_parts_by_peer[
    #             self.ordered_peer_ids.index(self.peer_id)
    #         ]
    #         != 0
    #     ):
    #         pending_tasks.add(asyncio.create_task(self._handle_missing_senders()))

    #     try:
    #         if len(self.sender_peer_ids) == 0:
    #             logger.debug(
    #                 f"{self} - finished all-reduce early: all peers are auxiliaries ({self.modes})"
    #             )
    #             self.finalize()

    #         elif self.peer_id in self.sender_peer_ids:
    #             uid = (
    #                 self.peerids_to_uids[str(self.peer_id)]
    #                 if self.peer_id in self.peerids_to_uids.keys()
    #                 else ""
    #             )
    #             bt.logging.info(
    #                 f"UID:{uid} - PeerID:{self.peer_id} peer_id in sender_peer_ids"
    #             )

    #             for peer_id, parts in zip(
    #                 self.ordered_peer_ids, self.tensor_part_container.num_parts_by_peer
    #             ):
    #                 if parts != 0:
    #                     pending_tasks.add(
    #                         asyncio.create_task(self._communicate_with_peer(peer_id))
    #                     )

    #             bt.logging.info(f"Succesfully Communicated With All Peers")

    #             async for averaged_tensor_delta in self.tensor_part_container.iterate_output_tensors():
    #                 yield averaged_tensor_delta

    #             bt.logging.info(f"Iterate Output Tensors Finished")

    #             self.finalize()

    #             bt.logging.info(f"Finalize Finished")

    #         else:  # auxiliary peer
    #             await self.tensor_part_reducer.finished.wait()
    #             self.finalize()

    #     except BaseException as e:
    #         bt.logging.info(f"All Reduce Runner failed with error {e}")

    #         self.finalize(exception=e)
    #         for task in pending_tasks:
    #             task.cancel()
    #         raise

    #     finally:
    #         for task in pending_tasks:
    #             try:
    #                 await task
    #             except asyncio.CancelledError:
    #                 pass
    #             except Exception as inner_exc:
    #                 logger.debug(f"Task {task} failed with {inner_exc}", exc_info=True)

    async def _generate_input_for_peer(
        self, peer_index: int, uid: int, peer_id: PeerID
    ) -> AsyncIterator[averaging_pb2.AveragingData]:
        try:
            parts_aiter = self.tensor_part_container.iterate_input_parts_for(peer_index)
            first_part = await anext(parts_aiter)
            yield averaging_pb2.AveragingData(
                code=averaging_pb2.PART_FOR_AVERAGING,
                group_id=self.group_id,
                tensor_part=first_part,
                weight=self.weight,
            )
            bt.logging.info(
                f"UID:{uid} - PeerID:{peer_id} - generate_input_for_peer finished"
            )

            async for part in parts_aiter:
                bt.logging.info("_generate_input_for_peers for loop")
                yield averaging_pb2.AveragingData(tensor_part=part, weight=self.weight)
            bt.logging.info(f"_generate_input_for_peer.. {peer_index} done")

        except Exception as e:
            logger.error(
                f"Error preparing input for peer {self.ordered_peer_ids[peer_index]}: {e}"
            )

            raise e

    async def _ban_sender(self, peer_id: PeerID):
        uid = (
            self.peerids_to_uids[str(peer_id)]
            if peer_id in self.peerids_to_uids.keys()
            else ""
        )
        bt.logging.info(f"UID:{uid} - PeerID:{peer_id} - Ban sender")
        async with self.banlock:
            if peer_id not in self.banned_senders:
                self.banned_senders.add(peer_id)
                self.tensor_part_reducer.on_sender_failed(
                    self.sender_peer_ids.index(peer_id)
                )
                error_message = f"UID:{uid} - PeerID:{peer_id} - Banning peer {peer_id} due to a failure."
                logger.error(error_message)

                raise Exception(error_message)


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

        # When custom_group_info is provided, bypass matchmaking and proceed directly
        custom_group_info = kwargs.get("custom_group_info", None)
        peerids_to_uids = kwargs.get("peerids_to_uids", None)

        if custom_group_info is not None:
            self._outer_pipe.send(
                (
                    "_step_custom",
                    [],
                    dict(
                        step=step,
                        future_for_init=future_for_init,
                        custom_group_info=custom_group_info,
                        peerids_to_uids=peerids_to_uids
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
            bt.logging.info("Wait Step")
            bt.logging.info(step.result())
        else:
            bt.logging.info("Non Wait Step")
            bt.logging.info(step)

        return step.result() if wait else step

    async def _step_custom(
        self,
        *,
        step: StepControl,
        future_for_init: MPFuture,
        custom_group_info: GroupInfo,
        peerids_to_uids: dict = {},
    ):
        self.current_group_info = custom_group_info
        self.p2p = await self.dht.replicate_p2p()

        try:
            trigger, cancel = MPFuture(), MPFuture()
            step.attach(trigger, cancel)
            future_for_init.set_result((trigger, cancel))

            while not step.done():
                try:
                    self._pending_groups_registered.clear()
                    step.stage = AveragingStage.LOOKING_FOR_GROUP

                    bt.logging.info("Starting distributed barrier.")
                    await self._run_distributed_barrier(step, custom_group_info)
                    bt.logging.info("Finished waiting for group to assemble.")

                    bt.logging.info("Running AllReduce..")
                    await self._run_allreduce(step, custom_group_info, peerids_to_uids)
                    # Averaging is finished, loop will now exit
                
                # except ConnectionResetError as e:
                #     logger.error(f"Connection reset error: {e}")
                #     if (
                #         step.done()
                #         or not step.allow_retries
                #         or get_dht_time() >= step.deadline
                #     ):
                #         if not step.cancelled():
                #             logger.exception(e)
                #         if not step.done():
                #             step.set_exception(e)
                #     else:
                #         logger.warning(f"ConnectionResetError occurred, retrying...")
                #         await asyncio.sleep(0.5)
                        
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
            bt.logging.info("Finally..")
            step.stage = AveragingStage.FINISHED
            if not step.done():
                step.set_exception(
                    RuntimeError(
                        "Internal sanity check failed: averager.step left future pending."
                        " Please report this to hivemind issues."
                    )
                )

    async def rpc_join_barrier(
        self, request: custom_averaging_pb2.JoinBarrierRequest, context: P2PContext
    ) -> custom_averaging_pb2.JoinBarrierResponse:
        logger.debug(f"Received join barrier request from {context.remote_id}")
        logger.debug(f"Current barrier state: {self.barrier_state}")
        logger.debug(
            f"Barrier group peers: {self.barrier_group.peer_ids if self.barrier_group else 'None'}"
        )

        if self.barrier_state != BarrierState.ASSEMBLING_GROUP:
            logger.warning(
                f"Rejecting join request: incorrect state {self.barrier_state}"
            )
            return custom_averaging_pb2.JoinBarrierResponse(
                code=custom_averaging_pb2.BarrierResponseCode.REJECTED
            )

        peer_id = PeerID(request.peer_id)
        if peer_id not in self.barrier_group.peer_ids:
            logger.warning(f"Rejecting join request: peer {peer_id} not in group")
            return custom_averaging_pb2.JoinBarrierResponse(
                code=custom_averaging_pb2.BarrierResponseCode.REJECTED
            )

        self.ready_peers[peer_id].set()
        logger.debug(f"Peer {peer_id} joined the barrier")

        return custom_averaging_pb2.JoinBarrierResponse(
            code=custom_averaging_pb2.BarrierResponseCode.ACCEPTED
        )

    async def rpc_notify_barrier_complete(
        self,
        request: custom_averaging_pb2.BarrierCompleteNotification,
        context: P2PContext,
    ) -> custom_averaging_pb2.BarrierCompleteResponse:
        self.barrier_state = BarrierState.BARRIER_COMPLETE
        self.barrier_complete.set()
        bt.logging.info(f"Peer {self.peer_id} received barrier complete notification.")

        return custom_averaging_pb2.BarrierCompleteResponse(
            code=custom_averaging_pb2.BarrierResponseCode.ACCEPTED
        )

    @contextmanager
    def _barrier_context(self, custom_group_info):
        self.barrier_state = BarrierState.LOOKING_FOR_GROUP
        self.barrier_group = custom_group_info
        self.barrier_complete = asyncio.Event()
        self.ready_peers = {
            peer_id: asyncio.Event() for peer_id in custom_group_info.peer_ids
        }
        try:
            yield
        finally:
            self.barrier_state = BarrierState.IDLE
            self.barrier_complete.clear()
            self.ready_peers.clear()

    async def _distributed_barrier(self, custom_group_info: GroupInfo):
        try:
            with self._barrier_context(custom_group_info):
                leader_id = custom_group_info.peer_ids[0]

                if self.peer_id == leader_id:
                    bt.logging.info(f"Peer {self.peer_id} is the leader")
                    self.barrier_state = BarrierState.ASSEMBLING_GROUP
                    await self._leader_coordinate_barrier()
                else:
                    bt.logging.info(f"Peer {self.peer_id} is a follower.")
                    await self._follower_join_barrier(leader_id)

                bt.logging.info(f"Waiting for barrier_complete for {self.peer_id}")
                await asyncio.wait_for(self.barrier_complete.wait(), timeout=60) # TODO set to a config param
                bt.logging.info(f"Barrier complete for {self.peer_id}")
        
        # except ConnectionResetError as e:
        #     logger.error(f"Connection reset during barrier for peer {self.peer_id}: {e}")
        #     if self.peer_id == leader_id:
        #         bt.logging.error("Validator connection faulting")
        #     raise BarrierError("Connection reset during barrier") from e
        except asyncio.TimeoutError:
            logger.error(f"Barrier timeout for peer {self.peer_id}")
            raise BarrierError("Barrier timeout") from None
        except Exception as e:
            logger.error(
                f"Error in distributed_barrier for peer {self.peer_id}: {str(e)}",
                exc_info=True,
            )
            raise BarrierError(f"Error in distributed_barrier: {str(e)}") from e

    async def _leader_coordinate_barrier(self):
        bt.logging.info(f"Leader {self.peer_id} starting barrier coordination")
        self.ready_peers = {
            peer_id: asyncio.Event() for peer_id in self.barrier_group.peer_ids
        }
        self.ready_peers[self.peer_id].set()

        async def notify_follower(peer_id: PeerID):
            try:
                follower_stub = type(self).get_stub(
                    self.p2p, peer_id, namespace=self.prefix
                )
                await follower_stub.rpc_notify_barrier_complete(
                    custom_averaging_pb2.BarrierCompleteNotification(
                        group_id=self.barrier_group.group_id
                    )
                )
            except Exception as e:
                logger.error(f"Failed to notify follower {peer_id}: {e}")

        try:
            bt.logging.info(f"Leader waiting for peers to be ready.")
            await asyncio.gather(
                *(
                    self._wait_for_peer_ready(peer_id)
                    for peer_id in self.barrier_group.peer_ids
                    if peer_id != self.peer_id
                )
            )
            self.barrier_state = BarrierState.BARRIER_COMPLETE
            self.barrier_complete.set()
            bt.logging.info(f"Leader {self.peer_id} completed barrier coordination.")

            # Notify all followers
            await asyncio.gather(
                *(
                    notify_follower(peer_id)
                    for peer_id in self.barrier_group.peer_ids
                    if peer_id != self.peer_id
                )
            )

            bt.logging.info(f"Leader {self.peer_id} notified all followers.")
        
        # except ConnectionResetError as e:
        #     logger.error(f"Connection reset during leader coordination: {e}")
        #     bt.logging.error("Validator connection faulting")
        #     raise
        except Exception as e:
            logger.error(f"Error in leader_coordinate_barrier: {e}")
            raise

    async def _follower_join_barrier(self, leader_id: PeerID):
        bt.logging.info(
            f"Follower {self.peer_id} attempting to join barrier led by {leader_id}."
        )
        max_retries = 3 # TODO set this to reasonable value/config param
        for attempt in range(max_retries):
            try:
                self.barrier_state = BarrierState.JOINED_GROUP
                leader_stub = type(self).get_stub(
                    self.p2p, leader_id, namespace=self.prefix
                )
                response = await asyncio.wait_for(
                    leader_stub.rpc_join_barrier(
                        custom_averaging_pb2.JoinBarrierRequest(
                            group_id=self.barrier_group.group_id,
                            peer_id=self.peer_id.to_bytes(),
                        )
                    ),
                    timeout=30,
                )
                logger.debug(f"Received response from leader: {response.code}")
                if response.code == custom_averaging_pb2.BarrierResponseCode.ACCEPTED:
                    bt.logging.info(
                        f"Follower {self.peer_id} successfully joined barrier, waiting for completion."
                    )
                    await self._wait_for_barrier_completion()
                    bt.logging.info(
                        f"Follower {self.peer_id} successfully waited for completion."
                    )
                    return
                elif (
                    response.code
                    == custom_averaging_pb2.BarrierResponseCode.BARRIER_COMPLETE
                ):
                    self.barrier_state = BarrierState.BARRIER_COMPLETE
                    self.barrier_complete.set()
                    return
                else:
                    logger.warning(
                        f"Failed to join barrier, attempt {attempt + 1}/{max_retries}"
                    )
                    if attempt == max_retries - 1:
                        raise MatchmakingException(
                            f"Failed to join barrier after {max_retries} attempts"
                        )
                    await asyncio.sleep(1)  # Wait before retrying
            
            # except ConnectionResetError as e:
            #     logger.error(f"Connection reset while joining barrier: {e}")
            #     if attempt == max_retries - 1:
            #         raise MatchmakingException(f"Failed to join barrier due to connection reset")
            #     await asyncio.sleep(0.2)  # Wait before retrying
            except Exception as e:
                logger.error(
                    f"Error in follower_join_barrier attempt {attempt + 1}: {e}"
                )
                if attempt == max_retries - 1:
                    raise

    async def _wait_for_barrier_completion(self):
        timeout = 60  # Adjust as needed
        try:
            await asyncio.wait_for(self.barrier_complete.wait(), timeout=timeout)
            bt.logging.info(f"Peer {self.peer_id} barrier completed")
        except asyncio.TimeoutError:
            logger.error(
                f"Peer {self.peer_id} timed out waiting for barrier completion"
            )
            raise MatchmakingException("Barrier timeout")

    async def _wait_for_peer_ready(self, peer_id: PeerID):
        try:
            bt.logging.info(f"_wait_for_peer_ready {peer_id}")
            await asyncio.wait_for(
                self.ready_peers[peer_id].wait(), timeout=30
            )  # Adjust timeout as needed
        except asyncio.TimeoutError:
            logger.warning(f"Timeout waiting for peer {peer_id}")
            raise

    async def _run_distributed_barrier(
        self, step: StepControl, custom_group_info: GroupInfo
    ):
        matchmaking_task = asyncio.create_task(
            self._distributed_barrier(custom_group_info)
        )
        check_cancel_task = asyncio.create_task(step.wait_for_cancel())

        done, pending = await asyncio.wait(
            {matchmaking_task, check_cancel_task},
            return_when=asyncio.FIRST_COMPLETED,
        )

        if step.cancelled():
            bt.logging.info("Step cancelled, aborting distributed barrier")
            matchmaking_task.cancel()
            raise asyncio.CancelledError()
        else:
            bt.logging.info(
                "Distributed barrier completed, cancelling check_cancel_task"
            )
            check_cancel_task.cancel()

        await matchmaking_task

    async def _run_allreduce(self, step: StepControl, custom_group_info: GroupInfo, peerids_to_uids: Dict):
        with self._register_allreduce_group(custom_group_info):
            bt.logging.info("Running AllReduce.")
            assert (
                custom_group_info.group_id in self._running_groups
            ), "Group was not properly registered"

            step.stage = AveragingStage.RUNNING_ALLREDUCE
            step.set_result(
                await asyncio.wait_for(
                    self._aggregate_with_group(
                        custom_group_info,
                        tensor_infos=self.tensor_infos,
                        weight=step.weight,
                        peerids_to_uids=peerids_to_uids,
                        **self.allreduce_kwargs,
                    ),
                    timeout=self._allreduce_timeout,
                )
            )

    async def _aggregate_with_group(
        self,
        group_info: GroupInfo,
        min_vector_size: int,
        peerids_to_uids: dict,
        **kwargs,
    ) -> GatheredData:
        """Run aggregation in a given group and update tensors in place, return gathered metadata"""
        try:
            num_peers = len(group_info.peer_ids)
            peer_fractions = [1.0 / num_peers] * num_peers
            group_id = group_info.group_id

            async with enter_asynchronously(self.get_tensors()) as local_tensors:
                runner = DTAllReduceRunner(
                    peerids_to_uids=peerids_to_uids,
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
            raise AllReduceError(f"Error during AllReduce: {str(e)}") from e


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
        # Custom matchmaking
        self.barrier_state = BarrierState.IDLE
        self.barrier_group = None
        self.barrier_complete = asyncio.Event()
        self.ready_peers = {}

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
        bt.logging.info(f"control {control}")
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
        bt.logging.info(f"control.stage {control.stage}")

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
