import asyncio
import logging
import os
import time
from enum import Enum, auto
from typing import AsyncIterator, Dict, List

import hivemind
import torch
import torch.nn as nn
from colorama import Fore, Style
from hivemind.averaging.averager import *
from hivemind.averaging.group_info import GroupInfo
from hivemind.averaging.load_balancing import load_balance_peers
from hivemind.averaging.matchmaking import MatchmakingException
from hivemind.dht.routing import DHTID
from hivemind.optim.grad_averager import GradientAverager
from hivemind.proto import averaging_pb2
from hivemind.utils import use_hivemind_log_handler
from hivemind.utils.asyncio import (aenumerate, as_aiter, azip,
                                    enter_asynchronously)
from hivemindy import AveragingMode, DTAllReduceRunner, DTGradientAverager

# Logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

use_hivemind_log_handler("nowhere")

# Delete the logfile if it exists
logfile = "logfile.log"
if os.path.exists(logfile):
    os.remove(logfile)

# Create a file handler
handler = logging.FileHandler(logfile)

# Create a formatter and add it to the handler
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(handler)


class Fault(Enum):
    NONE = auto()
    FAIL_BEFORE = auto()
    FAIL_SENDING = auto()
    SLOW_SENDING = auto()
    FAIL_REDUCING = auto()
    SLOW_REDUCING = auto()
    CANCEL = auto()


def launch_dht_instances(n_peers: int, **kwargs) -> List[hivemind.DHT]:
    dhts = [hivemind.DHT(start=True, **kwargs)]
    initial_peers = dhts[0].get_visible_maddrs()

    dhts.extend(
        hivemind.DHT(initial_peers=initial_peers, start=True, await_ready=False, **kwargs)
        for _ in range(n_peers - 1)
    )
    for process in dhts[1:]:
        process.wait_until_ready()

    return dhts


class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.param1 = nn.Parameter(torch.randn(16, 1024))
        self.param2 = nn.Parameter(torch.randn(3, 8192))
        self.param3 = nn.Parameter(torch.randn(4, 4, 4))
        self.param4 = nn.Parameter(torch.randn(1024, 1024))

    def forward(self, x1, x2, x3, x4):
        y1 = x1 + self.param1
        y2 = -(x2 + self.param2)
        y3 = 2 * (x3 + self.param3)
        y4 = x4 + self.param4

        return [y1, y2, y3, y4]


class FaultyGradientAverager(GradientAverager):
    # class FaultyGradientAverager(hivemind.DecentralizedAverager):
    def __init__(self, *args, fault: Fault = Fault.NONE, **kwargs):
        self.fault = fault
        super().__init__(*args, **kwargs)

    async def _aggregate_with_group(
        self, group_info: GroupInfo, min_vector_size: int, **kwargs
    ):
        """Run All-Reduce in a given group and update tensors in place, return gathered metadata"""
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
            download_bandwidths = [
                thr if mode != AveragingMode.CLIENT else 0.0
                for thr, mode in zip(bandwidths, modes)
            ]
            peer_fractions = await asyncio.get_event_loop().run_in_executor(
                None,
                load_balance_peers,
                self.total_size,
                download_bandwidths,
                min_vector_size,
            )

            if self.fault == Fault.FAIL_BEFORE:
                raise Exception(
                    f"Oops, I (peerID {[str(self.peer_id)]}) failed before aggregation!"
                )

            async with enter_asynchronously(self.get_tensors()) as local_tensors:
                allreduce = FaultyAllReduceRunner(
                    p2p=self._p2p,
                    servicer_type=type(self),
                    prefix=self.prefix,
                    group_id=group_info.group_id,
                    tensors=local_tensors,
                    ordered_peer_ids=group_info.peer_ids,
                    peer_fractions=peer_fractions,
                    modes=modes,
                    fault=self.fault,
                    **kwargs,
                )

                self._running_groups[group_info.group_id].set_result(allreduce)
                # TODO maybe this can be extracted into a method that checks if register_... context is active.

                if modes[group_info.peer_ids.index(self.peer_id)] != AveragingMode.AUX:
                    # iter_results = allreduce.run()
                    async for tensor, update in azip(
                        as_aiter(*local_tensors), allreduce
                    ):
                        # all-reduce is performed asynchronously while iterating
                        tensor.add_(update, alpha=self._averaging_alpha)
                    self._state_updated.set()

                else:
                    async for _ in allreduce:  # trigger all-reduce by iterating
                        raise ValueError(
                            "aux peers should not receive averaged tensors"
                        )

                return user_gathered
        except BaseException as e:
            logger.exception(e)
            raise MatchmakingException(f"Unable to run All-Reduce: {e}")


class FaultyAllReduceRunner(AllReduceRunner):
    def __init__(self, *args, fault: Fault, **kwargs):
        self.fault = fault
        super().__init__(*args, **kwargs)

    async def rpc_aggregate_part(
        self, stream, context
    ) -> AsyncIterator[averaging_pb2.AveragingData]:
        if self.fault in (Fault.FAIL_REDUCING, Fault.SLOW_REDUCING):
            async for i, message in aenumerate(
                super().rpc_aggregate_part(stream, context)
            ):
                yield message
                if i == 2:
                    if self.fault == Fault.FAIL_SENDING:
                        yield averaging_pb2.AveragingData(
                            code=averaging_pb2.INTERNAL_ERROR
                        )
                        break
                    else:
                        await asyncio.sleep(10)

        elif self.fault == Fault.CANCEL:
            yield averaging_pb2.AveragingData(code=averaging_pb2.CANCELLED)
        else:
            async for message in super().rpc_aggregate_part(stream, context):
                yield message

    async def _generate_input_for_peer(
        self, peer_index: int
    ) -> AsyncIterator[averaging_pb2.AveragingData]:
        parts_aiter = self.tensor_part_container.iterate_input_parts_for(peer_index)

        first_part = await anext(parts_aiter)
        yield averaging_pb2.AveragingData(
            code=averaging_pb2.PART_FOR_AVERAGING,
            group_id=self.group_id,
            tensor_part=first_part,
            weight=self.weight,
        )
        if self.fault in (Fault.FAIL_SENDING, Fault.SLOW_SENDING):
            last_reducer_index = (
                self.group_size
                - 1
                - (self.tensor_part_container.num_parts_by_peer[-1] == 0)
            )
            if peer_index == last_reducer_index:
                if self.fault == Fault.FAIL_SENDING:
                    raise Exception("Oops, I failed!")
                else:
                    await asyncio.sleep(10)
        async for part in parts_aiter:
            yield averaging_pb2.AveragingData(tensor_part=part, weight=self.weight)


class FaultyDTGradientAverager(DTGradientAverager):
    def __init__(self, *args, fault: Fault = Fault.NONE, **kwargs):
        self.fault = fault
        super().__init__(*args, **kwargs)

    async def _aggregate_with_group(self, group_info: GroupInfo, min_vector_size: int, peerids_to_uids: dict, **kwargs):
    # async def _aggregate_with_group(
    #     self, group_info: GroupInfo, min_vector_size: int, **kwargs
    # ):
        try:
            num_peers = len(group_info.peer_ids)
            peer_fractions = [1.0 / num_peers] * num_peers
            group_id = group_info.group_id

            if self.fault == Fault.FAIL_BEFORE:
                raise Exception(
                    f"Oops, I (peerID {[str(self.peer_id)]}) failed before aggregation!"
                )

            async with enter_asynchronously(self.get_tensors()) as local_tensors:
                runner = FaultyDTAllReduceRunner(
                    peerids_to_uids=peerids_to_uids,
                    p2p=self._p2p,
                    servicer_type=type(self),
                    prefix=self.prefix,
                    group_id=group_id,
                    tensors=local_tensors,
                    ordered_peer_ids=group_info.peer_ids,
                    fault=self.fault,
                    peer_fractions=peer_fractions,
                    **kwargs,
                )
                assert (
                    group_info.group_id in self._running_groups
                ), "Group was not properly registered"
                self._running_groups[group_info.group_id].set_result(runner)

                if (
                    runner.modes[group_info.peer_ids.index(self.peer_id)]
                    != AveragingMode.AUX
                ):
                    async for tensor, update in azip(as_aiter(*local_tensors), runner):
                        tensor.add_(update, alpha=self._averaging_alpha)
                    self._state_updated.set()
                else:
                    async for _ in runner:
                        raise ValueError(
                            "aux peers should not receive averaged tensors"
                        )

                return runner.banned_senders if runner.banned_senders else True
        except BaseException as e:
            logger.exception(e)
            raise MatchmakingException(f"Unable to run All-Reduce: {e}")


class FaultyDTAllReduceRunner(DTAllReduceRunner):
    def __init__(self, *args, fault: Fault, **kwargs):
        self.fault = fault

        super().__init__(*args, **kwargs)

    async def rpc_aggregate_part(
        self, stream, context
    ) -> AsyncIterator[averaging_pb2.AveragingData]:
        if self.fault in (Fault.FAIL_REDUCING, Fault.SLOW_REDUCING):
            async for i, message in aenumerate(
                super().rpc_aggregate_part(stream, context)
            ):
                yield message
                if i == 2:
                    if self.fault == Fault.FAIL_REDUCING:
                        yield averaging_pb2.AveragingData(
                            code=averaging_pb2.INTERNAL_ERROR
                        )
                        break
                    else:  # SLOW_REDUCING
                        await asyncio.sleep(10)
        elif self.fault == Fault.CANCEL:
            yield averaging_pb2.AveragingData(code=averaging_pb2.CANCELLED)
        else:
            async for message in super().rpc_aggregate_part(stream, context):
                yield message

    async def _generate_input_for_peer(self, peer_index: int, uid: str, peer_id: hivemind.PeerID) -> AsyncIterator[averaging_pb2.AveragingData]:
        parts_aiter = self.tensor_part_container.iterate_input_parts_for(peer_index)
        first_part = await anext(parts_aiter)
        yield averaging_pb2.AveragingData(
            code=averaging_pb2.PART_FOR_AVERAGING,
            group_id=self.group_id,
            tensor_part=first_part,
            weight=self.weight,
        )
        if self.fault in (Fault.FAIL_SENDING, Fault.SLOW_SENDING):
            last_reducer_index = (
                self.group_size
                - 1
                - (self.tensor_part_container.num_parts_by_peer[-1] == 0)
            )
            if peer_index == last_reducer_index:
                if self.fault == Fault.FAIL_SENDING:
                    # print(f"{Fore.YELLOW}UID: {uid} failed sending...{Style.RESET_ALL}")
                    raise Exception(f"Oops, I failed during sending!")
                else:  # SLOW_SENDING
                    print(
                        f"{Fore.YELLOW}Woopsie, is slow in sending...{Style.RESET_ALL}"
                    )
                    await asyncio.sleep(10)
        async for part in parts_aiter:
            yield averaging_pb2.AveragingData(tensor_part=part, weight=self.weight)


def make_tensors():
    return [
        torch.rand(16, 1024),
        -torch.rand(3, 8192),
        2 * torch.randn(4, 4, 4),
        torch.randn(1024, 1024),
    ]


def run_test(
    fault0: Fault,
    fault1: Fault,
    dht_instances,
    models,
    peerids_to_uids,
    custom_group,
    use_original=False,
):
    allreduce_timeout = 5

    if use_original:
        averagers = [
            FaultyGradientAverager(
                model.parameters(),
                dht=dht,
                prefix="allreduce_test",
                request_timeout=0.3,
                min_matchmaking_time=1.0,
                next_chunk_timeout=0.5,
                part_size_bytes=2**16,
                start=True,
                fault=fault0 if i == 0 else fault1 if i == 1 else Fault.NONE,
                allreduce_timeout=allreduce_timeout,
                # client_mode=True if i == 1 else False,

            )
            for i, (dht, model) in enumerate(zip(dht_instances, models))
        ]
    else:
        averagers = [
            FaultyDTGradientAverager(
                model.parameters(),
                dht=dht,
                prefix="allreduce_test",
                request_timeout=0.3,
                min_matchmaking_time=1.0,
                next_chunk_timeout=0.5,
                part_size_bytes=2**16,
                start=True,
                fault=fault0 if i == 0 else fault1 if i == 1 else Fault.NONE,
                allreduce_timeout=allreduce_timeout,
                # client_mode=True if i == 2 else False,
            )
            for i, (dht, model) in enumerate(zip(dht_instances, models))
        ]

    criterion = nn.MSELoss()

    for model, averager in zip(models, averagers):
        # Forward pass
        output = model(*make_tensors())

        dummy_target = torch.randn(sum(tensor.numel() for tensor in output))
        flat_output = torch.cat([tensor.flatten() for tensor in output])

        loss = criterion(flat_output, dummy_target)
        loss.backward()
        # Accumulate gras
        averager.accumulate_grads_(batch_size=1)

    ref_numerators = [torch.zeros_like(param.grad) for param in models[0].parameters()]
    ref_denominator = 0

    for averager in averagers:
        if averager.fault not in (Fault.FAIL_BEFORE, Fault.CANCEL):
            for i, param in enumerate(averager.parameters):
                ref_numerators[i] = ref_numerators[i] + param.grad.clone()
            ref_denominator += 1

    ref_tensors = [ref_numerator / ref_denominator for ref_numerator in ref_numerators]
    flat_ref = torch.cat(list(map(torch.flatten, ref_tensors)))
    flat_local_tensors = []

    for averager in averagers:
        averager_tensors = []
        for param in averager.parameters:
            averager_tensors.append(param.grad.clone().flatten())
    flat_local_tensors.append(torch.cat(averager_tensors))

    futures = []

    for averager in averagers:
        if use_original:
            future = averager.step(
                wait=False,
                allow_retries=False,
            )
        else:
            future = averager.step(
                wait=False,
                allow_retries=False,
                custom_group_info=custom_group,
                peerids_to_uids=peerids_to_uids
            )
        futures.append(future)

    for i, averager in enumerate(averagers):
        if averager.fault == Fault.CANCEL:
            futures[i].cancel()
            
    for future in futures[2:]:
        # future.result() will hold allreduce.runnner.banned_senders if any was faulting, otherwise True
        print(future.result())
        assert future.result()

    for averager, prev_local_tensors in zip(averagers[2:], flat_local_tensors[2:]):
        with averager.get_tensors() as tensors:
            flat_tensors = torch.cat(list(map(torch.flatten, tensors)))

        diff_with_reference = abs(flat_ref - flat_tensors)

        if all(
            fault in (Fault.FAIL_SENDING, Fault.SLOW_SENDING)
            for fault in (fault0, fault1)
        ):
            assert fault0 != Fault.FAIL_REDUCING and fault1 != Fault.FAIL_REDUCING
            assert diff_with_reference[: len(diff_with_reference) // 2].max() < 1e-5
        elif all(
            fault in (Fault.FAIL_REDUCING, Fault.SLOW_REDUCING)
            for fault in (fault0, fault1)
        ):
            diff_to_reference = abs(flat_ref - flat_tensors)
            diff_to_local = abs(prev_local_tensors - flat_tensors)
            assert (diff_with_reference < 1e-5).numpy().mean() > 0.5
            assert torch.all(
                torch.minimum(diff_to_reference, diff_to_local) < 1e-5
            ).item()
        elif any(fault == Fault.CANCEL for fault in (fault0, fault1)):
            pass  # late cancel may result in an arbitrary mix of averaging results with and without the cancelled peer
        elif fault0 == Fault.NONE:
            if fault1 == Fault.FAIL_BEFORE:
                # When fault1 is FAIL_BEFORE, we expect some difference due to missing peer1's data
                assert (
                    diff_with_reference < 1e-5
                ).numpy().mean() > 0.70  # At least ~70-80% of values should be close
            else:
                # For other cases where fault0 is NONE, we still expect high accuracy
                assert diff_with_reference.max() < 1e-5
        else:
            assert (diff_with_reference < 1e-5).numpy().mean() > 0.5

    for instance in averagers + dht_instances:
        instance.shutdown()


if __name__ == "__main__":
    fault_pairs = [
        # (Fault.NONE, Fault.NONE),
        (Fault.NONE, Fault.FAIL_BEFORE),
        (Fault.FAIL_BEFORE, Fault.NONE),
        (Fault.FAIL_BEFORE, Fault.FAIL_BEFORE),
        (Fault.SLOW_SENDING, Fault.FAIL_SENDING),
        (Fault.SLOW_SENDING, Fault.NONE),
        (Fault.NONE, Fault.SLOW_SENDING),
        (Fault.FAIL_SENDING, Fault.FAIL_BEFORE),
        (Fault.SLOW_REDUCING, Fault.FAIL_SENDING),
        (Fault.FAIL_REDUCING, Fault.FAIL_REDUCING),
        (Fault.NONE, Fault.CANCEL),
    ]

    # Run loop of fault scenarios
    for fault0, fault1 in fault_pairs:
        # Init models and DHT instances
        n_peers = 5
        dht_instances = launch_dht_instances(n_peers)
        models = [DummyModel() for _ in range(n_peers)]
        # Init custom group
        group_id = DHTID.generate().to_bytes()
        ordered_peer_ids = [dht.peer_id for dht in dht_instances]
        peerids_to_uids = {
            str(peer_id): f"UID_{i}" for i, peer_id in enumerate(ordered_peer_ids)
        }
        custom_group = GroupInfo(group_id, tuple(ordered_peer_ids), gathered=None)

        print(f"Running test with fault0={fault0}, fault1={fault1}")

        run_test(
            fault0,
            fault1,
            dht_instances,
            models,
            peerids_to_uids,
            custom_group,
            use_original=False,
        )
        print("Test completed\n")
