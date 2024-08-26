import asyncio
import logging
import os
import random
import time
from typing import List, Dict, AsyncIterator
from enum import Enum, auto

import hivemind
from hivemind.utils.asyncio import (aenumerate, enter_asynchronously)
from hivemind.proto import averaging_pb2

import torch
from hivemind.averaging.group_info import GroupInfo
from hivemind.dht import DHT, DHTID
from hivemind.utils import use_hivemind_log_handler
from hivemindy import DTGradientAverager, DTAllReduceRunner
from hivemind.optim.grad_averager import GradientAverager
from hivemind.averaging.allreduce import AllReduceRunner
from torch import nn

from colorama import init, Fore, Style
init(autoreset=True)

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

class FaultyDTGradientAverager(DTGradientAverager):
    def __init__(self, *args, fault: Fault = Fault.NONE, **kwargs):
        self.fault = fault
        super().__init__(*args, **kwargs)

    async def _aggregate_with_group(self, group_info: GroupInfo, min_vector_size: int, peerids_to_uids: dict, **kwargs):
        if self.fault == Fault.FAIL_BEFORE:
            raise Exception(f"Oops, I (UID: {peerids_to_uids[str(self.peer_id)]}) failed before aggregation!")
        
        async with enter_asynchronously(self.get_tensors()) as local_tensors:
            runner = FaultyDTAllReduceRunner(
                peerids_to_uids=peerids_to_uids,
                p2p=self._p2p,
                servicer_type=type(self),
                prefix=self.prefix,
                group_id=group_info.group_id,
                tensors=local_tensors,
                ordered_peer_ids=group_info.peer_ids,
                peer_fractions=[1.0 / len(group_info.peer_ids)] * len(group_info.peer_ids),
                fault=self.fault,
                **kwargs
            )

class FaultyDTAllReduceRunner(DTAllReduceRunner):
    def __init__(self, *args, fault: Fault, **kwargs):
        self.fault = fault
        super().__init__(*args, **kwargs)

    async def rpc_aggregate_part(self, stream, context) -> AsyncIterator[averaging_pb2.AveragingData]:
        if self.fault in (Fault.FAIL_REDUCING, Fault.SLOW_REDUCING):
            async for i, message in aenumerate(super().rpc_aggregate_part(stream, context)):
                yield message
                if i == 2:
                    if self.fault == Fault.FAIL_REDUCING:
                        yield averaging_pb2.AveragingData(code=averaging_pb2.INTERNAL_ERROR)
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
            last_reducer_index = self.group_size - 1 - (self.tensor_part_container.num_parts_by_peer[-1] == 0)
            if peer_index == last_reducer_index:
                if self.fault == Fault.FAIL_SENDING:
                    raise Exception(f"Oops, I (UID: {uid}) failed during sending!")
                else:  # SLOW_SENDING
                    print(f"{Fore.YELLOW}UID: {uid} is slow in sending...{Style.RESET_ALL}")
                    await asyncio.sleep(10)
        async for part in parts_aiter:
            yield averaging_pb2.AveragingData(tensor_part=part, weight=self.weight)

class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.fc = nn.Linear(1024, 1)

    def forward(self, x):
        return self.fc(x)


def launch_dht_instances(n_peers: int, **kwargs) -> List[DHT]:
    dhts = [DHT(start=True, **kwargs)]
    initial_peers = dhts[0].get_visible_maddrs()

    dhts.extend(
        DHT(initial_peers=initial_peers, start=True, await_ready=False, **kwargs)
        for _ in range(n_peers - 1)
    )
    for process in dhts[1:]:
        process.wait_until_ready()

    return dhts
def perform_all_reduce(custom_group: GroupInfo, models, dht_instances: List[DHT], faults: List[Fault], peerids_to_uids: Dict[str, str]):
    averagers = [
        FaultyDTGradientAverager(
            model.parameters(),
            dht=dht,
            prefix="allreduce_test",
            start=True,
            fault=fault
        )
        for (dht, model, fault) in zip(dht_instances, models, faults)
    ]

    # Define a dummy input and target
    dummy_input = torch.randn(
        1, 1024
    )  # Adjust the size according to your model's input size
    dummy_target = torch.randn(
        1, 1
    )  # Adjust the size according to your model's output size

    # Define a loss function
    criterion = nn.MSELoss()

    # Simulate dummy gradients for averaging
    for model in models:
        # num_params = sum(p.numel() for p in model.parameters())
        # print(f"Model 1 has {num_params} parameters.")
        # Forward pass
        output = model(dummy_input)
        # Calculate loss
        loss = criterion(output, dummy_target)
        # Backward pass
        loss.backward()

    for averager in averagers:
        averager.accumulate_grads_(batch_size=1)

    try:
        futures = []
        for averager in averagers:
            sleep_int = random.randint(1, 5)
            uid = peerids_to_uids[str(averager.peer_id)]
            print(f"Peer UID: {uid} with fault {averager.fault} sleeping for {sleep_int} seconds..")
            time.sleep(sleep_int)
            future = averager.step(wait=False, allow_retries=True, custom_group_info=custom_group, peerids_to_uids=peerids_to_uids)
            # future = averager.step(wait=False, allow_retries=True)
            futures.append(future)

        results = []
        for future in futures:
            try:
                result = future.result()
                results.append(result)
                print(f"Averaging result: {result}")
            except Exception as e:
                print(f"{Fore.RED}Averaging failed: {e}{Style.RESET_ALL}")
                results.append(None)

        # Check tensors only for successful averagers
        successful_averagers = [avg for avg, res in zip(averagers, results) if res is not None]
        if successful_averagers:
            with successful_averagers[0].get_tensors() as reference_tensors:
                for averager in successful_averagers[1:]:
                    with averager.get_tensors() as tensors:
                        for ref_tensor, tensor in zip(reference_tensors, tensors):
                            assert torch.allclose(ref_tensor, tensor, atol=1e-5), "Tensors are not equal across averagers"
            print(f"{Fore.GREEN}Tensor check passed for successful averagers.{Style.RESET_ALL}")
        else:
            print(f"{Fore.YELLOW}No successful averagers to check tensors.{Style.RESET_ALL}")

    except Exception as e:
        print(f"{Fore.RED}Exception occurred: {e}{Style.RESET_ALL}")
    finally:
        print("Shutting down...")
        for instance in averagers + dht_instances:
            if hasattr(instance, 'is_alive') and instance.is_alive():
                print(f"Shutting down {instance}")
                instance.shutdown()

def test_fault_scenarios():
    n_peers = 5
    scenarios = [
        [Fault.NONE] * 5,
        [Fault.FAIL_SENDING, Fault.SLOW_SENDING, Fault.NONE, Fault.NONE, Fault.NONE],
        [Fault.FAIL_BEFORE, Fault.NONE, Fault.NONE, Fault.NONE, Fault.NONE],
        [Fault.FAIL_REDUCING, Fault.SLOW_REDUCING, Fault.NONE, Fault.NONE, Fault.NONE],
        [Fault.CANCEL, Fault.NONE, Fault.NONE, Fault.NONE, Fault.NONE],
    ]

    for i, scenario in enumerate(scenarios):
        scenario_description = ", ".join([f"Peer {j}: {fault.name}" for j, fault in enumerate(scenario)])
        print(f"\n{Fore.CYAN}======== Starting Scenario {i+1} ========{Style.RESET_ALL}")
        print(f"{Fore.CYAN}Description: {scenario_description}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}======================================{Style.RESET_ALL}")
        dht_instances = launch_dht_instances(n_peers)
        models = [DummyModel() for _ in range(n_peers)]
        group_id = DHTID.generate().to_bytes()
        ordered_peer_ids = [dht.peer_id for dht in dht_instances]
        # Create peerids_to_uids dictionary
        peerids_to_uids = {str(peer_id): f"UID_{i}" for i, peer_id in enumerate(ordered_peer_ids)}
        print("Peer IDs to UIDs mapping:", peerids_to_uids)
        
        custom_group = GroupInfo(group_id, tuple(ordered_peer_ids), gathered=None)
        
        perform_all_reduce(custom_group, models, dht_instances, scenario, peerids_to_uids)
        
        time.sleep(2)  # Give some time for cleanup between scenarios

def main():
    test_fault_scenarios()
    print("Fault tolerance testing completed.")

if __name__ == "__main__":
    main()