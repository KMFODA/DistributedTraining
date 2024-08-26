import asyncio
import logging
import os
import random
import time
from typing import List
from enum import Enum, auto

import hivemind
import torch
from hivemind.averaging.group_info import GroupInfo
from hivemind.dht import DHT, DHTID
from hivemind.utils import use_hivemind_log_handler
from hivemindy import DTGradientAverager, DTAllReduceRunner
from torch import nn

# ... (keep your existing logging setup)

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
            raise Exception("Oops, I failed before aggregation!")
        return await super()._aggregate_with_group(group_info, min_vector_size, peerids_to_uids, **kwargs)

class FaultyDTAllReduceRunner(DTAllReduceRunner):
    def __init__(self, *args, fault: Fault, **kwargs):
        self.fault = fault
        super().__init__(*args, **kwargs)

    async def _communicate_with_peer(self, peer_id: hivemind.PeerID):
        if self.fault == Fault.FAIL_SENDING:
            raise Exception("Oops, I failed during sending!")
        elif self.fault == Fault.SLOW_SENDING:
            await asyncio.sleep(10)
        return await super()._communicate_with_peer(peer_id)

    async def _generate_input_for_peer(self, peer_index: int, uid: int, peer_id: hivemind.PeerID):
        if self.fault == Fault.FAIL_REDUCING:
            raise Exception("Oops, I failed during reducing!")
        elif self.fault == Fault.SLOW_REDUCING:
            await asyncio.sleep(10)
        return await super()._generate_input_for_peer(peer_index, uid, peer_id)

# ... (keep your existing DummyModel and launch_dht_instances functions)

def perform_all_reduce(custom_group: GroupInfo, models, dht_instances: List[DHT], faults: List[Fault]):
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

    # ... (keep your existing dummy input, target, and loss calculation)

    try:
        futures = []
        for averager in averagers:
            sleep_int = random.randint(1, 5)
            print(f"Peer with fault {averager.fault} sleeping for {sleep_int} seconds..")
            time.sleep(sleep_int)
            future = averager.step(wait=False, allow_retries=True, custom_group_info=custom_group)
            futures.append(future)

        results = []
        for future in futures:
            try:
                result = future.result()
                results.append(result)
                print(f"Averaging result: {result}")
            except Exception as e:
                print(f"Averaging failed: {e}")
                results.append(None)

        # Check tensors only for successful averagers
        successful_averagers = [avg for avg, res in zip(averagers, results) if res is not None]
        if successful_averagers:
            with successful_averagers[0].get_tensors() as reference_tensors:
                for averager in successful_averagers[1:]:
                    with averager.get_tensors() as tensors:
                        for ref_tensor, tensor in zip(reference_tensors, tensors):
                            assert torch.allclose(ref_tensor, tensor, atol=1e-5), "Tensors are not equal across averagers"
            print("Tensor check passed for successful averagers.")
        else:
            print("No successful averagers to check tensors.")

    except Exception as e:
        print(f"Exception occurred: {e}")
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
        [Fault.FAIL_BEFORE, Fault.NONE, Fault.NONE, Fault.NONE, Fault.NONE],
        [Fault.FAIL_SENDING, Fault.SLOW_SENDING, Fault.NONE, Fault.NONE, Fault.NONE],
        [Fault.FAIL_REDUCING, Fault.SLOW_REDUCING, Fault.NONE, Fault.NONE, Fault.NONE],
        [Fault.CANCEL, Fault.NONE, Fault.NONE, Fault.NONE, Fault.NONE],
    ]

    for scenario in scenarios:
        print(f"\nTesting scenario: {scenario}")
        dht_instances = launch_dht_instances(n_peers)
        models = [DummyModel() for _ in range(n_peers)]
        group_id = DHTID.generate().to_bytes()
        ordered_peer_ids = [dht.peer_id for dht in dht_instances]
        custom_group = GroupInfo(group_id, tuple(ordered_peer_ids), gathered=None)
        
        perform_all_reduce(custom_group, models, dht_instances, scenario)
        
        time.sleep(2)  # Give some time for cleanup between scenarios

def main():
    test_fault_scenarios()
    print("Fault tolerance testing completed.")

if __name__ == "__main__":
    main()