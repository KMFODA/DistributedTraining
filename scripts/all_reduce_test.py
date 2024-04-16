import asyncio
import logging
import time
from typing import List

import hivemind
import torch
from hivemind.averaging.group_info import GroupInfo
from hivemind.dht import DHT, DHTID
from hivemind.utils import use_hivemind_log_handler
from hivemindy import DTGradientAverager
from torch import nn

logger = logging.getLogger()
logger.setLevel(logging.DEBUG) 

use_hivemind_log_handler("nowhere")

# Create a file handler
handler = logging.FileHandler('logfile.log')

# Create a formatter and add it to the handler
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(handler)

class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.fc = nn.Linear(37, 1)

def launch_dht_instances(n_peers: int, **kwargs) -> List[DHT]:
    dhts = [DHT(start=True, **kwargs)]
    initial_peers = dhts[0].get_visible_maddrs()
    
    dhts.extend(
        DHT(
            initial_peers=initial_peers, 
            start=True, 
            await_ready=False, 
            **kwargs) 
        for _ in range(n_peers - 1)
        )
    
    for process in dhts[1:]:
        process.wait_until_ready()

    return dhts


def perform_all_reduce(custom_group: GroupInfo, models, dht_instances: List[DHT]):

    def _make_tensors():
        return [torch.rand(16, 1024), -torch.rand(3, 8192), 2 * torch.randn(4, 4, 4), torch.randn(1024, 1024)]

    averagers = [
        DTGradientAverager(
            model.parameters(),
            dht=dht,
            prefix="diller",
            auxilirary=True if i == 0 else False,
            start=True,
        )
        for i, (dht, model) in enumerate(zip(dht_instances[:-1], models))
    ]

    try:
        futures = [
            averager.step(
                wait=False, 
                custom_group_info=custom_group, 
                allow_retries=False
            )
            for averager in averagers
        ]
        
        for future in futures:
            print(future.result())

        ''' Check that tensors are averaged and within a certain threshold:'''
        # Get the averaged tensors from the first averager as the reference
        with averagers[0].get_tensors() as reference_tensors:
            for averager in averagers[1:]:
                with averager.get_tensors() as tensors:
                    for ref_tensor, tensor in zip(reference_tensors, tensors):
                        # Check that the tensors are approximately equal
                        assert torch.allclose(ref_tensor, tensor, atol=1e-5), "Tensors are not equal across averagers"

    except Exception as e:
        print("Exception occurred in averager.step():", e)
        print("Shutting down after failure..")
        time.sleep(2)
        for instance in averagers + dht_instances:
            print(instance)
            instance.shutdown()
        exit()

    finally:
        print("Shutting down finally..")
        for instance in averagers + dht_instances:
            if instance.is_alive():
                print(instance)
                instance.shutdown()


def main():
    
    n_peers = 5
    dht_instances = launch_dht_instances(n_peers)
    
    models = [DummyModel() for _ in range(n_peers)]
    # Simulate dummy gradients for averaging
    for model in models:
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Model 1 has {num_params} parameters.")
    
        for param in model.parameters():
            param.grad = torch.randn_like(param)

    # Define a custom group for all-reduce
    group_id = DHTID.generate().to_bytes()
    dht_instances.append(DHT(
                            initial_peers=dht_instances[0].get_visible_maddrs(), 
                            start=True))
    ordered_peer_ids = [dht.peer_id for dht in dht_instances]
    
    custom_group = GroupInfo(group_id, tuple(ordered_peer_ids), gathered=None)

    perform_all_reduce(custom_group, models, dht_instances)

    print("Averaging completed with custom GroupInfo.")

if __name__ == "__main__":
    main()