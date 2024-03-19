import asyncio
import logging
import time
from typing import List

import torch
from hivemind.averaging.group_info import GroupInfo
from hivemind.dht import DHT, DHTID
from hivemind.utils import get_logger
from template.utils.hivemind import DTGradientAverager
from torch import nn

logger = get_logger(__name__)
logger.setLevel(logging.DEBUG)

class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.fc = nn.Linear(19, 1)

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


async def perform_all_reduce(custom_group: GroupInfo, models: List[nn.Module], dht_instances: List[DHT]):
    try:
        averagers = [DTGradientAverager(
                                        model.parameters(),
                                        dht=dht, 
                                        prefix="diller",
                                        start=True)
                                    for model, dht, in zip(models, dht_instances)]
        
        
        futures = [averager.step(
                        wait=False, 
                        custom_group_info=custom_group,
                        ) 
                for averager in averagers]
         
        for future in futures:
            result = future.result()
            print(result)
            for averager in averagers:
                assert averager.peer_id in result
        
        print("Shutting down..")
        for instance in averagers + dht_instances:
            print(instance)
            instance.shutdown()
    
    except KeyboardInterrupt:
        for idx in range(len(averagers)):
            averagers[idx].shutdown()
            dht_instances[idx].shutdown()
            exit()
            
async def main():
    
    n_peers = 4
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
    ordered_peer_ids = [dht.peer_id for dht in dht_instances]
    custom_group = GroupInfo(group_id, tuple(ordered_peer_ids), gathered=None)

    await perform_all_reduce(custom_group, models, dht_instances)

    print("Averaging completed with custom GroupInfo.")

if __name__ == "__main__":
    asyncio.run(main())
    