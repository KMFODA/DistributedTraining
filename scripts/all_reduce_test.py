import asyncio
import logging
import time
from typing import List

import torch
from hivemind.averaging.group_info import GroupInfo
from hivemind.dht import DHT, DHTID
from hivemind.optim.grad_averager import GradientAverager
from hivemind.utils import get_logger, use_hivemind_log_handler
from hivemindy import DTGradientAverager
from torch import nn

logger = logging.getLogger()
logger.setLevel(logging.DEBUG) # Set this to logging.DEBUG to check hivemind debug messages -> Careful, it's a lot of output

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

async def handle_averager_step(averager, custom_group: GroupInfo):
    try:
        result = await averager.step(
            wait=False,
            custom_group_info=custom_group,
        )
        print(averager.peer_id, result)
    except Exception as e:
        print("Exception occurred in averager.step():", e)
        print("Shutting down after failure..")
        averager.shutdown()
        raise


async def perform_all_reduce(custom_group: GroupInfo, models: List[nn.Module], dht_instances: List[DHT]):
    
    def _make_tensors():
        return [torch.rand(16, 1024), -torch.rand(3, 8192), 2 * torch.randn(4, 4, 4), torch.randn(1024, 1024)]

    try:
        averagers = [
            DTGradientAverager(
                _make_tensors(), # Does it only work with _make_tensors() currently, because the model hasnt accumulated any gradients?
                #model.parameters(),
                dht=dht,
                prefix="diller",
                client_mode=True if i == 0 else False,
                start=True,
            )
            for i, dht in enumerate(dht_instances)
        ]

        # futures = [
        #     averager.step(
        #         wait=False,
        #         custom_group_info=custom_group,
        #         allow_retries=False
        #     )
        #     for averager in averagers
        # ]
        
        
        futures = [handle_averager_step(averager, custom_group) for averager in averagers]
        
        try:
            await asyncio.gather(*futures)
        except Exception as e:
            print("An exception occurred:", e)
        finally:
            print("Shutting down after success or failure..")
            for instance in averagers + dht_instances:
                print(instance)
                instance.shutdown()
        
        # futures[-1].cancel()
        # for future in futures:
        #     try:
        #         result = future.result()
        #     except Exception as e:
        #         print(e)
        #         print("Shutting down after failure..")
        #         for instance in averagers + dht_instances:
        #             print(instance)
        #             instance.shutdown()
        #             exit()
            
        #     for averager in averagers:
        #         try:
        #             print(averager.peer_id, result)
        #         except Exception as e:
        #             print("HEREYO:", e)
        #         # assert averager.peer_id in result

        # print("Shutting down after succes..")
        # for instance in averagers + dht_instances:
        #     print(instance)
        #     instance.shutdown()

    except Exception as e:
        print(e)
        print("Shutting down after failure..")
        for instance in averagers + dht_instances:
            print(instance)
            instance.shutdown()
            exit()

async def main():
    
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
    ordered_peer_ids = [dht.peer_id for dht in dht_instances]
    custom_group = GroupInfo(group_id, tuple(ordered_peer_ids), gathered=None)

    await perform_all_reduce(custom_group, models, dht_instances)

    print("Averaging completed with custom GroupInfo.")

if __name__ == "__main__":
    asyncio.run(main())
    # TODO
    # 1. Add various faulty scenarios - test if it correctly faults                                             (almost done)
    # 2. Fix so we only hit one Expecption instead of multiple                                                  (not done)
    # 3. Add peer_fraction param to avoid averaging the "validator" peer                                        (almost done)
    # - - We just need to set validator GradAverager to client_mode=True, then it's peer_fraction will be 0     
    # - - How do we ensure equal fractions for the rest of the peers?
    # 4. Should we use the bandwith scores to do load_balancing during AllReduce? 
    # - - Or should we just equally distribute the gradients?
    # - - Butterfly AllReduce uses load balancing, but does load balancing make our bandwidth incentive obsolete?
    
    
    # TODO today:
    # 1. Test if we can correctly fault peers
    # 2. Test if using a different GradientAverager get's caught
    # 3. Test on colab? - i.e. between instances
    # 4. Use bandwidth as load_balancing 
