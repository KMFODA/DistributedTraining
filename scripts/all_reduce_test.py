import hivemind
import random
import time

import numpy as np
import pytest
import torch

import hivemind
from hivemind.averaging import DecentralizedAverager
from hivemind.averaging.allreduce import AveragingMode
from hivemind.averaging.control import AveragingStage
from hivemind.averaging.key_manager import GroupKeyManager
from hivemind.averaging.load_balancing import load_balance_peers
from hivemind.averaging.partition import AllreduceException
from hivemind.p2p import PeerID
from hivemind.dht import DHT
from typing import Dict, List, Tuple


def launch_dht_instances(n_peers: int, **kwargs) -> List[DHT]:
    dhts = [DHT(start=True, **kwargs)]
    initial_peers = dhts[0].get_visible_maddrs()

    dhts.extend(DHT(initial_peers=initial_peers, start=True, await_ready=False, **kwargs) for _ in range(n_peers - 1))
    for process in dhts[1:]:
        process.wait_until_ready()

    return dhts

# Training Args
import torch.multiprocessing as mp
import torch
import time
from mapreduce import Peer
import bittensor as bt
from argparse import ArgumentParser
from template.base.neuron import BaseNeuron
from transformers import AutoModelForCausalLM, AutoTokenizer
from template.data.dataset import SubsetFalconLoader
import random

parser = ArgumentParser()
parser.add_argument("--netuid", type=int, help="Network netuid", default=25)

bt.wallet.add_args(parser)
bt.subtensor.add_args(parser)
bt.axon.add_args(parser)

config = BaseNeuron.config()
config.wallet.name = "test_dt"
config.wallet.hotkey = "validator_1"
config.netuid = 25
config.axon.port = 22043

wallet = bt.wallet(config=config)
subtensor = bt.subtensor(config=config)
metagraph = subtensor.metagraph(config.netuid)

# Init device
device = config.neuron.device

# Init Model
model = AutoModelForCausalLM.from_pretrained(config.neuron.model_name)

# Move the model to the appropriate device
model = model.to(device)

# Set up a decentralized optimizer that will average with peers in background
opt = torch.optim.AdamW(model.parameters(), lr=config.neuron.lr)

tokenizer = AutoTokenizer.from_pretrained(config.neuron.model_name)
# Add the EOS token as PAD token to ensure our dataloader doesn't throw an error for sequences of unequal length
tokenizer.pad_token = tokenizer.eos_token

dataloader = SubsetFalconLoader(
    batch_size=config.neuron.local_batch_size_train, sequence_length=1024, rows=random.choices(range(0,968000015), k = 1)
)

for i, batch in enumerate(dataloader):
    inputs = batch.to(device)
    # inputs = batch

    # Forward pass
    outputs = model(input_ids=inputs, labels=inputs)
    
    # loss = outputs.loss / config.neuron.local_batch_size_train_total  # Scale loss
    loss = outputs.loss
    loss.backward()

from typing import Callable, Iterable, Iterator, Optional, Sequence, TypeVar
def _grads_from_parameters() -> Iterator[torch.Tensor]:
    """gradient buffers associated with parameters"""
    for param in model.parameters():
        if param.grad is None:
            param.grad = torch.zeros_like(param)
        yield param.grad

averaged_grads = tuple(
    grad.detach().cpu().clone().share_memory_() for grad in _grads_from_parameters()
)

size = [50257, 1024, 768]
tensors1 = [torch.randn(size[0]), torch.zeros(size[1]), torch.zeros(size[2])]
tensors2 = [torch.rand(size[0]), torch.ones(size[1]), torch.ones(size[2])]
tensors3 = [-torch.rand(size[0]), torch.arange(size[1]).to(torch.float32), torch.arange(size[2]).to(torch.float32)]
tensors4 = [torch.randn(size[0]) ** 3, torch.arange(size[1]).to(torch.float32) / 2, torch.arange(size[2]).to(torch.float32) / 2]
peer_tensors = [tensors1, tensors2, tensors3, tensors4]
list[tensors[list]]

# tensors1 = [torch.randn(123), torch.zeros(3)]
# tensors2 = [torch.rand(123), torch.ones(3)]
# tensors3 = [-torch.rand(123), torch.arange(3).to(torch.float32)]
# tensors4 = [torch.randn(123) ** 3, torch.arange(3).to(torch.float32) / 2]
# peer_tensors = [tensors1, tensors2, tensors3, tensors4]

length = 1
tensors1_2 = [layer.grad[0].to("cpu")[:300] for layer in model.parameters()][:1]
tensors2_2 = [torch.clone(layer.grad[0]).detach() for layer in model.parameters()][:length]
tensors3_2 = [torch.clone(layer.grad[0]).detach() for layer in model.parameters()][:length]
tensors4_2 = [torch.clone(layer.grad[0]).detach() for layer in model.parameters()][:length]

for layer in model.parameters(): break
tensors1_2 = [layer.grad.to("cpu")]
tensors2_2 = [layer.grad.to("cpu")]
tensors3_2 = [layer.grad.to("cpu")]
tensors4_2 = [layer.grad.to("cpu")]

tensors1_2 = averaged_grads
tensors2_2 = averaged_grads
tensors3_2 = averaged_grads
tensors4_2 = averaged_grads

peer_tensors_2 = [tensors1_2, tensors2_2, tensors3_2, tensors4_2]
peer_tensors = peer_tensors_2

n_peers = 4
n_clients = 0
n_aux = 0
modes = (
    [AveragingMode.CLIENT] * n_clients
    + [AveragingMode.AUX] * n_aux
    + [AveragingMode.NODE] * (n_peers - n_clients - n_aux)
)
random.shuffle(modes)

reference = [
    sum(tensors[i] for tensors, mode in zip(peer_tensors, modes) if mode != AveragingMode.AUX)
    / max(1, n_peers - n_aux)
    for i in range(len(tensors1))
]

dht_instances = launch_dht_instances(len(peer_tensors))

averagers = [
    DecentralizedAverager(
        tensors,
        dht=dht,
        # target_group_size=4,
        min_matchmaking_time=15,
        prefix="mygroup",
        compression = hivemind.Uniform8BitQuantization(),
        client_mode=mode == AveragingMode.CLIENT,
        auxiliary=mode == AveragingMode.AUX,
        start=True,
    )
    for tensors, dht, mode in zip(peer_tensors, dht_instances, modes)
]

futures = []
for averager in averagers:
    futures.append(averager.step(wait=False, timeout=60))
for future in futures:
    result = future.result()
    for averager in averagers:
        assert averager.peer_id in result

for averager in averagers:
    if averager.mode != AveragingMode.AUX:
        with averager.get_tensors() as averaged_tensors:
            print(averaged_tensors)
            # for ref, our in zip(reference, averaged_tensors):
            #     assert torch.allclose(ref, our, atol=1e-6)

for process in averagers + dht_instances:
    process.shutdown()