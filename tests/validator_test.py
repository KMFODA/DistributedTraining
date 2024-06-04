import time
from functools import partial

import hivemind
import torch
import torch.nn.functional as F
from datasets import load_dataset
from hivemind.optim.state_averager import TrainingStateAverager
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          default_data_collator)

# Create dataset and model, same as in the basic tutorial
model = nn.Linear(2, 3)

opt = torch.optim.SGD(model.parameters(), lr=1e-5, momentum=0.9)

# Create DHT: a decentralized key-value storage shared between peers
dht = hivemind.DHT(
    initial_peers=[
        "/ip4/54.80.217.105/tcp/8009/p2p/12D3KooWQK6UNzrh59HDJe6t8unEBePU7rC1ULHHvKWULBnxknC9"
    ],
    start=True,
)

# Init State Averager
state_averager = TrainingStateAverager(
    dht=dht,
    optimizer=opt,
    scheduler=partial(
        torch.optim.lr_scheduler.LambdaLR, lr_lambda=lambda t: 1.0 / max(1, t)
    ),
    params=model.parameters(),
    allow_state_sharing=True,
    start=True,
    prefix="my_cifar_run_state_averager",
    # state_compression=hivemind.Float16Compression(),
    # bandwidth=optimizer_args.bandwidth,
    # client_mode=optimizer_args.client_mode,
    # **asdict(averager_args),
)


try:
    while True:
        state_averager.load_state_from_peers()
        print(state_averager.local_epoch)
        print(model.weight.data[...])
        time.sleep(10)
except KeyboardInterrupt:
    dht.shutdown()
    state_averager.shutdown()
    exit()
