import os
import time
from functools import partial
from ipaddress import ip_address

import hivemind
import requests
import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator
import bittensor as bt
import argparse
import wandb
from DTraining.template.base.neuron import BaseNeuron
from hivemind import utils
import re
from template.data.dataset import SubsetFalconLoader
import random

parser = argparse.ArgumentParser()
parser.add_argument("--netuid", type=int, help="Network netuid", default=25)

bt.wallet.add_args(parser)
bt.subtensor.add_args(parser)
bt.axon.add_args(parser)

config = BaseNeuron.config()
config.wallet.name = "polkadot_wallet_1"
config.wallet.hotkey = "polkadot_wallet_1_hotkey_25_validator_1"
config.netuid = 25
config.dht.announce_ip = "194.68.245.27"
config.dht.port = 22126
config.axon.port = 22127
# config.neuron.run_id = "s25_test_1"

wallet = bt.wallet(config=config)
subtensor = bt.subtensor(config=config)
metagraph = subtensor.metagraph(config.netuid)

# Init device
device = config.neuron.device

# Init DHT and model
if config.dht.use_google_dns:
    request = requests.get("https://api.ipify.org")
    request.raise_for_status()

    address = request.text
    bt.logging.info(f"Received public IP address of this machine: {address}")
    version = ip_address(address).version
    announce_maddrs = [f"/ip{version}/{address}/tcp/{config.dht.port}"]
else:
    version = "4"
    address = config.dht.announce_ip
    announce_maddrs = [f"/ip{version}/{address}/tcp/{config.dht.port}"]

# Init list of available DHT addresses from wandb
api = wandb.Api()
initial_peers_list = config.neuron.initial_peers
runs = api.runs(
    f"{config.neuron.wandb_entity}/{config.neuron.wandb_project}"
)
for ru in runs:
    if ru.state == "running":
        for peer in ru.config["neuron"]["initial_peers"]:
            if peer not in initial_peers_list:
                initial_peers_list.append(peer)

# Init DHT
retries = 0
while retries <= len(initial_peers_list):
    if retries == len(initial_peers_list):
        raise Exception("Max retries reached, operation failed.")
    try:
        # Init DHT
        dht = hivemind.DHT(
            host_maddrs=[
                f"/ip4/0.0.0.0/tcp/{config.dht.port}",
                f"/ip4/0.0.0.0/udp/{config.dht.port}/quic",
            ],
            initial_peers=[initial_peers_list[retries]],
            announce_maddrs=announce_maddrs,
            start=True,
        )
        bt.logging.info(
            f"Successfully initialised dht using initial_peer as {initial_peers_list[retries]}"
        )
        break
    except Exception as e:
        bt.logging.error(
            f"Attempt {retries + 1} to init DHT using initial_peer as {initial_peers_list[retries]} failed with error: {e}"
        )
        retries += 1
        bt.logging.error(f"Retrying...")
utils.log_visible_maddrs(dht.get_visible_maddrs(), only_p2p=True)
model = AutoModelForCausalLM.from_pretrained(config.neuron.model_name)

# Add DHT address to wandb config
config.neuron.initial_peers = config.neuron.initial_peers + [
    re.sub("ip4/?(.*?)/", f"ip{version}/{address}/", str(addr), flags=re.DOTALL)
    for addr in dht.get_visible_maddrs()
]

# Move the model to the appropriate device
model = model.to(device)

# Set up a decentralized optimizer that will average with peers in background
opt = torch.optim.AdamW(model.parameters(), lr=config.neuron.lr)
opt_2 = hivemind.Optimizer(
    dht=dht,  # use a DHT that is connected with other peers
    run_id=config.neuron.run_id,  # unique identifier of this collaborative run
    scheduler=None,
    batch_size_per_step=config.neuron.local_batch_size_train*config.neuron.local_gradient_accumilation_steps_train,  # each call to opt.step adds this many samples towards the next epoch
    target_batch_size=config.neuron.global_batch_size_train,  # after peers collectively process this many samples, average weights and begin the next epoch
    optimizer=opt,  # wrap the SGD optimizer defined above
    use_local_updates=False,  # perform optimizer steps with local gradients, average parameters in background
    matchmaking_time=15.0,  # when averaging parameters, gather peers in background for up to this many seconds
    averaging_timeout=60.0,  # give up on averaging if not successful in this many seconds
    verbose=True,  # print logs incessently
    grad_compression=hivemind.Uniform8BitQuantization(),
    state_averaging_compression=hivemind.Uniform8BitQuantization(),
)
from hivemind.optim.grad_averager import GradientAverager
# grad_averager = GradientAverager(model.parameters(), dht=dht, prefix=f"{config.neuron.run_id}_grad_averager",start=True)

tokenizer = AutoTokenizer.from_pretrained(config.neuron.model_name)
# Add the EOS token as PAD token to ensure our dataloader doesn't throw an error for sequences of unequal length
tokenizer.pad_token = tokenizer.eos_token

# Set config.neuron.local_batch_size_train
config.neuron.local_batch_size_train_total = 20

dataloader = SubsetFalconLoader(
    batch_size=config.neuron.local_batch_size_train, sequence_length=1024, rows=random.choices(range(0,968000015), k = config.neuron.local_batch_size_train_total)
)

for i, batch in enumerate(dataloader):
    inputs = batch.to(device)

    # Forward pass
    outputs = model(input_ids=inputs, labels=inputs)
    
    # loss = outputs.loss / config.neuron.local_batch_size_train_total  # Scale loss
    loss = outputs.loss
    loss.backward()
    opt.step()

grad_averager.accumulate_grads_(batch_size=i)
grad_averager.step(control=grad_averager.schedule_step(scheduled_time=hivemind.get_dht_time()))