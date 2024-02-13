import torch.multiprocessing as mp
import torch
import time
from mapreduce import Peer
import bittensor as bt
from argparse import ArgumentParser
from DTraining.template.base.neuron import BaseNeuron
from transformers import AutoModelForCausalLM, AutoTokenizer
from template.data.dataset import SubsetFalconLoader
import random

# tensor size for testing, set to 10 MB
tensor_size = 68 * 1024 * 1024
bandwidth = tensor_size * 4 # torch.float32 is 4 bytes

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
    batch_size=config.neuron.local_batch_size_train, sequence_length=1024, rows=random.choices(range(0,968000015), k = config.neuron.training_examples_per_miner)
)


# parser_2.add_argument("--netuid", type=int, help="Network netuid", default=32)
# parser_2.add_argument("--validator.uid", type=int, help="Network netuid", default=0)

# python3 test/test.py --netuid 32 --validator.uid 0 --wallet.name default --wallet.hotkey default --subtensor.network test

parser_2 = ArgumentParser()
parser_2.add_argument ('--port.range', default = '22043:22049', help = "Opened Port range" )
parser_2.add_argument('--validator.uid', type = int, default= 0, help='Validator UID')
parser_2.add_argument('--netuid', type = int, default= 32, help='Subnet UID')
parser_2.add_argument("--subtensor.network", type=str, help="Network netuid", default="test")

bt.subtensor.add_args(parser_2)
bt.logging.add_args(parser_2)
bt.wallet.add_args(parser_2)
bt.axon.add_args(parser_2)
config_2 = bt.config(parser_2)


def train(rank, peer_count, bandwidth):
    bt.logging.info(f"🔷 Starting peer with rank {rank}")
    # Initialize Peer instance
    peer = Peer(rank, peer_count, config=config_2, bandwidth=bandwidth)

    # Initialize process group with the fetched configuration
    peer.init_process_group()

    weights = [layer for layer in model.parameters()]

    if rank == 1: # if it is the first peer
        weights = torch.rand((tensor_size, 1), dtype=torch.float32)
        # First peer broadcasts the weights
        peer.broadcast(weights)
    else:
        # Other peers receive the weights
        weights = peer.broadcast(weights)
    
    # Should destroy process group after broadcasting
    peer.destroy_process_group()

    # Number of epochs
    epoch = 2

    # Your training loop here
    bt.logging.info(f"Peer {rank} is training...")   
    
    for i in range(epoch):

        bt.logging.success(f"🟢 Epoch: {i}")
        
        # Replace this with actual training code
        for i, batch in enumerate(dataloader):
            inputs = batch.to(device)

            # Forward pass
            outputs = model(input_ids=inputs, labels=inputs)
            
            # loss = outputs.loss / config.neuron.local_batch_size_train_total  # Scale loss
            loss = outputs.loss
            loss.backward()
        
        # After calculating gradients
        # gradients = torch.ones((tensor_size, 1), dtype=torch.float32)
        gradients = [layer.grad for layer in model.parameters()]
        
        if rank == 1:
            # gradients = torch.ones((tensor_size, 1), dtype=torch.float32) * 3
            gradients = gradients * 3

        # Initialize process group
        peer.init_process_group()
        
        # All-reducing the gradients (average of gradients)
        gradients = peer.all_reduce(gradients)
        
        # Destroy process group
        peer.destroy_process_group()
    
    bt.logging.success(f"Peer {rank} has finished training.")

def main():
    peer_count = 3
    processes = []

    # Start two peer processes
    for rank in range(1, peer_count + 1):
        p = mp.Process(target=train, args=(rank, peer_count, tensor_size))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

if __name__ == '__main__':
    # mp.set_start_method('spawn')  # This is often necessary in PyTorch multiprocessing
    main()
