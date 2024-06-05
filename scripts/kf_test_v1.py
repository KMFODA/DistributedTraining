import torch
import time
from mapreduce.peer import Peer
import bittensor as bt
from argparse import ArgumentParser
from datetime import timedelta

parser = ArgumentParser()
parser.add_argument("--validator.uid", type=int, default=0, help="Validator UID")
parser.add_argument("--netuid", type=int, default=10, help="Map Reduce Subnet NetUID")
parser.add_argument("--rank", type=int, default=1, help="Rank of the peer")
parser.add_argument("--count", type=int, default=1, help="Number of peers")

bt.subtensor.add_args(parser)
bt.logging.add_args(parser)
bt.wallet.add_args(parser)
bt.axon.add_args(parser)
config = bt.config(parser=parser)

config.wallet.name = "polkadot_wallet_4"
config.wallet.hotkey = "polkadot_wallet_4_hotkey_2502"

# size for testing, set to 100 MB
test_size = 100 * 1024 * 1024

rank = 1
peer_count = 1
bandwidth = test_size
# wallet = wallet
validator_uid = 0
netuid = 10
network = "finney"

config.validator.uid = 158
import torch.distributed as dist

self = Peer(
    rank,
    peer_count,
    config=config,
    port_range="22125:22131",
    bandwidth=1e9,
    benchmark_max_size=0,
)
self._connect_validator()
dist.init_process_group(
    init_method=f"tcp://{self.master_addr}:{self.master_port}",
    backend="nccl",
    rank=self.rank,
    world_size=self.world_size,
    timeout=timedelta(seconds=60),
)

# python3 test/test.py --netuid 32 --validator.uid 0 --wallet.name default --wallet.hotkey default --subtensor.network test


def train(rank, peer_count, bandwidth, wallet, validator_uid, netuid, network):
    bt.logging.info(f"🔷 Starting peer with rank {rank} netuid: {netuid}")
    # Initialize Peer instance
    peer = Peer(rank, peer_count, config, bandwidth, validator_uid, netuid, network)

    # Initialize process group with the fetched configuration
    peer.init_process_group()

    weights = None

    if rank == 1:  # if it is the first peer
        weights = torch.rand((int(test_size / 4), 1), dtype=torch.float32)
        peer.broadcast(weights)
    else:
        weights = peer.broadcast(weights)

    epoch = 2

    # Your training loop here
    bt.logging.info(f"Peer {rank} is training...")
    for i in range(epoch):
        bt.logging.success(f"🟢 Epoch: {i}")
        # Replace this with actual training code
        time.sleep(5)

        # After calculating gradients
        gradients = torch.ones((int(test_size / 4), 1), dtype=torch.float32)
        if rank == 1:
            gradients = torch.ones((int(test_size / 4), 1), dtype=torch.float32) * 3

        # All-reducing the gradients
        gradients = peer.all_reduce(gradients)

    peer.destroy_process_group()
    print(f"Peer {rank} has finished training.")


if __name__ == "__main__":
    train(
        config.rank,
        config.peer_count,
        test_size,
        wallet,
        config.validator.uid,
        config.netuid,
        config.subtensor.network,
    )
