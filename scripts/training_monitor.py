import argparse
from time import sleep

import bittensor as bt
import hivemind

from template.base.neuron import BaseNeuron

parser = argparse.ArgumentParser()
parser.add_argument("--netuid", type=int, help="Network netuid", default=25)

bt.wallet.add_args(parser)
bt.subtensor.add_args(parser)
bt.axon.add_args(parser)

config = BaseNeuron.config()
config.wallet.name = "test_dt"
config.wallet.hotkey = "miner_1"
config.netuid = 80
config.dht.announce_ip = "194.68.245.20"
config.dht.port = 22030
config.axon.port = 22031

dht = hivemind.DHT(
    initial_peers=config.neuron.initial_peers, start=True, client_mode=True
)
progress_tracker = hivemind.optim.progress_tracker.ProgressTracker(
    dht=dht,
    prefix=config.neuron.run_id,
    target_batch_size=config.neuron.global_batch_size_train,
    start=True,
)
while True:
    print(progress_tracker.global_progress)
    sleep(10)
