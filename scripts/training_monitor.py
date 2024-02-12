import hivemind
import argparse
from time import sleep
from DTraining.template.base.neuron import BaseNeuron
import bittensor as bt

parser = argparse.ArgumentParser()
parser.add_argument("--netuid", type=int, help="Network netuid", default=25)

bt.wallet.add_args(parser)
bt.subtensor.add_args(parser)
bt.axon.add_args(parser)

config = BaseNeuron.config()
config.wallet.name = "test_dt"
config.wallet.hotkey = "miner_1"
config.netuid = 25
config.dht.announce_ip = "194.68.245.27"
config.dht.port = 22045
config.axon.port = 22046
  
config.neuron.initial_peers = ["/ip4/213.173.99.12/tcp/11848/p2p/12D3KooWCQzCrKHT5Vak8tyA5SEfypuqnLkGa3H2rH8CyoKcAZkW"]
dht = hivemind.DHT(initial_peers=config.neuron.initial_peers, start=True, client_mode = True)
progress_tracker = hivemind.optim.progress_tracker.ProgressTracker(dht=dht, prefix=config.neuron.run_id, target_batch_size=config.neuron.global_batch_size_train, start = True)
while True:
    print(progress_tracker.global_progress)
    sleep(10)