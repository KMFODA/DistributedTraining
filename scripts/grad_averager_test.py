import argparse
import asyncio
import base64
import logging
import os
import random
import time
from contextlib import contextmanager

import hivemind
import schedulefree
import torch

from hivemind.averaging.group_info import GroupInfo
from hivemind.dht import DHTID
from hivemind.utils.logging import use_hivemind_log_handler
from transformers import AutoModelForCausalLM

from template.data.dataset import SubsetFalconLoader
from template.utils.hivemind import DTGradientAverager

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Run a distributed training script with Hivemind."
)
parser.add_argument(
    "--prefix", type=str, required=True, help="Prefix for DHT and gradient averager."
)
parser.add_argument(
    "--initial_peers", type=str, nargs="*", help="Initial peers for DHT (optional)."
)
args = parser.parse_args()

# Logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

use_hivemind_log_handler("nowhere")

# Delete the logfile if it exists
logfile = "logfile.log"
if os.path.exists(logfile):
    os.remove(logfile)

# Create a file handler
handler = logging.FileHandler(logfile)

# Create a formatter and add it to the handler
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(handler)

# DHT
version = "4"
address = "213.108.196.111"

announce_maddrs = [f"/ip{version}/{address}/tcp/7352"]

# Prepare DHT parameters
dht_params = {
    "host_maddrs": [
        f"/ip4/0.0.0.0/tcp/7352",
        f"/ip4/0.0.0.0/udp/7352/quic",
    ],
    "announce_maddrs": announce_maddrs,
    "start": True,
}

# Conditionally add initial_peers if provided
if args.initial_peers:
    dht_params["initial_peers"] = [str(peer) for peer in args.initial_peers]

# Initialize the DHT
dht = hivemind.DHT(**dht_params)

print(dht.get_visible_maddrs())

if not args.initial_peers:
    # Write the visible_maddrs to a text file
    with open("visible_maddrs.txt", "w") as f:
        for maddr in dht.get_visible_maddrs():
            f.write(str(maddr) + "\n")

    time.sleep(16)

model = AutoModelForCausalLM.from_pretrained("kmfoda/gpt2-250m")
model.to("cuda")

# Set up a decentralized optimizer that will average with peers in background
opt = torch.optim.AdamW(model.parameters(), lr=0.001)
# opt = schedulefree.AdamWScheduleFree(model.parameters(), lr=0.001)

global_target_batch_size = 50  # set your target batch size
grad_averager = DTGradientAverager(
    model.parameters(),
    dht=dht,
    prefix=args.prefix,
    start=True,
    compression=hivemind.Uniform8BitQuantization(),
)

tracker = hivemind.optim.progress_tracker.ProgressTracker(
    dht=dht,
    prefix=f"{args.prefix}_tracker",
    target_batch_size=global_target_batch_size,
    start=True,
)

# total_batch_size = 0
step_scheduled = False
local_epoch, local_samples = 0, 0

# * Make custom group:
# time.sleep(5)
loop = asyncio.new_event_loop()
group_is_set = False
# _p2p = loop.run_until_complete(dht.replicate_p2p())
BATCH_SIZE = 1

while True:
    print("Starting training..")
    # for i in range(0, 1):
    print("Getting new data..")
    dataloader = SubsetFalconLoader(
        batch_size=BATCH_SIZE,
        sequence_length=1024,
        rows=random.choices(range(0, 519_000_000), k=1000),
    )

    for i, batch in enumerate(dataloader):
        inputs = batch.to("cuda")

        # Forward pass
        outputs = model(input_ids=inputs, labels=inputs)

        loss = outputs.loss
        scaled_loss = (
            loss / global_target_batch_size / BATCH_SIZE
        )  # Minus batch size (in this case 1)
        print(loss)
        scaled_loss.backward()

        # Only use this if reuse_grad_buffers=False
        grad_averager.accumulate_grads_(batch_size=BATCH_SIZE)

        local_samples += BATCH_SIZE  # increment the total batch size

        tracker.report_local_progress(local_epoch, local_samples)
        print(
            "local samples:",
            tracker.local_progress.samples_accumulated,
            "global_samples:",
            tracker.global_progress.samples_accumulated,
        )
        print(
            "local epoch:",
            tracker.local_progress.epoch,
            "global epoch",
            tracker.global_progress.epoch,
        )

        # aggregate gradients and perform optimizer step when target batch size is reached
        if tracker.global_progress.samples_accumulated >= global_target_batch_size:
            if not group_is_set:
                _p2p = loop.run_until_complete(dht.replicate_p2p())

                group_id = base64.b64decode(b"akGgUCKXywtpOCU76x9Ncxzi2qk=")
                ordered_peer_ids = [dht.peer_id]
                remote_peer = loop.run_until_complete(_p2p.list_peers())
                remote_peer = [peer.peer_id for peer in remote_peer]
                ordered_peer_ids += remote_peer
                ordered_peer_ids.sort(key=lambda peer: peer.xor_id)
                custom_group = GroupInfo(group_id, tuple(ordered_peer_ids), gathered=None)
                print(custom_group)
                group_is_set = True

            with tracker.pause_updates():
                print("grad stepping..")
                # grad_averager.step(custom_group_info=custom_group)
                grad_step = grad_averager.step(
                    custom_group_info=custom_group)
                # if gradient_averaging_step.done():
                # while not grad_step.done():
                # print("Sleeping for 10")
                # time.sleep(10)
                with grad_averager.use_averaged_gradients():  # this will fill param.grads with aggregated gradients
                    print("opt stepping..")
                    opt.step()  # update model parameters using averaged gradients
                    grad_averager.reset_accumulated_grads_()  # prepare for next step
                    local_epoch = tracker.update_epoch(local_epoch + 1)
                    local_samples = 0
