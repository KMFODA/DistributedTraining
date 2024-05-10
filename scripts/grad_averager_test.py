import hivemind
from hivemind.utils.logging import use_hivemind_log_handler
from template.utils.misc import init_dht, setup_logging
from template.utils.hivemind import DTGradientAverager
from hivemind.dht import DHTID
from hivemind.averaging.group_info import GroupInfo

import torch
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
from template.data.dataset import SubsetFalconLoader
import logging
from contextlib import contextmanager
from template.base.neuron import BaseNeuron
import os
import asyncio
import time

logger = logging.getLogger()
logger.setLevel(logging.DEBUG) 

use_hivemind_log_handler("nowhere")

# Create a file handler
handler = logging.FileHandler('logfile.log')

# Create a formatter and add it to the handler
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(handler)

class MyGradientAverager(hivemind.optim.grad_averager.GradientAverager):
    '''
    Needs this wrapper class to ensure device is set properly when averaging gradients
    see: https://github.com/learning-at-home/hivemind/blob/d20e81017481aa2028efc33217522248aabd7d95/hivemind/optim/grad_averager.py#L224
    '''
    @contextmanager
    @torch.no_grad()
    def use_averaged_gradients(self):
        """Substitute model's main gradients with averaged gradients"""
        self._new_averaged_grads = False
        with self.get_tensors() as averaged_grads:
            assert len(averaged_grads) == len(self.parameters)
            try:
                old_grads = [param.grad for param in self.parameters]
                for param, new_grad in zip(self.parameters, averaged_grads):
                    # move new_grad to the same device as param before assigning
                    param.grad = new_grad.to(param.device)
                yield averaged_grads
            finally:
                for param, old_grad in zip(self.parameters, old_grads):
                    param.grad = old_grad

version = "4"
address = "91.150.160.38"
announce_maddrs = [f"/ip{version}/{address}/tcp/4336"]

dht = hivemind.DHT(
    host_maddrs=[
                f"/ip4/0.0.0.0/tcp/4336",
                f"/ip4/0.0.0.0/udp/4336/quic",
                ],
    #initial_peers=["/ip4/161.97.156.125/tcp/8000/p2p/12D3KooWSaqmfoX6NVLrnoKWhNwwFoyMtKGyAmoqASPKEzjVC6GN"], 
    
    announce_maddrs=announce_maddrs,
    start=True
)
print(dht.get_visible_maddrs())

# Write the visible_maddrs to a text file
with open('visible_maddrs.txt', 'w') as f:
    for maddr in dht.get_visible_maddrs():
        f.write(str(maddr) + "\n")

time.sleep(20)

model = AutoModelForCausalLM.from_pretrained("kmfoda/gpt2-250m")
# Move the model to the appropriate device
model = model.to("cuda")

# Set up a decentralized optimizer that will average with peers in background
opt = torch.optim.AdamW(model.parameters(), lr=0.001)

global_target_batch_size = 400  # set your target batch size
grad_averager = DTGradientAverager(
    model.parameters(), 
    dht=dht, 
    prefix=f"peniz",
    start=True,
    #accumulate_grads_on=torch.device(self.device),
    compression=hivemind.Uniform8BitQuantization(),
    # next_chunk_timeout = 30.0,
)

tracker = hivemind.optim.progress_tracker.ProgressTracker(
    dht=dht, 
    prefix="peniz", 
    target_batch_size=global_target_batch_size,
    start=True
)

#total_batch_size = 0
step_scheduled = False
local_epoch, local_samples = 0, 0

#* Make custom group:
time.sleep(5)
loop = asyncio.new_event_loop()
# _p2p = loop.run_until_complete(dht.replicate_p2p())

while True:
    print("Starting training..")
    # for i in range(0, 1):
    print("Getting new data..")
    dataloader = SubsetFalconLoader(
        batch_size=1, sequence_length=1024, rows=random.choices(range(0,968000015), k = 200)
    )

    for i, batch in enumerate(dataloader):
        
        inputs = batch.to("cuda")

        # Forward pass
        outputs = model(input_ids=inputs, labels=inputs)
        
        loss = outputs.loss
        scaled_loss = loss / global_target_batch_size # Minus batch size (in this case 1)
        print(loss)
        scaled_loss.backward()
        
        # Only use this if reuse_grad_buffers=False
        grad_averager.accumulate_grads_(batch_size=1)
        
        local_samples += 1  # increment the total batch size
        
        tracker.report_local_progress(local_epoch, local_samples)
        print("local samples:", tracker.local_progress.samples_accumulated, "global_samples:", tracker.global_progress.samples_accumulated)
        print("local epoch:", tracker.local_progress.epoch, "global epoch", tracker.global_progress.epoch)

        # aggregate gradients and perform optimizer step when target batch size is reached
        if tracker.global_progress.samples_accumulated >= global_target_batch_size:
            _p2p = loop.run_until_complete(dht.replicate_p2p())

            group_id = b'"}\xf3\xca\x86\xfe\xbb&\xdd\xb3\xe2\xffCtZ~\x8e\x10\xf9\xb5'
            ordered_peer_ids = [dht.peer_id] # TODO REMEMBER SAME ORDER FOR OTHER PEERS
            remote_peer = loop.run_until_complete(_p2p.list_peers())
            remote_peer = [peer.peer_id for peer in remote_peer]
            ordered_peer_ids += remote_peer
            ordered_peer_ids.sort(key=lambda peer: peer.xor_id)
            custom_group = GroupInfo(group_id, tuple(ordered_peer_ids), gathered=None)
            print(custom_group)
            with tracker.pause_updates():
                print("grad stepping..")
                grad_averager.step(custom_group_info=custom_group)
                with grad_averager.use_averaged_gradients():  # this will fill param.grads with aggregated gradients
                    print("opt stepping..")
                    opt.step()  # update model parameters using averaged gradients
                grad_averager.reset_accumulated_grads_()  # prepare for next step
                local_epoch = tracker.update_epoch(local_epoch + 1)
                local_samples = 0  