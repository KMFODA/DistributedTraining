from template.base.neuron import BaseNeuron
import os
from template.utils.misc import init_dht, setup_logging
from hivemind.optim.progress_tracker import ProgressTracker
from hivemind.optim.state_averager import TrainingStateAverager
from transformers import AutoModelForCausalLM
import torch
from DTraining.template.utils.gradient_averager import (
    DTGradientAverager,
    DTStateAverager,
)
import hivemind
from bitarray import bitarray
import random
import bittensor as bt
from template.data.dataset import SubsetFalconLoader
import time

setup_logging()


class Dummy:
    def __init__(*args, **kwargs):
        pass


self = Dummy()
self.subtensor = "dummy"
self.config = BaseNeuron.config()
self.config.wallet.name = "test_dt"
self.config.wallet.hotkey = "validator_1"
self.config.netuid = 80
self.config.subtensor.network = "test"
self.config.axon.port = os.environ["RUNPOD_TCP_PORT_70000"]
self.config.dht.port = os.environ["RUNPOD_TCP_PORT_70001"]
self.config.dht.announce_ip = os.environ["RUNPOD_PUBLIC_IP"]
self.config.model_name = "kmfoda/gpt-250m"

init_dht(self)

self.device = self.config.neuron.device

# Init Model
self.model = AutoModelForCausalLM.from_pretrained(self.config.neuron.model_name)

# Move the model to the appropriate device
self.model = self.model.to(self.device)

# Set up a decentralized optimizer that will average with peers in background
self.opt = torch.optim.AdamW(self.model.parameters(), lr=self.config.neuron.lr)

# Init Gradient Averager
self.grad_averager = DTGradientAverager(
    self.model.parameters(),
    dht=self.dht,
    prefix=f"{self.config.neuron.run_id}_grad_averager",
    compression=hivemind.Uniform8BitQuantization(),
    # reuse_grad_buffers=True,
    accumulate_grads_on=torch.device(self.device),
    start=True,
    next_chunk_timeout=30.0,
)

# Init Tracker
self.tracker = ProgressTracker(
    dht=self.dht,
    prefix=f"{self.config.neuron.run_id}",
    target_batch_size=self.config.neuron.global_batch_size_train,
    start=True,
)

# Init State Averager
self.state_averager = DTStateAverager(
    optimizer=self.opt,
    initialize_optimizer=False,
    dht=self.dht,
    prefix=f"{self.config.neuron.run_id}_state_averager",
    state_compression=hivemind.Uniform8BitQuantization(),
    start=True,
    next_chunk_timeout=30.0,
)

# Load dataset
self.dataset_loader = ()
dataset_length = 968000015
self.dataset_indices = bitarray(dataset_length)

self.step_scheduled = False
self.local_epoch, self.local_samples = 0, 0

search_start = random.choice(
    range(
        len(self.dataset_indices) - self.config.neuron.training_examples_per_miner + 1
    )
)
start = self.dataset_indices.index(
    bitarray("0" * self.config.neuron.training_examples_per_miner), search_start
)
group = [
    i for i in range(start, start + self.config.neuron.training_examples_per_miner)
]
self.dataset_indices[group] = True

self.config.neuron.local_batch_size_train = 100
# Create Dataloader
dataloader = SubsetFalconLoader(
    batch_size=self.config.neuron.local_batch_size_train,
    sequence_length=1024,
    rows=group,
)

total_loss = 0
# Train data for one epoch
for index, batch in enumerate(dataloader):
    inputs = batch.to(self.device)

    # Forward pass
    outputs = self.model(input_ids=inputs, labels=inputs)

    # Normalize loss to account for batch accumulation
    loss = outputs.loss

    # Accumulate Total Loss
    total_loss += outputs.loss.detach().item()

    # Backward Pass
    loss.backward()

    # Copy gradients
    gradients = tuple(
        (
            param.grad.detach().cpu().clone()
            if param.grad is not None
            else torch.zeros_like(param)
        )
        for param in self.model.parameters()
    )

    # Accumulate Gradients
    self.grad_averager.accumulate_grads_(batch_size=len(inputs))

    # Zero Gradients
    self.opt.zero_grad()

    # Update Tracker
    self.local_samples += 1
    self.tracker.report_local_progress(self.local_epoch, self.local_samples)

    # Log accumulation status
    bt.logging.info(
        f"Local samples: {self.local_samples} | Local epoch: {self.local_epoch} | Loss: {outputs.loss.detach().item():.2f}"
    )
    bt.logging.info(
        f"Global samples: {self.tracker.global_progress.samples_accumulated} | Global epoch: {self.tracker.global_progress.epoch} | Number of Peers: {self.tracker.global_progress.num_peers}"
    )

# breakpoint()
from datetime import datetime

while True:
    if datetime.now().strftime("%H:%M:%S") <= "16:40:00":
        continue
    else:
        break

# bt.logging.info("Scheduling Step")
# next_step_time = hivemind.get_dht_time() + 15
# next_step_control = self.grad_averager.schedule_step(scheduled_time = next_step_time)
# time.sleep(15)

bt.logging.info("Performing Gradient Averaging")
# self.grad_averager.step(control = next_step_control)
# next_step_control = self.grad_averager.schedule_step()
time.sleep(10)
bt.logging.info("Scheduling Grad Step")
# self.grad_averager.step(control = next_step_control, timeout=120)
self.grad_averager.step()
bt.logging.info("Model Weights Before Optimizer Step")
bt.logging.info([layer for layer in self.model.parameters()][-1][-10:])
with self.grad_averager.use_averaged_gradients():  # this will fill param.grads with aggregated gradients
    bt.logging.info("Performing Optimizer Step")
    self.opt.step()  # update model parameters using averaged gradients
bt.logging.info("Model Weights After Optimizer Step")
bt.logging.info([layer for layer in self.model.parameters()][-1][-10:])
self.grad_averager.reset_accumulated_grads_()  # prepare for next step
