import hivemind
from hivemind.utils.logging import use_hivemind_log_handler
import torch
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
from template.data.dataset import SubsetFalconLoader
import logging
from contextlib import contextmanager
from template.base.neuron import BaseNeuron
import os
import time


class MyGradientAverager(hivemind.optim.grad_averager.GradientAverager):
    """
    Needs this wrapper class to ensure device is set properly when averaging gradients
    see: https://github.com/learning-at-home/hivemind/blob/d20e81017481aa2028efc33217522248aabd7d95/hivemind/optim/grad_averager.py#L224
    """

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


use_hivemind_log_handler("in_root_logger")
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

config = BaseNeuron.config()

version = "4"
address = "206.125.129.213"
announce_maddrs = [f"/ip{version}/{address}/tcp/43312"]

dht = hivemind.DHT(
    host_maddrs=[
        f"/ip4/0.0.0.0/tcp/43312",
        f"/ip4/0.0.0.0/udp/43312/quic",
    ],
    # initial_peers=["/ip4/161.97.156.125/tcp/8001/p2p/12D3KooWF7Ryy6537eehmd19DpSpexQ8gZbsgNornpFNknhRGmqX"],
    announce_maddrs=announce_maddrs,
    start=True,
)
print(dht.get_visible_maddrs())

# Write the visible_maddrs to a text file
with open("visible_maddrs.txt", "w") as f:
    for maddr in dht.get_visible_maddrs():
        f.write(str(maddr) + "\n")

time.sleep(15)

model = AutoModelForCausalLM.from_pretrained(config.neuron.model_name)
# Move the model to the appropriate device
model = model.to("cuda")

# Set up a decentralized optimizer that will average with peers in background
opt = torch.optim.AdamW(model.parameters(), lr=config.neuron.lr)

global_target_batch_size = 200  # set your target batch size
grad_averager = MyGradientAverager(
    model.parameters(),
    dht=dht,
    prefix=f"test",
    start=True,
    reuse_grad_buffers=True,
)

tracker = hivemind.optim.progress_tracker.ProgressTracker(
    dht=dht, prefix="test", target_batch_size=global_target_batch_size, start=True
)

tokenizer = AutoTokenizer.from_pretrained(config.neuron.model_name)
# Add the EOS token as PAD token to ensure our dataloader doesn't throw an error for sequences of unequal length
tokenizer.pad_token = tokenizer.eos_token

# total_batch_size = 0
step_scheduled = False
local_epoch, local_samples = 0, 0

# Log gradients to a file
# log_file = "/workspace/DTraining/logs/gradients.txt"
# os.makedirs(os.path.dirname(log_file), exist_ok=True)
# with open(log_file, "w") as f:
print("Starting training..")
for i in range(0, 100):
    print("Getting new data..")
    dataloader = SubsetFalconLoader(
        batch_size=1,
        sequence_length=1024,
        rows=random.choices(range(0, 968000015), k=25),
    )

    for i, batch in enumerate(dataloader):

        inputs = batch.to("cuda")

        # Forward pass
        outputs = model(input_ids=inputs, labels=inputs)

        loss = outputs.loss
        print(loss)
        loss.backward()

        # Only use this if reuse_grad_buffers=False
        # grad_averager.accumulate_grads_(batch_size=1)

        # # Store gradients
        # gradients_2 = list(grad_averager._grad_accumulators())[-1]
        # gradients_3 = list(grad_averager._grads_from_parameters())[-1]
        # # Get the gradients directly from the model parameters
        # gradients_model = [param.grad for param in model.parameters()][-1]

        # # Write the gradients to a text file
        # with open('gradients.txt', 'w') as f:
        #     for gradients in [gradients_2, gradients_3, gradients_model]:
        #         for gradient in gradients:
        #             f.write(str(gradient) + "\n")
        #         f.write("\n")

        # # Compare the gradients programmatically
        # print(torch.allclose(gradients_2, gradients_3)) #False
        # print(torch.allclose(gradients_3, gradients_model)) #True
        # print(torch.allclose(gradients_2, gradients_model)) #False

    local_samples += 1  # increment the total batch size

    tracker.report_local_progress(local_epoch, local_samples)
    print(
        "local samples:",
        local_samples,
        "global_samples:",
        tracker.global_progress.samples_accumulated,
    )
    print("local epoch:", local_epoch, "global epoch", tracker.global_progress.epoch)

    if local_epoch < tracker.global_progress.epoch:
        # if peer is out of sync, synchronize it with the swarm
        grad_averager.load_state_from_peers()

    time.sleep(1)
    if (
        global_target_batch_size - tracker.global_progress.samples_accumulated <= 25
        and not step_scheduled
    ):  # Prepare groups for averaging
        print("scheduling grad step..")
        next_step_control = grad_averager.schedule_step()
        step_scheduled = True  # Set the flag to True

    # aggregate gradients and perform optimizer step when target batch size is reached
    if tracker.global_progress.samples_accumulated >= global_target_batch_size:
        with tracker.pause_updates():
            print("grad stepping..")
            grad_averager.step(control=next_step_control)
            with grad_averager.use_averaged_gradients():  # this will fill param.grads with aggregated gradients
                print("opt stepping..")
                opt.step()  # update model parameters using averaged gradients
            grad_averager.reset_accumulated_grads_()  # prepare for next step
            local_epoch = tracker.update_epoch(local_epoch + 1)
            local_samples = 0
            step_scheduled = False
