import hivemind
from hivemind.utils.logging import use_hivemind_log_handler
import torch
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
from template.data.dataset import SubsetFalconLoader
import logging
from contextlib import contextmanager
from template.base.neuron import BaseNeuron
import time

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


use_hivemind_log_handler("in_root_logger")
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

config = BaseNeuron.config()
config.wallet.name = "test_dt"
config.wallet.hotkey = "validator_1"
config.netuid = 30
config.axon.port = 22026
config.dht.port = 22027

dht = hivemind.DHT(
    host_maddrs=["/ip4/0.0.0.0/tcp/0", "/ip4/0.0.0.0/udp/0/quic"],
    initial_peers=config.neuron.initial_peers, 
    start=True
)

print(dht.get_visible_maddrs())
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
    accumulate_grads_on=torch.device("cuda")
)

tracker = hivemind.optim.progress_tracker.ProgressTracker(
    dht=dht, 
    prefix="test", 
    target_batch_size=global_target_batch_size,
    start=True
)

tokenizer = AutoTokenizer.from_pretrained(config.neuron.model_name)
# Add the EOS token as PAD token to ensure our dataloader doesn't throw an error for sequences of unequal length
tokenizer.pad_token = tokenizer.eos_token

#total_batch_size = 0
step_scheduled = False
local_epoch, local_samples = 0, 0

print("Starting training..")
for i in range(0, 100):
    print("Getting new data..")
    dataloader = SubsetFalconLoader(
    batch_size=1, sequence_length=1024, rows=random.choices(range(0,968000015), k = 25)
    )
    
    for i, batch in enumerate(dataloader):

        inputs = batch.to("cuda")

        # Forward pass
        outputs = model(input_ids=inputs, labels=inputs)
        
        loss = outputs.loss
        print(loss)
        loss.backward()

        grad_averager.accumulate_grads_(batch_size=1)
        local_samples += 1  # increment the total batch size
        
        tracker.report_local_progress(local_epoch, local_samples)
        print("local samples:", local_samples, "global_samples:", tracker.global_progress.samples_accumulated)
        print("local epoch:", local_epoch, "global epoch", tracker.global_progress.epoch)
        
        # TODO!! This is the part that needs proper implementation into our pipeline
        # TODO!! I.e. if a few peers have accumulated just 200 samples, and then they are idle until the target_batch_size
        # TODO!! how will gradient_avering work? Should we load state from peers only when querying miners, checking if they are out of sync?
        # if local_epoch < tracker.global_progress.epoch:
        #     # if peer is out of sync, synchronize it with the swarm
        #     grad_averager.load_state_from_peers()

        time.sleep(1)
        if global_target_batch_size - tracker.global_progress.samples_accumulated <= 25 and not step_scheduled:  # Prepare groups for averaging
            print("scheduling grad step..")
            next_step_control = grad_averager.schedule_step()
            step_scheduled = True  # Set the flag to True

        # # aggregate gradients and perform optimizer step when target batch size is reached
        if tracker.global_progress.samples_accumulated >= global_target_batch_size:
            with tracker.pause_updates():
                print("grad stepping..")
                grad_averager.step(control=next_step_control)
                with grad_averager.use_averaged_gradients():  # this will fill param.grads with aggregated gradients
                    print("opt steppeing..")
                    opt.step()  # update model parameters using averaged gradients
                grad_averager.reset_accumulated_grads_()  # prepare for next step
                local_epoch = tracker.update_epoch(local_epoch + 1)
                local_samples = 0  
                step_scheduled = False 
