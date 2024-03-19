#import hivemind
#from hivemind.utils.logging import use_hivemind_log_handler
import logging
import math
import random
import time
from contextlib import contextmanager

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import wandb
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          default_data_collator)
from transformers.optimization import (get_inverse_sqrt_schedule,
                                       get_linear_schedule_with_warmup)

from template.base.neuron import BaseNeuron
from template.data.dataset import SubsetFalconLoader

# class WarmupCosineSchedule(LambdaLR):
#     """ Linear warmup and then cosine decay.
#         Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
#         Decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps following a cosine curve.
#         If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
#     """
#     def __init__(self, optimizer, warmup_steps, t_total, cycles=.5, last_epoch=-1):
#         self.warmup_steps = warmup_steps
#         self.t_total = t_total
#         self.cycles = cycles
#         super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

#     def lr_lambda(self, step):
#         if step < self.warmup_steps:
#             return float(step) / float(max(1.0, self.warmup_steps))
#         # progress after warmup
#         progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
#         return max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))

# class MyGradientAverager(hivemind.optim.grad_averager.GradientAverager):
#     '''
#     Needs this wrapper class to ensure device is set properly when averaging gradients
#     see: https://github.com/learning-at-home/hivemind/blob/d20e81017481aa2028efc33217522248aabd7d95/hivemind/optim/grad_averager.py#L224
#     '''
#     @contextmanager
#     @torch.no_grad()
#     def use_averaged_gradients(self):
#         """Substitute model's main gradients with averaged gradients"""
#         self._new_averaged_grads = False
#         with self.get_tensors() as averaged_grads:
#             assert len(averaged_grads) == len(self.parameters)
#             try:
#                 old_grads = [param.grad for param in self.parameters]
#                 for param, new_grad in zip(self.parameters, averaged_grads):
#                     # move new_grad to the same device as param before assigning
#                     param.grad = new_grad.to(param.device)
#                 yield averaged_grads
#             finally:
#                 for param, old_grad in zip(self.parameters, old_grads):
#                     param.grad = old_grad

#warmup_iters = 4 # how many steps to warm up for (opt.step)
lr_decay_iters = 150 # total opt.steps (global_target_batch_size*100)
learning_rate = 2.726e-3 #2.5e-3 #2.5e-4 # max learning rate 0.001 1e-3
min_lr = 2.5e-7 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

# def get_lr(it): # From: https://github.com/karpathy/nanoGPT/blob/master/train.py
#     # 1) linear warmup for warmup_iters steps
#     if it < warmup_iters:
#         return learning_rate #* it / warmup_iters
#     # 2) if it > lr_decay_iters, return min learning rate
#     if it > lr_decay_iters:
#         return min_lr
#     # 3) in between, use cosine decay down to min learning rate
#     decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
#     assert 0 <= decay_ratio <= 1
#     coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
#     return min_lr + coeff * (learning_rate - min_lr)

# use_hivemind_log_handler("in_root_logger")
# root_logger = logging.getLogger()
# root_logger.setLevel(logging.INFO)

config = BaseNeuron.config()
# config.wallet.name = "test_dt"
# config.wallet.hotkey = "validator_1"
# config.netuid = 30
# config.axon.port = 22026
# config.dht.port = 22027

# version = "4"
# address = "64.247.206.230"
# announce_maddrs = [f"/ip{version}/{address}/tcp/26822"]

# dht = hivemind.DHT(
#     host_maddrs=["/ip4/0.0.0.0/tcp/26822", "/ip4/0.0.0.0/udp/26822/quic"],
#     #initial_peers=[""], 
#     announce_maddrs=announce_maddrs,
#     start=True
# )

# print(dht.get_visible_maddrs())
# file_path = 'visible_maddrs.txt'
# with open(file_path, 'w') as file:
#     file.write(f"{dht.get_visible_maddrs()}\n")
wandb.init(project="test_single_instance_LRScheduler_0.0025LR")
# time.sleep(20)
model = AutoModelForCausalLM.from_pretrained("kmfoda/gpt2-200m")
# Move the model to the appropriate device
model = model.to("cuda")

# # Set a new dropout value 
# new_dropout_value = 0.0  # Good for pretraining - as per nanoGPT
# for module in model.modules():
#     if hasattr(module, 'dropout'):
#         module.dropout.p = new_dropout_value

# # apply special scaled init to the residual projections, per GPT-2 paper
# for pn, p in model.named_parameters():
#     if pn.endswith('c_proj.weight'):
#         torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * model.config.n_layer))


# for module in model.modules():
#     if isinstance(module, nn.Linear):
#         torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
#         if module.bias is not None:
#             torch.nn.init.zeros_(module.bias)
#     elif isinstance(module, nn.Embedding):
#         torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

# # # From: https://github.com/karpathy/nanoGPT/blob/eba36e84649f3c6d840a93092cb779a260544d08/model.py#L264C9-L275C10
# param_dict = {pn: p for pn, p in model.named_parameters()}
# param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
# decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
# nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
# params = [
#     {'params': decay_params, 'weight_decay': 0.01},
#     {'params': nodecay_params, 'weight_decay': 0.0}
# ]

# Set up a decentralized optimizer that will average with peers in background
opt = torch.optim.AdamW(
        model.parameters(), 
        lr=learning_rate)
        #betas=(0.9, 0.95), 
        #eps=1e-8)
#scheduler = WarmupCosineSchedule(opt, warmup_steps=warmup_iters, t_total=lr_decay_iters)
scheduler = get_linear_schedule_with_warmup(opt, num_warmup_steps=5, num_training_steps=50)
# scheduler = get_inverse_sqrt_schedule(opt, num_warmup_steps=5)
# print(dict(scheduler))

# opt = Lamb(
#         params,
#         lr=5/((2**2.5)*(10**3)),
#         betas=(0.9, 0.95),
#         eps=1e-8,
#         weight_decay=0.01,
#         clamp_value=1.0,
#         debias=True,
#     )


# Target batch size for gradient accumulation
global_target_batch_size = 10_000

#global_target_batch_size = 200  # set your target batch size
# grad_averager = MyGradientAverager(
#     model.parameters(), 
#     dht=dht, 
#     prefix=f"test",
#     start=True,
#     accumulate_grads_on=torch.device("cuda")
# )

# tracker = hivemind.optim.progress_tracker.ProgressTracker(
#     dht=dht, 
#     prefix="test", 
#     target_batch_size=global_target_batch_size,
#     start=True
# )

tokenizer = AutoTokenizer.from_pretrained("kmfoda/gpt2-677m")
# Add the EOS token as PAD token to ensure our dataloader doesn't throw an error for sequences of unequal length
tokenizer.pad_token = tokenizer.eos_token

#total_batch_size = 0
# step_scheduled = False
# local_epoch, local_samples = 0, 0

# Initialize an accumulator for the batch size
accumulated_batch_size = 0

loss_values = []

# opt.zero_grad()

# iter_num = 1
print("Starting training..")
current_epoch = 0
pbar = tqdm(total=global_target_batch_size, desc="Training Progress")

# determine and set the learning rate for initial iterations
# for param_group in opt.param_groups:
#     param_group['lr'] = get_lr(iter_num)

while True:

    # Define the size of the group as per "training_examples_per_miner"
    group_size = 25
    total_dataset_size = 968000015  # Example dataset size
    max_index_value = total_dataset_size - group_size
    start_index = random.randint(0, max_index_value)
    group = list(range(start_index, start_index + group_size))
    # Use this group in your DataLoader
    #print("Getting new data..")
    dataloader = SubsetFalconLoader(
        batch_size=1, sequence_length=1024, rows=group
    )
    
    for i, batch in enumerate(dataloader):
        
        inputs = batch.to("cuda")

        # Forward pass
        outputs = model(input_ids=inputs, labels=inputs)
        
        loss = outputs.loss 
        #print(loss.item())
        loss = loss / global_target_batch_size
        loss.backward()

        #grad_averager.accumulate_grads_(batch_size=inputs.size(0))
        #local_samples += inputs.size(0)  # increment the local batch size
        
        # Accumulate batch size
        accumulated_batch_size += inputs.size(0)

        # Check if accumulated batch size reached the target
        if accumulated_batch_size >= global_target_batch_size:
            print("Opt stepping..")
            # Update model parameters
            opt.step()
            opt.zero_grad()
            scheduler.step()
            # determine and set the learning rate for next iterations
            # Reset accumulated batch size
            # for param_group in opt.param_groups:
            #     param_group['lr'] = get_lr(iter_num)
            accumulated_batch_size = 0
            #iter_num +=1
            
            # Update the current_epoch 
            current_epoch += 1
            
        # Log loss and update progress bar
        loss_value = loss.item() * global_target_batch_size
        wandb.log({"loss": loss_value})
        pbar.set_description(f'Epoch: {current_epoch} - Current loss: {loss_value:.4f}')
        pbar.update()
        
        #loss_values.append(loss.item())  # Append the loss to the list

        # # Plotting the loss values
        # plt.figure(figsize=(10, 6))
        # plt.plot(loss_values, label='Loss Value')
        # plt.title('Training Loss')
        # plt.xlabel('Step')
        # plt.ylabel('Loss')
        # plt.legend()
        # plt.grid(True)

        # # Save the plot to a file
        # plt.savefig('training_loss_plot.png')
        # plt.close()  # Close the plot to free memory
        
        # tracker.report_local_progress(local_epoch, local_samples)
        # print("local samples:", local_samples, "global_samples:", tracker.global_progress.samples_accumulated)
        # print("local epoch:", local_epoch, "global epoch", tracker.global_progress.epoch)
        
        # if local_epoch < tracker.global_progress.epoch:
        #     # if peer is out of sync, synchronize it with the swarm
        #     grad_averager.load_state_from_peers()

        # # # time.sleep(1)
        # if global_target_batch_size - tracker.global_progress.samples_accumulated <= 100 and not step_scheduled:  # Prepare groups for averaging
        #     print("scheduling grad step..")
        #     next_step_control = grad_averager.schedule_step()
        #     step_scheduled = True  # Set the flag to True

        # # aggregate gradients and perform optimizer step when target batch size is reached
        # if tracker.global_progress.samples_accumulated >= global_target_batch_size:
        #     with tracker.pause_updates():
        #         print("grad stepping..")
        #         grad_averager.step(control=next_step_control, wait=True, timeout=120)
        #         with grad_averager.use_averaged_gradients():  # this will fill param.grads with aggregated gradients
        #             print("opt steppeing..")
        #             opt.step()  # update model parameters using averaged gradients
        #         grad_averager.reset_accumulated_grads_()  # prepare for next step
        #         local_epoch = tracker.update_epoch(local_epoch + 1)
        #         local_samples = 0  
        #         step_scheduled = False 
        #         iter_num +=1
        #     # scheduler.step()
        #     # determine and set the learning rate for next iterations
        #     for param_group in opt.param_groups:
        #         param_group['lr'] = get_lr(iter_num)
