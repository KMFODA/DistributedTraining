import math
import os
import random

import numpy as np
import torch
import wandb
from torch_optimizer import Lamb
from transformers import AutoModelForCausalLM
from template.data import SubsetFalconLoader


# Custom learning rate scheduler function
def get_lr(it, warmup_iters, lr_decay_iters, learning_rate, min_lr):
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

def load_tokens(filename):
    npt = np.load(filename)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoaderLite:
    def __init__(self, B, T, split):
        self.B = B
        self.T = T
        assert split in {'train', 'val'}

        # get the shard filenames
        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        self.reset()

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B*T + 1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = 0
        return x, y

# Function to run training with specified optimizer and learning rate
def train_model(optimizer_name, learning_rate, global_target_batch_size=524_288, num_epochs=1, warmup_proportion=0.1, min_lr=1e-6):
    # Initialize wandb with configuration details
    wandb.init(project="optimizer_ablation_study", config={
        "optimizer": optimizer_name,
        "learning_rate": learning_rate,
        "target_batch_size": global_target_batch_size,
        "num_epochs": num_epochs,
        "warmup_proportion": warmup_proportion,
        "min_lr": min_lr
    })

    config = wandb.config

    # Load the model
    model = AutoModelForCausalLM.from_pretrained("kmfoda/gpt2-200m")
    model.to("cuda")

    # Set up the optimizer
    if optimizer_name == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    elif optimizer_name == "LAMB":
        optimizer = Lamb(model.parameters(), lr=learning_rate)
    else:
        raise ValueError("Unsupported optimizer")

    BATCH_SIZE = 1    
    SEQ_LENGTH = 1024
    grad_accum_steps = global_target_batch_size // (BATCH_SIZE * SEQ_LENGTH)

    total_steps = global_target_batch_size * 10 * num_epochs
    warmup_steps = int(warmup_proportion * total_steps)
    lr_decay_iters = total_steps - warmup_steps

    local_samples = 0
    step = 0
    current_step = 0
    lr = get_lr(step, warmup_steps, lr_decay_iters, learning_rate, min_lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
    #train_loader = DataLoaderLite(B=B, T=T, process_rank=0, num_processes=1, split="train")

    for current_step in range(total_steps):
        dataloader = SubsetFalconLoader(
        batch_size=BATCH_SIZE,
        sequence_length=SEQ_LENGTH,
        rows=random.choices(range(0, 519_000_000), k=1000),
        )

        for i, batch in enumerate(dataloader):
    
            inputs = batch.to("cuda")

            # Forward pass
            outputs = model(input_ids=inputs, labels=inputs)

            loss = outputs.loss
            scaled_loss = loss / grad_accum_steps
            print(f"Current step {current_step+1}, Batch {i+1}, Loss: {loss.item()}")
            scaled_loss.backward()

            local_samples += BATCH_SIZE
            step += BATCH_SIZE
            current_step += BATCH_SIZE
                
            if current_step % grad_accum_steps == 0:

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                lr = get_lr(step, warmup_steps, 
                            lr_decay_iters, learning_rate, min_lr)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

                optimizer.step()
                optimizer.zero_grad()  # Reset gradients after each step
                local_samples = 0
        
        # TODO - implement on chain
        # tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
        # tokens_per_sec = tokens_processed / dt

        # Log metrics to wandb
        wandb.log({
            "loss": loss.item(),
            "current step": current_step+1,
            "batch": i+1,
            "learning_rate": lr,
            "optimizer": optimizer_name,
            "learning_rate_config": learning_rate
        })
        
    # Finish the wandb run
    wandb.finish()


# Initialize wandb
wandb.init(project="optimizer_ablation_study")

# Define the grid search parameters
optimizers = ["AdamW", "LAMB"]
learning_rates = [5e-3, 5e-2, 5e-4]

# Run the grid search
for optimizer_name in optimizers:
    for lr in learning_rates:
        train_model(optimizer_name, lr, min_lr=lr*0.1)
