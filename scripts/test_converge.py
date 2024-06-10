import random
import wandb
import torch
import math
from transformers import AutoModelForCausalLM
from template.data.dataset import SubsetFalconLoader
from torch_optimizer import Lamb

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

# Initialize wandb
wandb.init(project="optimizer_ablation_study")

# Function to run training with specified optimizer and learning rate
def train_model(optimizer_name, learning_rate, global_target_batch_size=32000, num_epochs=1, warmup_proportion=0.1, min_lr=1e-6):
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
    model = AutoModelForCausalLM.from_pretrained("kmfoda/gpt2-500m")
    model.to("cuda")

    # Set up the optimizer
    if optimizer_name == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    elif optimizer_name == "LAMB":
        optimizer = Lamb(model.parameters(), lr=learning_rate)
    else:
        raise ValueError("Unsupported optimizer")

    BATCH_SIZE = 2    

    total_steps = 640_000 * num_epochs
    warmup_steps = int(warmup_proportion * total_steps)
    lr_decay_iters = total_steps - warmup_steps

    local_samples = 0
    step = 0
    current_step = 0
    lr = get_lr(step, warmup_steps, lr_decay_iters, learning_rate, min_lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    while current_step < total_steps:
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
            scaled_loss = loss / global_target_batch_size / BATCH_SIZE
            print(f"Current step {current_step+1}, Batch {i+1}, Loss: {loss.item()}")
            scaled_loss.backward()

            local_samples += BATCH_SIZE
            step += BATCH_SIZE
            current_step += BATCH_SIZE

            if local_samples >= global_target_batch_size:
                lr = get_lr(step, warmup_steps, 
                            lr_decay_iters, learning_rate, min_lr)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

                optimizer.step()
                optimizer.zero_grad()  # Reset gradients after each step
                local_samples = 0

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

# Define the grid search parameters
optimizers = ["AdamW", "LAMB"]
learning_rates = [5e-3, 5e-2, 1e-3]

# Run the grid search
for optimizer_name in optimizers:
    for lr in learning_rates:
        train_model(optimizer_name, lr)
