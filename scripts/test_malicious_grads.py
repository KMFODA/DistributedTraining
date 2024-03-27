import gc
import random
from itertools import islice

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from template.data.dataset import SubsetFalconLoader

# Setting a random seed for reproducibility
random.seed(42)
torch.manual_seed(42)

# Device configuration for CUDA compatibility
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def statistical_gradient_checks(gradients, z_layer, z_full, check_type="sum_gradients"):
    device = gradients.device  # Get the device of the gradients tensor

    # Ensure all tensors are on the same device
    z_layer = z_layer.to(device)
    z_full = z_full.to(device)
    if check_type == "sum_gradients":
        metric = float(torch.sum(torch.abs(gradients)))
    elif check_type == "random_direction_projection_layer":
        # Compute the projection of the gradients on z
        projection = torch.dot(gradients.view(-1), z_layer)
        metric = projection.item()
    elif check_type == "random_direction_projection_full":
        
        # Compute the projection of the gradients on z
        projection = torch.dot(gradients.view(-1), z_full)
        metric = projection.item()
    elif check_type == "gradient_norms" or check_type == "gradient_norms_full":
        metric = torch.norm(gradients).item()
    else:
        raise ValueError(f"Unknown gradient check type: {check_type}")
    return metric

def train(model, dataloader, optimizer, trainer_type, metrics_log, indices, mimic_models, z_layer, z_full):
    model.train()
    optimizer.zero_grad()
    total_loss = 0
    
    idx = 1
    
    print(trainer_type + " training..")
    
    
    if trainer_type == "idle" and mimic_models:
        print("Mimicking model..")
        mimic_model = random.choice(list(mimic_models.values()))
        model.load_state_dict(mimic_model.state_dict())
        model = model.to(device)
    
    if trainer_type != "idle":
        for idx, batch in enumerate(tqdm(dataloader, desc=f"Training {trainer_type}")):
            
            inputs = batch.to("cuda")

            outputs = model(input_ids=inputs, labels=inputs)
            loss = outputs.loss
            # Accumulate Total Loss
            total_loss += loss.detach().item()
            loss.backward()

            if trainer_type == "malicious":
                with torch.no_grad():
                    for param in model.parameters():
                        if param.grad is not None:
                            param.grad += torch.randn_like(param.grad) * 0.01
        
    gradients = tuple(param.grad.detach().cpu().clone() if param.grad is not None else torch.zeros_like(param) for param in model.parameters())
    gradients = gradients[indices]
    
    for check_type in metrics_log[trainer_type].keys():
        if check_type == "random_direction_projection_full" or check_type == "gradient_norms_full":
            gradients = torch.cat([param.grad.detach().cpu().clone().view(-1) if param.grad is not None else torch.zeros_like(param).view(-1) for param in model.parameters()])
        else:
            gradients = tuple(param.grad.detach().cpu().clone() if param.grad is not None else torch.zeros_like(param) for param in model.parameters())
            gradients = gradients[indices]
        
        if not check_type == "loss" and not check_type == "perplexity":    
            metric = statistical_gradient_checks(gradients, z_layer, z_full, check_type)
        else:
            if check_type == "loss":
                metric = total_loss / idx
            elif check_type == "perplexity":
                perplexity = torch.exp(torch.tensor(total_loss / idx))
                metric = perplexity.item()
                
        metrics_log[trainer_type][check_type].append(metric)
        
    optimizer.step()
    optimizer.zero_grad()
    # Clean memory
    torch.cuda.empty_cache()
    gc.collect()

def plot_metrics(metrics_log, epoch):
    metrics_df = pd.DataFrame()

    # Compile data into a DataFrame
    for trainer_type, checks in metrics_log.items():
        for check_type, values in checks.items():
            temp_df = pd.DataFrame({
                'Trainer': trainer_type,
                'Check Type': check_type,
                'Metric Value': values,
                'Iteration': list(range(1, len(values)+1))
            })
            metrics_df = pd.concat([metrics_df, temp_df], ignore_index=True)

    # Save DataFrame
    metrics_df.to_csv(f"{epoch}_metrics_data.csv", index=False)
    
    # Define a color palette for your trainers
    palette = {
        "normal_small": "skyblue",
        "normal_small2": "lightblue",
        "normal_large": "lightgreen",
        "normal_large2": "green",
        "idle": "orange",
        "malicious": "red"
    }

    # Plotting
    for check_type in metrics_df['Check Type'].unique():
        plt.figure(figsize=(10, 6))
        plot_data = metrics_df[metrics_df['Check Type'] == check_type]
        
        sns.barplot(data=plot_data, x='Trainer', y='Metric Value', palette=palette)
        
        plt.title(f'Bar Plot of {check_type} for Epoch {epoch}')
        plt.xlabel('Trainer')
        plt.ylabel('Metric Value')
        
        # Save plot
        plt.savefig(f'{epoch}_bar_plot_{check_type}.png')
        plt.close()


def main():

    model_templates = {
        t: AutoModelForCausalLM.from_pretrained("kmfoda/gpt2-67m").to(device)
        for t in ["normal_small", "normal_small2", "normal_large", "normal_large2", "malicious", "idle"]
    }
    mimic_models = {k: model_templates[k] for k in islice(model_templates.keys(), 4)}
    
    optimizers = {
        t: AdamW(model_templates[t].parameters(), lr=5e-5)
        for t in model_templates
    }

    batch_sizes = {
        "normal_small": 2,
        "normal_large": 2,
        "normal_small2": 2,
        "normal_large2": 2,
        "idle": 2, 
        "malicious": 2,
    }
    
    group_sizes = {
        "normal_small": 25,
        "normal_large": 250,
        "normal_small2": 25,
        "normal_large2": 250,
        "idle": 25, 
        "malicious": 25,
    }
    
    epochs = range(10)
    
    total_dataset_size = 968000015  # Example dataset size
    
    for epoch in epochs:
        
        # Layer-wise statistical check setup (already in place)
        test_layer_indices = [i for i, layer in enumerate(model_templates["normal_small"].parameters()) if len(layer.size()) == 1]
        indices = random.choice(test_layer_indices)
        model_example = next(iter(model_templates.values()))  # Get an example model
        example_param = next(islice(model_example.parameters(), indices, None))  # Get the specific parameter for layer-wise
        z_size_layer = example_param.data.size()  # Size of the specific parameter
        
        # Generate `z` for the layer-wise check
        z_layer_wise = torch.randn(z_size_layer)#.to(device)
        z_layer_wise /= torch.norm(z_layer_wise)
        
        # Full gradient space statistical check setup
        # Calculate the total size needed for `z` that covers all model gradients
        total_grad_size = sum(p.numel() for p in model_example.parameters() if p.requires_grad)
        
        # Generate `z` for the full gradient space check
        z_full_gradient = torch.randn(total_grad_size)#.to(device)
        z_full_gradient /= torch.norm(z_full_gradient)
        
        
           
        metrics_log = {
            trainer_type: {
                check_type: [] for check_type in ["sum_gradients", "random_direction_projection_layer", "random_direction_projection_full", "gradient_norms", "gradient_norms_full", "loss", "perplexity"]
            } for trainer_type in ["normal_small", "normal_small2", "normal_large", "normal_large2", "idle", "malicious"]
        }
        for trainer_type, model in model_templates.items():
            
            group_size = group_sizes[trainer_type]
            print(group_size, trainer_type)
            total_dataset_size = 968000015  # Example dataset size
            max_index_value = total_dataset_size - group_size
            start_index = random.randint(0, max_index_value)
            group = list(range(start_index, start_index + group_size))
            
            dataloader = SubsetFalconLoader(
                                        batch_size=batch_sizes[trainer_type], 
                                        sequence_length=1024, 
                                        rows=group)

            optimizer = optimizers[trainer_type]
            train(model, dataloader, optimizer, trainer_type, metrics_log, indices, mimic_models=mimic_models if trainer_type == "idle" else None, z_layer=z_layer_wise, z_full=z_full_gradient)

        plot_metrics(metrics_log, epoch)
    
if __name__ == "__main__":
    main()