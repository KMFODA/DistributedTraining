import torch
import os
import time
from hivemind.utils import use_hivemind_log_handler
import logging
import torch.nn as nn
import torch.optim as optim
from hivemind import DHT
from hivemind.optim.grad_averager import GradientAverager

def launch_dht_instances(n_peers: int, **kwargs):
    dhts = [DHT(start=True, **kwargs)]
    initial_peers = dhts[0].get_visible_maddrs()

    dhts.extend(
        DHT(initial_peers=initial_peers, start=True, await_ready=False, **kwargs)
        for _ in range(n_peers - 1)
    )
    for process in dhts[1:]:
        process.wait_until_ready()

    return dhts

# Logging setup
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

use_hivemind_log_handler("nowhere")

logfile = "logfile.log"
if os.path.exists(logfile):
    os.remove(logfile)

handler = logging.FileHandler(logfile)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

import torch
import os
import time
from hivemind.utils import use_hivemind_log_handler
import logging
import torch.nn as nn
import torch.optim as optim
from hivemind import DHT
from hivemind.optim.grad_averager import GradientAverager

# ... [Previous setup code remains the same] ...

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

def create_consistent_data():
    torch.manual_seed(42)  # Set a fixed seed for reproducibility
    batch_size = 4
    x = torch.randn(batch_size, 10)
    y = torch.randn(batch_size, 1)
    return x, y

def initialize_model():
    torch.manual_seed(43)  # Set a fixed seed for model initialization
    return SimpleModel()

def compute_gradients(model, x, y, num_accumulation_steps, norm=True):
    model.zero_grad()
    for _ in range(num_accumulation_steps):
        output = model(x)
        loss = nn.MSELoss()(output, y)
        if norm:
            loss = loss / num_accumulation_steps  # Normalize loss
        loss.backward()

def test_without_averager(x, y):
    model = initialize_model()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    num_accumulation_steps = 4

    print("Without GradientAverager:")
    compute_gradients(model, x, y, num_accumulation_steps)
    
    print("Gradients with manual normalization:")
    for name, param in model.named_parameters():
        print(f"{name}: {param.grad}")

def simulate_distributed_training(x, y, num_nodes=2):
    dhts = launch_dht_instances(2)
    
    models = [initialize_model() for _ in range(num_nodes)]
    optimizers = [optim.SGD(model.parameters(), lr=0.01) for model in models]

    grad_averagers = [
        GradientAverager(model.parameters(), dht=dht, prefix="test", start=True)
        for model, dht in zip(models, dhts)
    ]

    num_accumulation_steps = 4

    print("\nWith GradientAveragers:")
    for i, (model, optimizer, grad_averager) in enumerate(zip(models, optimizers, grad_averagers)):
        print(f"Node {i}:")
        compute_gradients(model, x, y, num_accumulation_steps, norm=False)
        grad_averager.accumulate_grads_(batch_size=x.size(0))

        print("Local gradients before averaging:")
        for name, param in model.named_parameters():
            print(f"{name}: {param.grad}")

    futures = []
    for grad_averager in grad_averagers:
        futures.append(grad_averager.step(wait=False, allow_retries=False))
    
    time.sleep(1)
    for future in futures:
        assert future.result()
    
    for i, (model, grad_averager) in enumerate(zip(models, grad_averagers)):
        print(f"\nNode {i} after averaging:")
                
        print("Model parameters:")
        with grad_averager.use_averaged_gradients():
            for name, param in model.named_parameters():
                print(f"{name}: {param.grad}")
        

if __name__ == "__main__":
    x, y = create_consistent_data()
    test_without_averager(x, y)
    simulate_distributed_training(x, y)