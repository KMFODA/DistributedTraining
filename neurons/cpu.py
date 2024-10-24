from transformers import AutoModelForCausalLM
from bitsandbytes.optim import LAMB
from torch.optim import optimizer
from distributed_training.utils.gradient_averager import (
    DTGradientAverager,
)
import torch
import hivemind
import os
from distributed_training import __version__, __run__
from torch.optim import Adam

device = "cpu"
model_name = "distributed/optimized-gpt2-500m"
announce_maddrs = [
    f"/ip4/{os.environ['RUNPOD_PUBLIC_IP']}/tcp/{os.environ['RUNPOD_TCP_PORT_70001']}"
]
initial_peer = "/ip4/161.97.156.125/tcp/8000/p2p/12D3KooWFBR9RY52qMki1V59QpMoHKcW8qz2LhAsD8No4pLtwMC2"
run_id = f"v{__version__.replace('.','_')}_r{__run__}"
optimizer = "lamb"

dht = hivemind.DHT(
    host_maddrs=[
        f"/ip4/0.0.0.0/tcp/{os.environ['RUNPOD_TCP_PORT_70001']}",
        f"/ip4/0.0.0.0/udp/{os.environ['RUNPOD_TCP_PORT_70001']}/quic",
    ],
    initial_peers=[initial_peer],
    announce_maddrs=announce_maddrs,
    start=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    revision=str(0),
    trust_remote_code=True,
)
# Move the model to the appropriate device
model.to(device)

# For simplicity only pick layers with a dim of 1
test_layer_indices = [
    i for i, layer in enumerate(model.parameters()) if len(layer.size()) == 1
]

# Init All Reduce Variables
train_timeout = 120
all_reduce_timeout = 420
load_state_timeout = 120
model_upload_retry_limit = 3
model_upload_retry_delay = 10
maximum_steps = 306 * 4  # 10_000_000_000/(32000*1024)
warmup_steps = 62  # 306 / 5
learning_rate_maximum = 0.0025
learning_rate = 0.0025
average_loss = None
weight_decay = 0.1

# Init Optimizer
param_dict = {pn: p for pn, p in model.named_parameters()}
param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
# create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
# i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
optim_groups = [
    {"params": decay_params, "weight_decay": weight_decay},
    {"params": nodecay_params, "weight_decay": 0.0},
]
if optimizer == "lamb":
    opt = LAMB(optim_groups, lr=learning_rate_maximum, betas=(0.9, 0.95), eps=1e-8)
elif optimizer == "adam":
    opt = Adam(optim_groups, lr=learning_rate_maximum, betas=(0.9, 0.95), eps=1e-8)

# Init Gradient Averager
grad_averager = DTGradientAverager(
    model.parameters(),
    dht=dht,
    prefix=f"{run_id}_grad_averager",
    # compression=hivemind.Uniform8BitQuantization(),
    accumulate_grads_on=torch.device(device),
    start=True,
    min_group_size=5,
    min_matchmaking_time=30.0,
    request_timeout=10.0,
    next_chunk_timeout=45.0,
    allreduce_timeout=all_reduce_timeout - 30.0 - 15.0,
)

for param in model.parameters():
    if len(param.size()) == 1:
        param.grad += 0.5
        break

opt.step()
print("Successfully stepped optimizer")
