import os
from datetime import timedelta

import torch
import torch.distributed as dist

# WORLD_RANK=0 WORLD_SIZE=2 MASTER_ADDR=tcp://149.36.0.92:18101 BACKEND=nccl python /root/DTraining/scripts/all_reduce_test_dist.py
# WORLD_RANK=1 WORLD_SIZE=2 MASTER_ADDR=tcp://149.36.0.92:18101 BACKEND=nccl python /root/DTraining/scripts/all_reduce_test_dist.py
WORLD_SIZE = int(os.environ["WORLD_SIZE"])
WORLD_RANK = int(os.environ["WORLD_RANK"])
MASTER_ADDR = str(os.environ["MASTER_ADDR"])
BACKEND = str(os.environ["BACKEND"])

if BACKEND == "gloo":
    device = "cpu"
elif BACKEND == "nccl":
    device = "cuda:0"

dist.init_process_group(
    init_method=MASTER_ADDR,
    backend=BACKEND,
    rank=WORLD_RANK,
    world_size=WORLD_SIZE,
    timeout=timedelta(seconds=20),
)
print(f"Finished Init Process Group")

# Define the tensor at this rank
t = torch.tensor([WORLD_RANK, WORLD_RANK, WORLD_RANK]).to(device)

print(f"Tensor Before-All Reduce: {t}")
torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
print(f"Tensor After-All Reduce: {t}")
