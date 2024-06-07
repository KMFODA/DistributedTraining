import argparse
import logging
import os
import random

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from datasets import load_dataset

from transformers import AutoModelForCausalLM, AutoTokenizer

# from template.data.dataset import SubsetFalconLoader

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '172.18.0.1'
    os.environ['MASTER_PORT'] = '50464'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size, args):
    setup(rank, world_size)

    logfile = f"logfile_{rank}.log"
    if os.path.exists(logfile):
        os.remove(logfile)

    logging.basicConfig(
        filename=logfile,
        filemode='a',
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.DEBUG
    )

    logger = logging.getLogger()
    logger.info(f"Running training on rank {rank}.")

    model = AutoModelForCausalLM.from_pretrained("kmfoda/gpt2-500m").to(rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    opt = torch.optim.AdamW(model.parameters(), lr=0.001)
    global_target_batch_size = 600
    local_batch_size = 1
    accumulation_steps = global_target_batch_size // (local_batch_size * world_size)
    local_samples_processed = 0
    
    # Load dataset
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2", use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    
    dataset = load_dataset('HuggingFaceFW/fineweb', 'sample-10BT', split='train')
    dataset = dataset.map(lambda e: tokenizer(e['text'], truncation=True, padding='max_length', max_length=1024), batched=True, num_proc=mp.cpu_count())
    dataset.set_format(type='torch', columns=['input_ids'])

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=local_batch_size, sampler=sampler)

    while True:
        logger.info("Starting training..")
        # dataloader = SubsetFalconLoader(
        #     batch_size=local_batch_size,
        #     sequence_length=1024,
        #     rows=random.choices(range(0, 519_000_000), k=1000),
        # )

        model.zero_grad()
        for i, batch in enumerate(dataloader):
            inputs = batch.to(rank)

            outputs = model(input_ids=inputs, labels=inputs)
            loss = outputs.loss
            scaled_loss = loss / accumulation_steps

            logger.info(f"Rank {rank} - Loss: {loss.item()}")

            scaled_loss.backward()
            local_samples_processed += local_batch_size

            # All-reduce to get the global number of samples processed
            tensor = torch.tensor(local_samples_processed, dtype=torch.float32, device=rank)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            global_samples_processed = tensor.item()

            logger.info(f"Rank {rank} - Global samples processed: {global_samples_processed}")

            if global_samples_processed >= global_target_batch_size:
                logger.info(f"Rank {rank} - Performing optimizer step")
                opt.step()
                model.zero_grad()
                local_samples_processed = 0

    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed training script with PyTorch.")
    parser.add_argument("--world_size", type=int, default=2, help="Number of GPUs to use for training.")
    args = parser.parse_args()

    world_size = args.world_size
    mp.spawn(train, args=(world_size, args), nprocs=world_size, join=True)
