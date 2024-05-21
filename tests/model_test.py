import os
import time
from functools import partial
from ipaddress import ip_address

import hivemind
import requests
import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator


# Define encoding function
def encode(examples):
    return tokenizer(
        examples["text"], truncation=True, max_length=512, padding="max_length"
    )


# Create dataset and model, same as in the basic tutorial
model = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=torch.float16)
device = "cuda"
# model = model.to(dtype=torch.float16, device=device)
model = model.to(device=device)
opt = torch.optim.AdamW(model.parameters(), lr=0.0001)

tokenizer = AutoTokenizer.from_pretrained("gpt2")
# Add the EOS token as PAD token to ensure our dataloader doesn't throw an error for sequences of unequal length
tokenizer.pad_token = tokenizer.eos_token

# Load dataset
dataset = load_dataset("wikitext", "wikitext-2-v1", split="train")
encoded_dataset = dataset.map(encode, batched=True)

dataloader = DataLoader(
    encoded_dataset, batch_size=20, collate_fn=default_data_collator
)

# pb_bar = tqdm(
#     enumerate(dataloader), total=len(dataloader), dynamic_ncols=True, leave=False
# )

# Train data for one epoch
for step, batch in enumerate(dataloader):
    input_ids = batch["input_ids"].to(device)
    outputs = model(input_ids=input_ids, labels=input_ids)
    loss = outputs.loss
    loss.backward()
    print(loss)
