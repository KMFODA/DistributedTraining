import torch
# from bitsandbytes.optim import LAMB
from transformers import AutoModelForCausalLM, AutoConfig, AutoModelForMaskedLM, DataCollatorForLanguageModeling, AutoTokenizer
from datasets import load_dataset
from torch_optimizer import Lamb

# Hyperparameters
optimizer_name = "LAMB"
global_target_batch_size = 512
learning_rate = 5 / ((2^3)*10^3)
warmup_ratio = 1/320

BATCH_SIZE = 1    
SEQ_LENGTH = 512 # Paper indicates first 9/10 epochs have a sequence lenght of 128. Last 1/10 have a sequence length of 512.
number_of_epochs = 10
gradient_accumilation_steps = global_target_batch_size // (BATCH_SIZE)
mlm_probability = 0.15

from datasets import concatenate_datasets, load_dataset

bookcorpus = load_dataset("bookcorpus", split="train")
wiki = load_dataset("wikipedia", "20220301.en", split="train")
wiki = wiki.remove_columns([col for col in wiki.column_names if col != "text"])  # only keep the 'text' column

assert bookcorpus.features.type == wiki.features.type
dataset = concatenate_datasets([bookcorpus, wiki])
breakpoint()

# Load the model
config = AutoConfig.from_pretrained("google-bert/bert-large-uncased")
model = AutoModelForMaskedLM.from_config(config)
# model.to("cuda:0")

# Set up the optimizer
if optimizer_name == "AdamW":
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
elif optimizer_name == "LAMB":
    # optimizer = LAMB(model.parameters(), lr=learning_rate)
    optimizer = Lamb(model.parameters(), lr=learning_rate)
else:
    raise ValueError("Unsupported optimizer")

from torch.utils.data import DataLoader
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-large-uncased")
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=mlm_probability)

# DataLoaders creation:
train_dataloader = DataLoader(
    dataset['text'], shuffle=True, collate_fn=data_collator, batch_size=BATCH_SIZE
)

total_steps = len(dataset['train']) * number_of_epochs
warmup_steps = int(warmup_ratio * total_steps)
local_samples = 0
current_step = 0
for epoch in range(number_of_epochs):
    for step, batch in enumerate(train_dataloader):
    
        # inputs = torch.tensor(batch['input_ids']).to("cuda")

        # Forward pass
        # outputs = model(input_ids=inputs, labels=inputs)
        breakpoint()
        # outputs = model(input_ids=batch['input_ids'].to("cuda"), labels=batch['labels'].to("cuda"))
        outputs = model(**batch)

        # Backward pass
        loss = outputs.loss
        scaled_loss = loss / gradient_accumilation_steps
        scaled_loss.backward()
        print(f"Current epoch {epoch+1}, Batch {step+1}, Loss: {loss.item()}")
        
        # Update local_samples
        local_samples += BATCH_SIZE
        current_step += BATCH_SIZE
            
        if current_step % gradient_accumilation_steps == 0:

            print(f"Running Opt Step")
            optimizer.step()
            optimizer.zero_grad()  # Reset gradients after each step
            local_samples = 0