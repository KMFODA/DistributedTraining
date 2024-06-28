import torch
# from bitsandbytes.optim import LAMB
from transformers import AutoModelForCausalLM, AutoConfig, AutoModelForMaskedLM, DataCollatorForLanguageModeling, AutoTokenizer
from datasets import load_dataset
from torch_optimizer import Lamb
from torch.utils.data import DataLoader
from itertools import chain

# Hyperparameters
optimizer_name = "LAMB"
global_target_batch_size = 512
learning_rate = 5 / ((2^3)*10^3)
warmup_ratio = 1/320

batch_size = 1    
max_seq_length = 512 # Paper indicates first 9/10 epochs have a sequence lenght of 128. Last 1/10 have a sequence length of 512.
number_of_epochs = 10
gradient_accumilation_steps = global_target_batch_size // (batch_size)
mlm_probability = 0.15

from datasets import concatenate_datasets, load_dataset

bookcorpus = load_dataset("bookcorpus", split="train")
wiki = load_dataset("wikipedia", "20220301.en", split="train")
wiki = wiki.remove_columns([col for col in wiki.column_names if col != "text"])  # only keep the 'text' column

assert bookcorpus.features.type == wiki.features.type
raw_datasets = concatenate_datasets([bookcorpus, wiki])

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

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-large-uncased")

column_names = raw_datasets.column_names
text_column_name = "text" if "text" in column_names else column_names[0]

# Otherwise, we tokenize every text, then concatenate them together before splitting them in smaller parts.
# We use `return_special_tokens_mask=True` because DataCollatorForLanguageModeling (see below) is more
# efficient when it receives the `special_tokens_mask`.
def tokenize_function(examples):
    return tokenizer(examples[text_column_name], return_special_tokens_mask=True)

tokenized_datasets = raw_datasets.map(
    tokenize_function,
    batched=True,
    num_proc=None,
    remove_columns=column_names,
    load_from_cache_file=False,
    desc="Running tokenizer on every text in dataset",
)

# Main data processing function that will concatenate all texts from our dataset and generate chunks of
# max_seq_length.
def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, and if the total_length < max_seq_length  we exclude this batch and return an empty dict.
    # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
    total_length = (total_length // max_seq_length) * max_seq_length
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
        for k, t in concatenated_examples.items()
    }
    return result

# Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a
# remainder for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value
# might be slower to preprocess.
#
# To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
# https://huggingface.co/docs/datasets/process#map

tokenized_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    num_proc=None,
    load_from_cache_file=False,
    desc=f"Grouping texts in chunks of {max_seq_length}",
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=mlm_probability)

# DataLoaders creation:
train_dataloader = DataLoader(
    tokenized_datasets=['text'], shuffle=True, collate_fn=data_collator, batch_size=batch_size
)
breakpoint()
total_steps = len(train_dataloader['text']) * number_of_epochs
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