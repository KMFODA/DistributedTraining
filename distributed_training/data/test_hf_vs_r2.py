import asyncio
from transformers import AutoTokenizer
from datasets import load_dataset  # HuggingFace datasets

async def compare_data_sources():
    # Load data from R2
    r2_pages = await R2DatasetLoader.next_pages(offset=0, n_pages=1, seed=42)
    r2_loader = await R2DatasetLoader.create(
        batch_size=32,
        sequence_length=512,
        pages_info=r2_pages,
        tokenizer=tokenizer
    )
    r2_batch = next(iter(r2_loader))
    
    # Load same data from HuggingFace
    hf_dataset = load_dataset("fineweb-edu", split="train", streaming=True)
    hf_sample = next(iter(hf_dataset.take(1)))  # Get first sample
    
    # Compare
    print("R2 first tokens:", r2_batch[0][:20])  # First 20 tokens of first sample
    print("HF first tokens:", tokenizer(hf_sample["text"])["input_ids"][:20])
    
    # You could add more rigorous comparison here
    # For example checking lengths, token distributions, etc.

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    asyncio.run(compare_data_sources())
