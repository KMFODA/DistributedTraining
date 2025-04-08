import asyncio
from transformers import AutoTokenizer
import torch
import bittensor as bt

from tplr.dataset import DatasetLoader
from tplr.r2_dataset import R2DatasetLoader

class ComparisonTest:
    def __init__(self):
        # Common configuration
        self.uid = 42
        self.tokenizer = AutoTokenizer.from_pretrained("distilgpt2", use_fast=False)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Test configuration - using smaller numbers for quick testing
        self.batch_size = 2
        self.sequence_length = 32
        self.n_pages = 2
        self.offset = 0

    async def run_hf_dataloader(self):
        """Run the HF dataloader and return the first batch"""
        pages = await DatasetLoader.next_pages(
            offset=self.offset,
            n_pages=self.n_pages,
            seed=self.uid,
        )
        print('hf pages', pages)
        
        dataset = await DatasetLoader.create(
            batch_size=self.batch_size,
            sequence_length=self.sequence_length,
            pages_info=pages,
            tokenizer=self.tokenizer,
        )
        
        # Get first batch
        batch, labels = next(iter(dataset))
        return batch, labels

    async def run_r2_dataloader(self):
        """Run the R2 dataloader and return the first batch"""
        pages = await R2DatasetLoader.next_pages(
            offset=self.offset,
            n_pages=self.n_pages,
            seed=self.uid
        )
        print('r2 pages', pages)
        
        loader = await R2DatasetLoader.create(
            batch_size=self.batch_size,
            sequence_length=self.sequence_length,
            pages_info=pages,
            tokenizer=self.tokenizer
        )
        
        # Get first batch
        for i, batch in enumerate(loader):
            if i == 0:
                input_ids = torch.tensor(batch, dtype=torch.long)
                labels = input_ids[:, 1:]  # Shift to create labels
                input_ids = input_ids[:, :-1]  # Adjust input_ids to match HF format
                return input_ids, labels
            break

    async def compare_outputs(self):
        """Run both dataloaders and compare their outputs"""
        hf_batch, hf_labels = await self.run_hf_dataloader()
        r2_batch, r2_labels = await self.run_r2_dataloader()
        
        print("\n=== HF Dataloader Output ===")
        print(f"Batch shape: {hf_batch.shape}")
        print(f"First 5 tokens: {hf_batch[0, :5].tolist()}")
        print(f"Labels shape: {hf_labels.shape}")
        print(f"First 5 labels: {hf_labels[0, :5].tolist()}")
        
        print("\n=== R2 Dataloader Output ===")
        print(f"Batch shape: {r2_batch.shape}")
        print(f"First 5 tokens: {r2_batch[0, :5].tolist()}")
        print(f"Labels shape: {r2_labels.shape}")
        print(f"First 5 labels: {r2_labels[0, :5].tolist()}")
        
        # Compare shapes
        assert hf_batch.shape == r2_batch.shape, f"Batch shape mismatch: {hf_batch.shape} vs {r2_batch.shape}"
        assert hf_labels.shape == r2_labels.shape, f"Labels shape mismatch: {hf_labels.shape} vs {r2_labels.shape}"
        
        # Compare content
        assert torch.equal(hf_batch, r2_batch), "Batch content differs"
        assert torch.equal(hf_labels, r2_labels), "Labels content differs"
        
        print("\n✅ Both dataloaders produce identical output!")

async def main():
    tester = ComparisonTest()
    await tester.compare_outputs()

if __name__ == "__main__":
    asyncio.run(main())