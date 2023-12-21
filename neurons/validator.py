# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 KMFODA

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import asyncio
import hivemind
import time

import bittensor as bt

import torch
from datasets import load_dataset
from hivemind.optim.state_averager import TrainingStateAverager
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from functools import partial
from template.utils.misc import AsyncDendritePool, load_wandb
from template.utils.uids import get_random_uids
from template.validator.validator_core import DatasetStateSingelton, ModelSingleton, upload_checkpoint
from template.validator import forward
from template.base.validator import BaseValidatorNeuron

class Validator(BaseValidatorNeuron):

    def __init__(self, config=None):
        super(Validator, self).__init__(config=config)

        bt.logging.info("load_state()")
        self.load_state()
        
        
        self.dataset_common_state = None

        # Init Dendrite Pool
        self.dendrite_pool = AsyncDendritePool( wallet = self.wallet, metagraph = self.metagraph )

        # Init Loss
        self.previous_loss = 0
        self.latest_upload = 0
        self.latest_weight_update = 0
        self.step = 0

        # Init device
        self.device = self.config.neuron.device

        # Init Model
        self.model = ModelSingleton.get_instance(self.config.neuron.model_name, self.config.neuron.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.neuron.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # DHT initialization
        self.dht = hivemind.DHT(initial_peers=[self.config.neuron.initial_peers], start=True)
        
        # Init State Averager
        self.state_averager = TrainingStateAverager(
            dht=self.dht, 
            optimizer=partial(torch.optim.AdamW, lr=self.config.neuron.lr),
            scheduler=partial(torch.optim.lr_scheduler.LambdaLR, lr_lambda=lambda t: 1.0 / max(1, t)),
            params=self.model.parameters(),
            allow_state_sharing=False,
            start=True,
            prefix=f"{self.config.neuron.run_id}_state_averager", 
            # state_compression=hivemind.Float16Compression(),
            # bandwidth=optimizer_args.bandwidth,
            # client_mode=optimizer_args.client_mode,
            # **asdict(averager_args),
        )

        # Start Main Validation Loop
        bt.logging.info("Starting validator loop.")
        
    async def async_init(self):
        
        # Init Dataset
        self.dataset = load_dataset(self.config.neuron.dataset_name, 'wikitext-2-v1', split='train')
        self.dataset_indices = [i for i in range(0, len(self.dataset))]
        
        # Asynchronous DatasetStateSingleton initialization 
        self.dataset_common_state = DatasetStateSingelton(self.dht, self.dataset_indices, self.config.neuron.run_id)
        await self.dataset_common_state.initialize_async()
        bt.logging.info("Finished async intiatlization.")
        
    # Define encoding function
    def encode(self, examples):
        return self.tokenizer(examples['text'], truncation=True, max_length=512, padding='max_length', return_tensors='pt')

    async def forward(self):
        return await forward(self)


# Async main function
async def main():
    async with Validator() as validator:
        while True:
            await asyncio.sleep(5)

if __name__ == "__main__":
    asyncio.run(main())