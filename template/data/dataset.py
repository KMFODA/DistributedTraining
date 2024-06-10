# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 const

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

import math
import time
import typing

import bittensor as bt
import requests
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
import torch
from torch.utils.data import IterableDataset
from transformers import AutoTokenizer

model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token


# Modified version of https://github.com/RaoFoundation/pretraining/blob/main/pretrain/dataset.py
class SubsetFalconLoader(IterableDataset):
    max_pages: int = 27_000_000_000

    def __init__(
        self, batch_size, sequence_length, rows: typing.List[int], tokenizer=tokenizer
    ):
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.tokenizer = tokenizer
        self.base_url = "https://datasets-server.huggingface.co/rows"
        self.params = {
            "dataset": "HuggingFaceFW/fineweb",
            "config": "default",
            "split": "train",
        }
        self.rows = rows
        self.buffer = []
        self.retry_limit = 10  # Number of retries
        self.retry_delay = 5  # Seconds to wait between retries
        self.fetch_data_for_page(min(self.rows), len(self.rows))

    def fetch_data_for_page(self, offset, length):
        iterations = math.ceil(length / 100)
        for iteration in range(iterations):
            self.params["offset"] = offset + (iteration * 100)
            self.params["length"] = min(100, length - (iteration * 100))
            attempt = 0
            while attempt < self.retry_limit:
                try:
                    response = requests.get(self.base_url, params=self.params)
                    response.raise_for_status()  # This will raise an HTTPError if the HTTP request returned an unsuccessful status code

                    for row in response.json()["rows"]:
                        content = row["row"]["text"]
                        self.buffer += self.tokenizer(content, truncation=True)[
                            "input_ids"
                        ]
                        self.buffer += [self.tokenizer.eos_token_id]
                    break  # If the request was successful, break out of the retry loop
                except requests.exceptions.RequestException as e:
                    attempt += 1
                    bt.logging.warning(
                        f"Failed to fetch data, retrying. Attempt {attempt}/{self.retry_limit}"
                    )
                    if attempt < self.retry_limit:
                        time.sleep(self.retry_delay)  # Wait before the next retry
                    else:
                        bt.logging.error(
                            "Maximum retry limit reached. Unable to fetch data."
                        )
                        raise

    def __iter__(self):
        while len(self.buffer) >= self.sequence_length * self.batch_size:
            batch = []
            for _ in range(self.batch_size):
                batch.append(torch.tensor(self.buffer[: self.sequence_length]))
                self.buffer = self.buffer[self.sequence_length :]
            yield torch.stack(batch)

    def __next__(self):
        batch = []
        for _ in range(self.batch_size):
            batch.append(torch.tensor(self.buffer[: self.sequence_length]))
            self.buffer = self.buffer[self.sequence_length :]
        return torch.stack(batch)
