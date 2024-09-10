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
import math
import time
from typing import Optional

import bittensor as bt
import hivemind
import torch
from bitarray import bitarray
from bitsandbytes.optim import LAMB
from hivemind.compression import deserialize_torch_tensor
from hivemind.proto import averaging_pb2
from hivemind.utils import get_logger
from hivemind.utils.asyncio import aiter_with_timeout
from hivemind.utils.streaming import combine_from_streaming
from transformers import AutoModelForCausalLM

from distributed_training import __spec_version__, __version__
from distributed_training.base.validator import BaseValidatorNeuron
from distributed_training.data.dataset import DataLoader
from distributed_training.utils.gradient_averager import (
    DTGradientAverager,
)
from distributed_training.utils.state_loader import (
    load_state_from_peer,
)

from distributed_training.utils.progress_tracker import (
    GlobalTrainingProgress, LocalTrainingProgress, update_global_tracker_state)
from distributed_training.utils.state_loader import load_state_from_peer
from distributed_training.utils.uids import map_uid_to_peerid
from distributed_training.validator import forward
from distributed_training.utils.misc import (AsyncDendritePool, init_dht,
                                             load_wandb, setup_logging, warmup)

logger = get_logger(__name__)


class Validator(BaseValidatorNeuron):
    def __init__(self, config=None):
        super(Validator, self).__init__(config=config)

        # Init Logging
        setup_logging(
            network=self.config.subtensor.network,
            netuid=self.config.netuid,
            hotkey=self.wallet.hotkey.ss58_address,
            version=__version__,
            spec_version=__spec_version__,
            run_id=None,
            ip=self.config.axon.ip
            if self.config.axon.ip != "[::]"
            else bt.utils.networking.get_external_ip(),
            port=self.config.axon.port,
            uid=self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address),
            neuron_type="validator",
        )

        bt.logging.info("load_state()")
        self.load_state()

        # Init Dendrite Pool
        self.dendrite_pool = AsyncDendritePool(
            wallet=self.wallet, metagraph=self.metagraph
        )

        # Init DHT
        init_dht(self)

        # Init Local & Global Progress
        self.local_progress = LocalTrainingProgress(
            peer_id=self.dht.peer_id.to_bytes(),
            epoch=0,
            samples_accumulated=0,
            samples_per_second=0.0,
            time=0.0,
            client_mode=False,
        )
        self.global_progress = GlobalTrainingProgress(epoch=0, samples_accumulated=0)
        update_global_tracker_state(self)

        # Init Wandb
        if not self.config.neuron.dont_wandb_log:
            self.wandb = load_wandb(
                self, self.config, self.wallet, "validator", str(self.dht.peer_id)
            )

        # Track AllReduce operation
        self.allreduce_history = []

        # Init Dataset
        dataset_length = DataLoader.max_pages
        self.dataset_indices = bitarray(dataset_length)

        # Init Device & Model
        self.device = self.config.neuron.device
        if self.global_progress.epoch is None:
            bt.logging.error(
                f"Model Tag Is None. Make Sure You Are Using The Correct Model Name"
            )
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                self.config.neuron.model_name, revision=str(self.global_progress.epoch)
            )
            if self.global_progress.epoch
            else AutoModelForCausalLM.from_pretrained(self.config.neuron.model_name)
        )

        # Move the model to the appropriate device
        self.model.to(self.device)

        # For simplicity only pick layers with a dim of 1
        self.test_layer_indices = [
            i
            for i, layer in enumerate(self.model.parameters())
            if len(layer.size()) == 1
        ]

        # Init UID
        self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)

        # Init Optimizer
        self.opt = LAMB(self.model.parameters(), lr=self.config.neuron.learning_rate)

        # Init Gradient Averager
        self.grad_averager = DTGradientAverager(
            self.model.parameters(),
            dht=self.dht,
            prefix=f"{self.config.neuron.run_id}_grad_averager",
            compression=hivemind.Uniform8BitQuantization(),
            accumulate_grads_on=torch.device("cuda"),
            start=True,
            next_chunk_timeout=30.0,
            barrier_timeout=40.0,
        )

        self.loop = asyncio.new_event_loop()
        self._p2p = self.loop.run_until_complete(self.dht.replicate_p2p())
        self.peer_list = self.loop.run_until_complete(self._p2p.list_peers())

        self.peer_id = self.dht.peer_id
        self.get_stub = self.grad_averager.get_stub
        self.serializer = self.grad_averager.serializer

        # Create mapping between uids to peerids
        self.uids_to_peerids = self.loop.run_until_complete(
            map_uid_to_peerid(self, range(0, self.metagraph.n))
        )
        max_retries = 3
        retries = 0
        while all(value is None for value in self.uids_to_peerids.values()) and (
            retries >= max_retries
        ):
            for retries in range(0, max_retries):
                self.uids_to_peerids = self.loop.run_until_complete(
                    map_uid_to_peerid(self, range(0, self.metagraph.n))
                )
                time.sleep(1)
        self.uids_to_peerids[self.uid] = self.dht.peer_id
        bt.logging.info(f"UID To PeerID Mapping: {self.uids_to_peerids}")

        # Init All Reduce Variables
        self.all_reduce_timeout = 300
        self.step_scheduled = False
        self.model_upload_retry_limit = 3
        self.model_upload_retry_delay = 10
        self.maximum_steps = 306 * 4

        # Load state from peers if validator is not on latest global epoch
        if self.global_progress.epoch is not None:
            if self.local_progress.epoch < self.global_progress.epoch:
                load_state_from_peer(self, epoch=self.global_progress.epoch)

        # Start Main Validation Loop
        bt.logging.info("Starting validator loop.")

    def update_local_tracker_state(self, rewards, responses):
        for reward, response in zip(rewards, responses[0]):
            if reward != 0:
                self.local_progress.samples_accumulated += len(response.dataset_indices)
            else:
                continue

    def get_learning_rate(self):
        if self.global_progress.epoch is None:
            return self.config.neuron.learning_rate

        learning_rate_minimum = self.config.neuron.learning_rate * 0.1
        # 1) linear warmup for warmup_steps
        if self.global_progress.epoch < self.config.neuron.warmup_steps:
            return (
                self.config.neuron.learning_rate
                * (self.global_progress.epoch + 1)
                / self.config.neuron.warmup_steps
            )
        # 2) if epoch > lr_decay_iters, return learning_rate_minimum
        if self.global_progress.epoch > self.maximum_steps:
            return learning_rate_minimum
        # 3) if in between, use cosine decay down to min learning rate
        decay_ratio = (self.global_progress.epoch - self.config.neuron.warmup_steps) / (
            self.maximum_steps - self.config.neuron.warmup_steps
        )
        assert 0 <= decay_ratio <= 1
        # coeff starts at 1 and goes to 0
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return (learning_rate_minimum + coeff) * (
            self.config.neuron.learning_rate - learning_rate_minimum
        )

    def get_validator_info(self):
        return {
            "block": self.metagraph.block.item(),
            "stake": self.metagraph.stake[self.uid],
            "rank": self.metagraph.ranks[self.uid],
            "vtrust": self.metagraph.validator_trust[self.uid],
            "dividends": self.metagraph.dividends[self.uid],
            "emissions": self.metagraph.emission[self.uid],
        }

    async def load_state_from_miner(self, peer, timeout: Optional[float] = None):
        if timeout is not None:
            timeout = (
                self.next_chunk_timeout
                if self.next_chunk_timeout is not None
                else self.request_timeout
            )

        metadata = None
        logger.info(f"Downloading parameters from peer {peer}")
        try:
            stub = self.get_stub(
                self._p2p,
                peer,
                namespace=self.grad_averager.matchmaking_kwargs["prefix"],
            )
            stream = await stub.rpc_download_state(averaging_pb2.DownloadRequest())
            current_tensor_parts, tensors = [], []

            # TODO merge this with hivemind.compression.deserialize_tensor_stream
            async for message in aiter_with_timeout(stream, timeout=timeout):
                if message.metadata:
                    metadata = self.serializer.loads(message.metadata)
                if message.tensor_part.dtype and current_tensor_parts:
                    # tensor_part.dtype indicates the start of the new tensor, so we should wrap up this one
                    tensors.append(
                        deserialize_torch_tensor(
                            combine_from_streaming(current_tensor_parts)
                        )
                    )
                    current_tensor_parts = []
                current_tensor_parts.append(message.tensor_part)
            if current_tensor_parts:
                tensors.append(
                    deserialize_torch_tensor(
                        combine_from_streaming(current_tensor_parts)
                    )
                )

            if not metadata:
                logger.exception(f"Peer {peer} did not send its state")
                return

            logger.info(f"Finished downloading state from {peer}")
            return metadata, tensors
        except Exception as e:
            logger.exception(f"Failed to download state from {peer} - {repr(e)}")
            return None, None

    async def forward(self):
        return await forward(self)

    def warmup(
        self,
    ):
        warmup(self)


# The main function parses the configuration and runs the validator.
if __name__ == "__main__":
    with Validator() as validator:
        while True:
            time.sleep(5)
