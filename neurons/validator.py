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


import os
import time
from typing import Optional

os.environ["NEST_ASYNCIO"] = "0"
import math
import threading

import bitsandbytes
import bittensor as bt
from bitarray import bitarray
from bitsandbytes.cextension import lib
from hivemind.compression import deserialize_torch_tensor
from hivemind.proto import averaging_pb2
from hivemind.utils import get_logger
from hivemind.utils.asyncio import aiter_with_timeout
from hivemind.utils.streaming import combine_from_streaming

from distributed_training.averaging.avg_handler import AveragingHandler
from distributed_training.base.validator import BaseValidatorNeuron
from distributed_training.data.dataset import DataLoader
from distributed_training.utils.chain import UIDIterator, log_peerid_to_chain
from distributed_training.utils.misc import (
    AsyncDendritePool,
    get_bandwidth,
    init_dht,
    load_wandb,
    setup_logging,
)
from distributed_training.utils.progress_tracker import (
    GlobalTrainingProgress,
    LocalTrainingProgress,
)
from distributed_training.utils.state_loader import (
    ModelLoadingManager,
    load_model_optimizer_gradient_averager,
    load_state_from_peer,
)
from distributed_training.utils.uids import map_uid_to_peerid
from distributed_training.validator import forward

# Add lamb to bnb str2optimizer8bit_blockwise
bitsandbytes.functional.str2optimizer8bit_blockwise
bitsandbytes.functional.str2optimizer8bit_blockwise["lamb"] = (
    lib.cadam_8bit_blockwise_grad_fp32,
    lib.cadam_8bit_blockwise_grad_fp16,
    lib.cadam_8bit_blockwise_grad_bf16,
)

hivemind_logger = get_logger(__name__)


class Validator(BaseValidatorNeuron):
    def __init__(self, config=None):
        super(Validator, self).__init__(config=config)

        # Initialize class variables
        self.train_timeout = 120
        self.allreduce_timeout = 540
        self.load_state_timeout = 180
        self.model_upload_retry_limit = 3
        self.model_upload_retry_delay = 10
        self.maximum_steps = 306 * 4  # 10_000_000_000/(32000*1024)
        self.warmup_steps = 62  # 306 / 5
        self.learning_rate_maximum = 0.0025
        self.weight_decay = 0.1
        self.num_inner_steps = 500
        self.offload_optimizer = True
        self.failed_is_alive_counter_threshold = 10

        # Initialize components
        self._init_basic_components()
        self._init_model_components()
        self._init_network_components()
        self._init_uid_components()

    def _init_basic_components(self):
        """Initialize basic validator components"""

        # Logging setup
        setup_logging(config=self.config)

        bt.logging.debug("load_state()")
        self.load_state()

        # Init Dendrite Pool
        self.dendrite_pool = AsyncDendritePool(
            wallet=self.wallet, metagraph=self.metagraph
        )

        # Init DHT
        init_dht(self)

        # Init progress tracking
        self.local_progress = LocalTrainingProgress(
            peer_id=self.dht.peer_id.to_bytes(),
            epoch=0,
            samples_accumulated=0,
            samples_per_second=0.0,
            time=0.0,
            client_mode=False,
        )
        self.global_progress = GlobalTrainingProgress(epoch=0, samples_accumulated=0)
        # update_global_tracker_state(self)
        self.global_progress.epoch = 10
        self.local_progress.epoch = self.global_progress  # TODO Fix this

        # Init Wandb
        if not self.config.neuron.dont_wandb_log:
            self.wandb = load_wandb(
                self, self.config, self.wallet, "validator", str(self.dht.peer_id)
            )

        # Init Dataset
        dataset_length = DataLoader.max_rows
        self.dataset_indices = bitarray(dataset_length)

        # Init Device
        self.device = self.config.neuron.device

    def _init_model_components(self):
        """Initialize model and training components"""
        # Init learning rate and loss tracking
        self.learning_rate = self.get_learning_rate()
        self.average_loss = None

        # Init Model, Optimizer & Gradient Averager
        load_model_optimizer_gradient_averager(self, self.global_progress.epoch)

        # Select test layers
        self.test_layer_indices = [
            i
            for i, layer in enumerate(self.model.parameters())
            if len(layer.size()) == 1
        ]

        # Init model loading manager
        self.model_loading_manager = ModelLoadingManager()

        # Load state if needed
        if self.local_progress.epoch < self.global_progress.epoch:
            load_state_from_peer(self, epoch=self.global_progress.epoch)

        # Initialize AveragingHandler for allreduce
        self.avg_handler = AveragingHandler(
            self.model, self.grad_averager, self.state_averager
        )

    def _init_network_components(self):
        """Initialize network and P2P components"""

        # Log PeerID to chain
        bt.logging.info("Logging PeerID to chain")
        log_peerid_to_chain(self)

    def _init_uid_components(self):
        """Initialize UID related components"""
        # Set UIDs
        self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        self.master_uid = self.metagraph.hotkeys.index(
            self.config.neuron.master_ss58_address,
        )

        # Init UID mappings
        self.uid_metadata_tracker = {
            uid: {
                "peer_id": None,
                "model_huggingface_id": None,
                "last_updated_block": None,
            }
            for uid in self.metagraph.uids.tolist()
        }

        # Init UID to PeerID mapping
        self.stop_event = threading.Event()
        map_uid_to_peerid(self)

        # Update PeerID list
        # update_run_peerid_list(self)

        # Init UID is_alive counter
        self.failed_is_alive_counter = {uid: 0 for uid in self.metagraph.uids.tolist()}

        # Init last_allreduce_block to current block # TODO needs to be set properly for newcomers
        self.last_allreduce_block = self.block

    def update_local_tracker_state(self, rewards, responses):
        for reward, response in zip(rewards, responses[0]):
            if (reward != 0) and (response.dataset_indices is not None):
                self.local_progress.samples_accumulated += len(response.dataset_indices)
            else:
                continue

    def get_learning_rate(self):
        learning_rate_minimum = self.learning_rate_maximum * 0.1
        # 1) linear warmup for warmup_steps
        if self.global_progress.epoch < self.warmup_steps:
            return (
                self.learning_rate_maximum
                * (self.global_progress.epoch + 1)
                / self.warmup_steps
            )
        # 2) if epoch > lr_decay_iters, return learning_rate_minimum
        if self.global_progress.epoch > self.maximum_steps:
            return learning_rate_minimum
        # 3) if in between, use cosine decay down to min learning rate
        decay_ratio = (self.global_progress.epoch - self.warmup_steps) / (
            self.maximum_steps - self.warmup_steps
        )
        assert 0 <= decay_ratio <= 1
        # coeff starts at 1 and goes to 0
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return (learning_rate_minimum + coeff) * (
            self.learning_rate_maximum - learning_rate_minimum
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
        metadata = None
        hivemind_logger.info(f"Downloading parameters from peer {peer}")
        try:
            stub = self.grad_averager.get_stub(
                self._p2p,
                peer,
                namespace=self.grad_averager.matchmaking_kwargs["prefix"],
            )
            stream = await stub.rpc_download_state_partial(
                averaging_pb2.DownloadRequest()
            )
            current_tensor_parts, tensors = [], []

            # TODO merge this with hivemind.compression.deserialize_tensor_stream
            async for message in aiter_with_timeout(stream, timeout=timeout):
                if message.metadata:
                    metadata = self.grad_averager.serializer.loads(message.metadata)
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
                hivemind_logger.exception(f"Peer {peer} did not send its state")
                return

            hivemind_logger.info(f"Finished downloading state from {peer}")
            return metadata, tensors
        except Exception as e:
            hivemind_logger.exception(
                f"Failed to download state from {peer} - {repr(e)}"
            )
            return None, None

    async def forward(self):
        return await forward(self)


# The main function parses the configuration and runs the validator.
if __name__ == "__main__":
    with Validator() as validator:
        while True:
            time.sleep(5)
