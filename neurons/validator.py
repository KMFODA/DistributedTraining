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
import time
from ipaddress import ip_address

import bittensor as bt
import hivemind
from transformers import AutoModelForCausalLM
import torch
import wandb

from template.base.validator import BaseValidatorNeuron
from template.utils.misc import AsyncDendritePool, load_wandb, setup_logging, DTGradientAverager, init_dht
from template.validator import forward
from bitarray import bitarray

import asyncio
from typing import Optional
from hivemind.utils import get_logger
from hivemind.proto import averaging_pb2
from hivemind.utils.asyncio import aiter_with_timeout
from hivemind.optim.progress_tracker import ProgressTracker

from hivemind.p2p import PeerID
from hivemind.compression import deserialize_torch_tensor
from hivemind.utils.streaming import combine_from_streaming
from hivemind.utils.timed_storage import ValueWithExpiration
logger = get_logger(__name__)

class Validator(BaseValidatorNeuron):
    def __init__(self, config=None):
        super(Validator, self).__init__(config=config)

        bt.logging.info("load_state()")
        self.load_state()

        # Init DHT
        init_dht(self)

        # Init Wandb
        if not self.config.neuron.dont_wandb_log:
            self.wandb = load_wandb(self.config, self.wallet, "validator", str(self.dht.peer_id))

        # Init Dendrite Pool
        self.dendrite_pool = AsyncDendritePool(
            wallet=self.wallet, metagraph=self.metagraph
        )

        # # Init Dataset
        dataset_length = 968000015
        self.dataset_indices = bitarray(dataset_length)

        # Init Device, Model & Tokenizer
        self.device = self.config.neuron.device
        self.model = AutoModelForCausalLM.from_pretrained(self.config.neuron.model_name)
        self.model.to(self.device)

        # For simplicity only pick layers with a dim of 1
        self.test_layer_indices = [i for i, layer in enumerate(self.model.parameters()) if len(layer.size()) == 1]

        # Init UID
        self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)

        # Init Optimizer
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=self.config.neuron.lr)

        self.grad_averager = DTGradientAverager(
            self.model.parameters(),
            dht=self.dht,
            prefix=f"{self.config.neuron.run_id}_grad_averager",
            compression=hivemind.Uniform8BitQuantization(),
            # reuse_grad_buffers=True,
            accumulate_grads_on=torch.device("cuda"),
            start = True
        )

        self.tracker = ProgressTracker(
            dht=self.dht, 
            prefix=f"{self.config.neuron.run_id}_progress", 
            target_batch_size=self.config.neuron.global_batch_size_train,
            start=True
        )

        self.step_scheduled = False

        self.loop = asyncio.new_event_loop()
        self._p2p = self.loop.run_until_complete(self.dht.replicate_p2p())
        self.peer_list = self.loop.run_until_complete(self._p2p.list_peers())
        self.peer_id = self.dht.peer_id
        self.get_stub = self.grad_averager.get_stub
        self.serializer = self.grad_averager.serializer

        # Start Main Validation Loop
        bt.logging.info("Starting validator loop.")

    def get_validator_info(self):
        return {
            "block": self.metagraph.block.item(),
            "stake": self.metagraph.stake[self.uid],
            "rank": self.metagraph.ranks[self.uid],
            "vtrust": self.metagraph.validator_trust[self.uid],
            "dividends": self.metagraph.dividends[self.uid],
            "emissions": self.metagraph.emission[self.uid],
        }

    async def map_uid_to_peerid(self, uid):
        miner_ip = self.metagraph.axons[uid].ip

        # Get all peers connected to our DHT and their ips
        peer_list_dht = await self._p2p.list_peers()
        peer_list_dht_addrs = [str(peer.addrs[0]).split('/ip4/')[1].split('/')[0] for peer in peer_list_dht]

        # Get only peers connected to the current run id
        prefix = self.grad_averager.matchmaking_kwargs['prefix']
        metadata, _ = self.dht.get(f"{prefix}.all_averagers", latest=True) or ({}, None)

        if metadata is None:
            return None
        peer_list_run = [str(PeerID(peer_id)) for peer_id, info in metadata.items() if isinstance(info, ValueWithExpiration) and isinstance(info.value, (float, int))]

        # If the UIDs ip address is not in the list of peer addrs then it is not connected to our DHT
        if miner_ip not in peer_list_dht_addrs:
            return None
        else:
            peer_id = peer_list_dht[peer_list_dht_addrs.index(miner_ip)].peer_id

        # If peer_id is not in the list of peer ids for our run then it is not connected to our run ID
        if str(peer_id) not in peer_list_run:
            return None
        else:
            return peer_id

    async def load_state_from_miner(self, peer, timeout: Optional[float] = None):
        if timeout is not None:
            timeout = self.next_chunk_timeout if self.next_chunk_timeout is not None else self.request_timeout

        metadata = None
        logger.info(f"Downloading parameters from peer {peer}")
        try:
            stub = self.get_stub(self._p2p, peer, namespace=self.grad_averager.matchmaking_kwargs['prefix'])
            stream = await stub.rpc_download_state(averaging_pb2.DownloadRequest())
            current_tensor_parts, tensors = [], []

            # TODO merge this with hivemind.compression.deserialize_tensor_stream
            async for message in aiter_with_timeout(stream, timeout=timeout):
                if message.metadata:
                    metadata = self.serializer.loads(message.metadata)
                if message.tensor_part.dtype and current_tensor_parts:
                    # tensor_part.dtype indicates the start of the new tensor, so we should wrap up this one
                    tensors.append(deserialize_torch_tensor(combine_from_streaming(current_tensor_parts)))
                    current_tensor_parts = []
                current_tensor_parts.append(message.tensor_part)
            if current_tensor_parts:
                tensors.append(deserialize_torch_tensor(combine_from_streaming(current_tensor_parts)))

            if not metadata:
                logger.exception(f"Peer {peer} did not send its state")
                return

            logger.info(f"Finished downloading state from {peer}")
            # future.set_result((metadata, tensors))
            return metadata, tensors
        except Exception as e:
            logger.exception(f"Failed to download state from {peer} - {repr(e)}")
            return None, None

    async def forward(self):
        return await forward(self)


# # The main function parses the configuration and runs the validator.
if __name__ == "__main__":
    setup_logging()
    with Validator() as validator:
        while True:
            time.sleep(5)
