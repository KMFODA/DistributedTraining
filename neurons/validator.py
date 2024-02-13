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
import re
import time
from functools import partial
from ipaddress import ip_address

import bittensor as bt
import hivemind
import requests
import torch
import wandb
from datasets import load_dataset
from hivemind.optim.state_averager import TrainingStateAverager
from hivemind.optim.progress_tracker import ProgressTracker
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from template.base.validator import BaseValidatorNeuron
from template.utils.misc import AsyncDendritePool, load_wandb, setup_logging
from template.validator import forward
from template.validator.validator_core import DatasetState
from bitarray import bitarray


import asyncio
import multiprocessing as mp
from typing import Any, Dict, Optional, Sequence, Tuple
from hivemind.utils import MPFuture, get_logger
from hivemind.proto import averaging_pb2
from hivemind.utils.asyncio import (
    achain,
    aiter_with_timeout,
    anext,
    as_aiter,
    azip,
    enter_asynchronously,
    switch_to_uvloop,
)
from hivemind.p2p import P2P, P2PContext, P2PDaemonError, P2PHandlerError, PeerID, ServicerBase
import random
from hivemind.compression import CompressionBase, CompressionInfo, NoCompression, deserialize_torch_tensor
from hivemind.utils.streaming import combine_from_streaming, split_for_streaming
from hivemind.utils.timed_storage import DHTExpiration, ValueWithExpiration, get_dht_time
logger = get_logger(__name__)

class Validator(BaseValidatorNeuron):
    def __init__(self, config=None):
        super(Validator, self).__init__(config=config)

        bt.logging.info("load_state()")
        self.load_state()

        self.init_dht()

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
        # self.dataset_dict = dict()  # Init a dict to use as placeholder DHT

        self.dataset_common_state = DatasetState(
            self.dht, self.dataset_indices, self.config.neuron.run_id
        )
        
        # self.dataset_indices_list_test = self.dataset_common_state.get_dht("dataset_indices_train")
        # if self.dataset_indices_list_test is None:
        #     self.dataset_indices_list_test = self.dataset_common_state.get_dht("dataset_indices_test")
        self.dataset_indices_list_test = (
            self.dataset_common_state.get_dataset_indices_test(
                self.config.neuron.local_batch_size_test * self.config.neuron.local_gradient_accumilation_steps_test
            )
        )

        self.global_step = self.dataset_common_state.get_dht("step")
        if self.global_step is None:
            self.global_step = 0
            # self.dataset_common_state.set_dht("step")

        # Init Loss
        self.previous_loss = self.dataset_common_state.get_dht("loss")
        self.latest_upload = 0
        self.latest_weight_update = 0
        self.step = 0
        self.global_step = self.dataset_common_state.get_dht("step")

        # Init device
        self.device = self.config.neuron.device

        # Init Model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.neuron.model_name
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.neuron.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        # For simplicity only pick layers with a dim of 1
        self.test_layer_indices = [i for i, layer in enumerate(self.model.parameters()) if len(layer.size()) == 1]

        # # Init State Averager
        # self.state_averager = TrainingStateAverager(
        #     dht=self.dht,
        #     optimizer=partial(torch.optim.AdamW, lr=self.config.neuron.lr),
        #     scheduler=None,
        #     params=self.model.parameters(),
        #     allow_state_sharing=False,
        #     start=True,
        #     prefix=f"{self.config.neuron.run_id}_state_averager",
        #     state_compression=hivemind.Uniform8BitQuantization(),
        #     # **asdict(averager_args),
        # )

        # Init UID
        self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)

        opt = torch.optim.AdamW(self.model.parameters(), lr=self.config.neuron.lr)
        self.opt = hivemind.Optimizer(
            dht=self.dht,  # use a DHT that is connected with other peers
            run_id=self.config.neuron.run_id,  # unique identifier of this collaborative run
            scheduler=None,
            batch_size_per_step=self.config.neuron.local_batch_size_train*self.config.neuron.local_gradient_accumilation_steps_train,  # each call to opt.step adds this many samples towards the next epoch
            target_batch_size=self.config.neuron.global_batch_size_train,  # after peers collectively process this many samples, average weights and begin the next epoch
            optimizer=opt,  # wrap the SGD optimizer defined above
            use_local_updates=False,  # perform optimizer steps with local gradients, average parameters in background
            load_state_timeout=240,
            matchmaking_time=15.0,  # when averaging parameters, gather peers in background for up to this many seconds
            averaging_timeout=60.0,  # give up on averaging if not successful in this many seconds
            verbose=False,  # print logs incessently
            grad_compression=hivemind.Uniform8BitQuantization(),
            state_averaging_compression=hivemind.Uniform8BitQuantization(),
        )

        self.loop = asyncio.new_event_loop()
        # self._inner_pipe = self.state_averager._inner_pipe
        # self._outer_pipe = self.state_averager._outer_pipe
        self._inner_pipe, self._outer_pipe = mp.Pipe(duplex=True) 
        self._p2p = self.loop.run_until_complete(self.dht.replicate_p2p())
        self.opt.state_averager.prefix = self.opt.state_averager.matchmaking_kwargs["prefix"]
        self.peer_id = self.dht.peer_id
        self.training_progress_key = f"{self.config.neuron.run_id}_progress"
        self.get_stub = self.opt.state_averager.get_stub
        self.serializer = self.opt.state_averager.serializer

        # TEST
        self.loop.close()
        self.loop = asyncio.new_event_loop()
        self.peer_list = self.loop.run_until_complete(self._p2p.list_peers())

        # breakpoint()
        self.loop.close()
        self.loop = asyncio.new_event_loop()
        import template
        test = self.loop.run_until_complete(self.dendrite(self.metagraph.axons[1], template.protocol.IsAlive(), deserialize=False, timeout=2.3))
        breakpoint()
        # _ = self.loop.run_until_complete(self._load_state_from_peers(self.peer_list[1].peer_id))
        
        # metadata, _expiration = self.dht.get(self.opt.tracker.training_progress_key, latest=True) or (None, -float("inf"))
        # valid_peer_entries = [str(PeerID(peer_state.value['peer_id'])) for peer_state in metadata.values() if peer_state.value is not None]
        # breakpoint()
        # # metadata, _expiration = 
        # breakpoint()

        # breakpoint()
        # future = MPFuture()
        # metadata, _expiration = self.dht.get(self.training_progress_key, latest=True) or (None, -float("inf"))
        # self.loop = asyncio.new_event_loop()
        # self.loop.run_until_complete(self._load_state_from_peers(peer_list[0].peer_id, future))
        # self.loop.close()

        # Get Current Epoch
        self.current_epoch = 1 # Dummy fix need to swithc to self.opt.tracker.global_progress.epoch
        
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
        metadata, _ = self.dht.get(self.opt.tracker.training_progress_key, latest=True) or (None, -float("inf"))
        if metadata is None:
            return None
        peer_list_run = [str(PeerID(peer_state.value['peer_id'])) for peer_state in metadata.values() if peer_state.value is not None]

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
        
    def load_state_from_peers(
        self, peer, wait: bool = True, timeout: Optional[float] = None,
    ) -> Optional[Tuple[Any, Sequence[torch.Tensor]]]:
        """
        Try to download the latest optimizer state one of the existing peer.
        :returns: on success, return a 2-tuple with (metadata, tensors), where

        - metadata is a small object containing metadata (e.g. hyperparameters, scalars, etc)
        - tensors is a sequence of pytorch tensors meant to contain peer's model weights and optimizer statistics

        The exact contents of both metadata and tensors are determined by get_current_state method
        """
        future = MPFuture()
        self._outer_pipe.send(("_load_state_from_peers", [], dict(peer = peer, timeout=timeout, future=future)))
        return future.result(timeout=timeout) if wait else future

    async def load_state_from_miner(self, peer, timeout: Optional[float] = None):
        if timeout is not None:
            timeout = self.next_chunk_timeout if self.next_chunk_timeout is not None else self.request_timeout

        metadata = None
        logger.info(f"Downloading parameters from peer {peer}")
        try:
            stub = self.get_stub(self._p2p, peer, namespace=self.opt.state_averager.prefix)
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

    # Define encoding function
    def encode(self, examples):
        return self.tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors="pt",
        )

    def init_dht(self):
        # Init DHT
        if self.config.dht.use_google_dns:
            request = requests.get("https://api.ipify.org")
            request.raise_for_status()

            address = request.text
            bt.logging.info(f"Received public IP address of this machine: {address}")
            version = ip_address(address).version
            announce_maddrs = [f"/ip{version}/{address}/tcp/{self.config.dht.port}"]
        else:
            version = "4"
            address = self.config.dht.announce_ip
            announce_maddrs = [f"/ip{version}/{address}/tcp/{self.config.dht.port}"]

        # Init list of available DHT addresses from wandb
        api = wandb.Api()
        initial_peers_list = self.config.neuron.initial_peers
        runs = api.runs(
            f"{self.config.neuron.wandb_entity}/{self.config.neuron.wandb_project}"
        )
        for ru in runs:
            if ru.state == "running":
                for peer in ru.config["neuron"]["initial_peers"]:
                    if peer not in initial_peers_list:
                        initial_peers_list.append(peer)

        retries = 0
        while retries <= len(initial_peers_list):
            if retries == len(initial_peers_list):
                raise Exception("Max retries reached, operation failed.")
            try:
                # Init DHT
                self.dht = hivemind.DHT(
                    host_maddrs=[
                        f"/ip4/0.0.0.0/tcp/{self.config.dht.port}",
                        f"/ip4/0.0.0.0/udp/{self.config.dht.port}/quic",
                    ],
                    initial_peers=[initial_peers_list[retries]],
                    announce_maddrs=announce_maddrs,
                    start=True,
                    # client_mode = True,
                )
                bt.logging.info(
                    f"Successfully initialised dht using initial_peer as {initial_peers_list[retries]}"
                )
                break
            except Exception as e:
                bt.logging.info(
                    f"Attempt {retries + 1} to init DHT using initial_peer as {initial_peers_list[retries]} failed with error: {e}"
                )
                retries += 1
                bt.logging.info(f"Retrying...")

        # Write local dht address to config
        self.config.neuron.initial_peers = self.config.neuron.initial_peers + [
            re.sub("ip4/?(.*?)/", f"ip{version}/{address}/", str(addr), flags=re.DOTALL)
            for addr in self.dht.get_visible_maddrs()
        ]

    async def forward(self):
        return await forward(self)


# # The main function parses the configuration and runs the validator.
if __name__ == "__main__":
    setup_logging()
    with Validator() as validator:
        while True:
            time.sleep(5)
