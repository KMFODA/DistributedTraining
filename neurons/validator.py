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
import datetime as dt
import threading
import time
import traceback
from typing import Optional

import bittensor as bt
import hivemind
import torch
from bitarray import bitarray
from hivemind.compression import deserialize_torch_tensor
from hivemind.optim.progress_tracker import ProgressTracker
from hivemind.optim.state_averager import TrainingStateAverager
from hivemind.p2p import PeerID
from hivemind.proto import averaging_pb2
from hivemind.utils import get_logger
from hivemind.utils.asyncio import aiter_with_timeout
from hivemind.utils.streaming import combine_from_streaming
from hivemind.utils.timed_storage import ValueWithExpiration
from transformers import AutoModelForCausalLM

from template.base.validator import BaseValidatorNeuron
from template.utils.hivemind import DTGradientAverager, DTStateAverager
from template.utils.misc import AsyncDendritePool, init_dht, load_wandb, setup_logging
from template.validator import forward

logger = get_logger(__name__)

class Validator(BaseValidatorNeuron):
    def __init__(self, config=None):
        super(Validator, self).__init__(config=config)

        bt.logging.info("load_state()")
        self.load_state()

        num_peers = 0
        while num_peers == 0:

            # Init DHT
            init_dht(self)

            # Init Tracker
            self.tracker = ProgressTracker(
                dht=self.dht, 
                prefix=f"{self.config.neuron.run_id}", 
                target_batch_size=self.config.neuron.global_batch_size_train,
                start=True
            )
            num_peers = self.tracker.global_progress.num_peers

            bt.logging.info(f'Number of connected peers after initialising the DHT is {num_peers}')
            if num_peers == 0:
                bt.logging.info('Re-initialising the DHT')
            time.sleep(10)

        # Init Wandb
        if not self.config.neuron.dont_wandb_log:
            self.wandb = load_wandb(self, self.config, self.wallet, "validator", str(self.dht.peer_id))

        # Init Dendrite Pool
        self.dendrite_pool = AsyncDendritePool(
            wallet=self.wallet, metagraph=self.metagraph
        )
        
        # TODO set correctly
        self.previous_loss = 0
        self.previous_ppl = 0
        
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

        # Init Gradient Averager
        self.grad_averager = DTGradientAverager(
            self.model.parameters(),
            dht=self.dht,
            prefix=f"{self.config.neuron.run_id}_grad_averager",
            compression=hivemind.Uniform8BitQuantization(),
            # reuse_grad_buffers=True,
            accumulate_grads_on=torch.device("cuda"),
            start = True,
            next_chunk_timeout = 30.0,
            auxiliary = True
        )
        
        # Init State Averager
        self.state_averager = DTStateAverager(
            optimizer = self.opt,
            dht=self.dht,
            prefix=f"{self.config.neuron.run_id}_state_averager",
            state_compression=hivemind.Uniform8BitQuantization(),
            start = True,
            next_chunk_timeout = 30.0,
        )

        self.step_scheduled = False
        self.local_epoch, self.local_samples = 0, 0
        self.loop = asyncio.new_event_loop()
        self._p2p = self.loop.run_until_complete(self.dht.replicate_p2p())
        self.peer_list = self.loop.run_until_complete(self._p2p.list_peers())
        self.peer_id = self.dht.peer_id
        self.get_stub = self.grad_averager.get_stub
        self.serializer = self.grad_averager.serializer
        
        # Create mapping between uids to peerids 
        self.uids_to_peerids = self.loop.run_until_complete(self.map_uid_to_peerid(range(0, self.metagraph.n)))
        max_retries = 3
        retries = 0
        while all(value is None for value in self.uids_to_peerids.values()) and (retries >= max_retries):
            for retries in range(0, max_retries):
                self.uids_to_peerids = self.loop.run_until_complete(self.map_uid_to_peerid(range(0, self.metagraph.n)))
                time.sleep(1)

        # self.miner_iterator = MinerIterator(self.metagraph.uids.tolist())
        # self.stop_event = threading.Event()
        # self.update_thread = threading.Thread(target=self.map_uids_to_peerids, daemon=True)
        # self.update_thread.start()
        
        # Start Main Validation Loop
        bt.logging.info("Starting validator loop.")

    def map_uids_to_peerids(self):
        # Track how recently we updated each uid
        uid_last_checked = dict()

        # The below loop iterates across all miner uids and checks to see
        # if they should be updated.
        while not self.stop_event.is_set():
            try:
                # Get the next uid to check
                next_uid = next(self.miner_iterator)

                # Confirm that we haven't checked it in the last 5 minutes.
                time_diff = (
                    dt.datetime.now() - uid_last_checked[next_uid]
                    if next_uid in uid_last_checked
                    else None
                )

                if time_diff and time_diff < dt.timedelta(minutes=5):
                    # If we have seen it within 5 minutes then sleep until it has been at least 5 minutes.
                    time_to_sleep = (
                        dt.timedelta(minutes=5) - time_diff
                    ).total_seconds()
                    bt.logging.trace(
                        f"Update loop has already processed all UIDs in the last 5 minutes. Sleeping {time_to_sleep} seconds."
                    )
                    time.sleep(time_to_sleep)

                uid_last_checked[next_uid] = dt.datetime.now()

                # Get their hotkey from the metagraph.
                hotkey = self.metagraph.hotkeys[next_uid]

                # Compare metadata and tracker, syncing new model from remote store to local if necessary.
                metadata = bt.extrinsics.serving.get_metadata(self.subtensor, self.config.netuid,  self.metagraph.hotkeys[next_uid])

                if not metadata:
                    updated = None
                else:
                    commitment = metadata["info"]["fields"][0]
                    hex_data = commitment[list(commitment.keys())[0]][2:]
                    chain_str = bytes.fromhex(hex_data).decode()
                    updated = PeerID(chain_str)

                if self.uids_to_peerids[next_uid] != updated:
                    bt.logging.trace(
                        f"Updated peerID for UID={next_uid}. Was new = {updated}"
                    )
                    self.uids_to_peerids[next_uid] = updated

            except Exception as e:
                bt.logging.error(
                    f"Error in update loop: {e} \n {traceback.format_exc()}"
                )

        bt.logging.info("Exiting update models loop.")


    def get_validator_info(self):
        return {
            "block": self.metagraph.block.item(),
            "stake": self.metagraph.stake[self.uid],
            "rank": self.metagraph.ranks[self.uid],
            "vtrust": self.metagraph.validator_trust[self.uid],
            "dividends": self.metagraph.dividends[self.uid],
            "emissions": self.metagraph.emission[self.uid],
        }

    async def map_uid_to_peerid(self, uids):
        uids_to_peerids = {}
        for uid in uids:
            miner_ip = self.metagraph.axons[uid].ip

            # Get all peers connected to our DHT and their ips
            peer_list_dht = await self._p2p.list_peers()
            peer_list_dht_addrs = [str(peer.addrs[0]).split('/ip4/')[1].split('/')[0] for peer in peer_list_dht]

            # Get only peers connected to the current run id
            prefix = self.grad_averager.matchmaking_kwargs['prefix']
            metadata, _ = self.dht.get(f"{prefix}.all_averagers", latest=True) or ({}, None)

            if metadata is None:
                # return None
                uids_to_peerids[uid] = None
                continue
            peer_list_run = [str(PeerID(peer_id)) for peer_id, info in metadata.items() if isinstance(info, ValueWithExpiration) and isinstance(info.value, (float, int))]

            # If the UIDs ip address is not in the list of peer addrs then it is not connected to our DHT
            if miner_ip not in peer_list_dht_addrs:
                # return None
                uids_to_peerids[uid] = None
                continue
            else:
                peer_id = peer_list_dht[peer_list_dht_addrs.index(miner_ip)].peer_id

            # If peer_id is not in the list of peer ids for our run then it is not connected to our run ID
            if str(peer_id) not in peer_list_run:
                # return None
                uids_to_peerids[uid] = None
                continue
            else:
                # return peer_id
                uids_to_peerids[uid] = peer_id
                continue
        return uids_to_peerids

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
