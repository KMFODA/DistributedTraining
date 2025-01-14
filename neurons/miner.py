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
import os
import gc
import random
import time
import typing

os.environ["NEST_ASYNCIO"] = "0"
import copy

import bitsandbytes
import bittensor as bt
import numpy as np
import torch
from bitarray import bitarray
from bitsandbytes.optim import LAMB8bit
from bitsandbytes.cextension import lib
from transformers import AutoModelForCausalLM
import copy
import numpy as np
import threading

# Bittensor Miner Template:
import distributed_training
import hivemind
from distributed_training import __spec_version__, __version__
from distributed_training.base.miner import BaseMinerNeuron
from distributed_training.data.dataset import DataLoader
from distributed_training.utils.gradient_averager import (
    DTGradientAverager,
)
from distributed_training.utils.state_loader import (
    load_state_from_peer,
    ModelLoadingManager,
    load_model_optimizer_gradient_averager,
)

from distributed_training.utils.chain import log_peerid_to_chain
from distributed_training.utils.gradient_averager import DTGradientAverager
from distributed_training.utils.misc import (
    init_dht,
    load_wandb,
    setup_logging,
)
from distributed_training.utils.progress_tracker import (
    GlobalTrainingProgress,
    LocalTrainingProgress,
    get_global_epoch,
)
from distributed_training import __version__, __spec_version__

from huggingface_hub import hf_hub_download

# GPU optimizations.
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Seeds
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Add lamb to bnb str2optimizer8bit_blockwise
bitsandbytes.functional.str2optimizer8bit_blockwise
bitsandbytes.functional.str2optimizer8bit_blockwise["lamb"] = (
    lib.cadam_8bit_blockwise_grad_fp32,
    lib.cadam_8bit_blockwise_grad_fp16,
    lib.cadam_8bit_blockwise_grad_bf16,
)


class Miner(BaseMinerNeuron):
    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)

        # Init Logging
        setup_logging(
            network=self.config.subtensor.network,
            netuid=self.config.netuid,
            hotkey=self.wallet.hotkey.ss58_address,
            version=__version__,
            spec_version=__spec_version__,
            run_id=None,
            ip=(
                self.config.axon.ip
                if self.config.axon.ip != "[::]"
                else bt.utils.networking.get_external_ip()
            ),
            port=self.config.axon.port,
            uid=self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address),
            neuron_type="miner",
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
        self.global_progress.epoch = get_global_epoch(self)
        self.local_progress.epoch = self.global_progress.epoch
        if self.global_progress.epoch is None:
            bt.logging.error(
                f"Model Tag Is None. Make Sure You Are Using The Correct Model Name"
            )

        # Init Device
        self.device = self.config.neuron.device

        # Init UID
        self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)

        # Init Optimizer & Gradient Averager Variables
        self.learning_rate_maximum = 6e-4
        self.weight_decay = 0.1
        self.all_reduce_timeout = 360

        # Init Model, Optimizer & Gradient Averager
        load_model_optimizer_gradient_averager(self, self.global_progress.epoch)

        # Init Background Loop
        self.loop = asyncio.new_event_loop()
        self._p2p = self.loop.run_until_complete(self.dht.replicate_p2p())
        self.peer_list = self.loop.run_until_complete(self._p2p.list_peers())

        # Create mapping between uids to peerids
        self.uids_to_peerids = {uid: None for uid in self.metagraph.uids.tolist()}

        # Load dataset
        self.dataset_loader = ()
        dataset_length = DataLoader.max_rows
        self.dataset_indices = bitarray(dataset_length)

        # Init Wandb
        if not self.config.neuron.dont_wandb_log:
            self.wandb = load_wandb(
                self, self.config, self.wallet, "miner", str(self.dht.peer_id)
            )

        # Init model_loading_manager
        self.model_loading_manager = ModelLoadingManager()

        # Load state from peers if miner is not on latest global epoch
        if self.local_progress.epoch != self.global_progress.epoch:
            load_state_from_peer(self, epoch=self.global_progress.epoch)

        # Init Tracking event
        self.event = {}

        # Init background threads
        self.stop_event = threading.Event()

        self.update_model_thread = threading.Thread(
            target=self.load_latest_model, daemon=True
        )
        self.update_model_thread.start()

        # Log PeerID to chain
        bt.logging.info("Logging PeerID to chain")
        log_peerid_to_chain(self)

    def start_dataloader_thread(self):
        """Start a new dataloader thread if the previous one is finished"""
        if hasattr(self, "dataloader_thread") and self.dataloader_thread.is_alive():
            self.dataloader_thread.join()

        self.dataloader_thread = threading.Thread(
            target=self.load_dataloader, daemon=True
        )
        self.dataloader_thread.start()

    def is_dataloader_thread_alive(self):
        """Check if dataloader thread is alive"""
        return hasattr(self, "dataloader_thread") and self.dataloader_thread.is_alive()

    def load_latest_model(self):
        while not self.stop_event.is_set():
            # Skip checking if we're currently loading
            if (self.model_loading_manager.is_loading) or (
                hasattr(self, "model") == False
            ):
                time.sleep(5)  # Short sleep before checking again
                continue

            self.global_progress.epoch = get_global_epoch(self)

            if self.global_progress.epoch is None:
                time.sleep(30)
                continue

            if (
                self.global_progress.epoch
                == self.model_loading_manager.last_loaded_epoch
                and self.global_progress.epoch == self.local_progress.epoch
            ):
                time.sleep(30)
                continue

            needs_update = (
                self.local_progress.epoch < self.global_progress.epoch
                or sum(
                    np.isnan(
                        [layer for layer in self.model.parameters()][-2][-10:].tolist()
                    )
                )
                > 1
            )

            if needs_update:
                bt.logging.info(
                    f"Local Epoch {self.local_progress.epoch} Behind Global Epoch {self.global_progress.epoch}. Loading Latest Model State."
                )
                if not self.model_loading_manager.is_loading:
                    load_state_from_peer(self, epoch=self.global_progress.epoch)
            else:
                time.sleep(30)

    def load_dataloader(self):
        bt.logging.info("DataLoader initialisation started")
        print("DataLoader initialisation started")
        search_start = random.choice(
            range(
                len(self.dataset_indices)
                - self.config.neuron.training_examples_per_miner
                + 1
            )
        )
        start = self.dataset_indices.index(
            bitarray("0" * self.config.neuron.training_examples_per_miner), search_start
        )
        self.group = [
            i
            for i in range(
                start, start + self.config.neuron.training_examples_per_miner
            )
        ]

        self.dataset_indices[self.group] = True

        # Create Dataloader
        self.dataloader = DataLoader(
            batch_size=self.config.neuron.local_batch_size_train,
            sequence_length=1024,
            rows=self.group,
        )

    def get_miner_info(self):
        return {
            "block": self.metagraph.block.item(),
            "stake": self.metagraph.stake[self.uid],
            "trust": self.metagraph.trust[self.uid],
            "consensus": self.metagraph.consensus[self.uid],
            "incentive": self.metagraph.incentive[self.uid],
            "emissions": self.metagraph.emission[self.uid],
        }

    async def is_alive(
        self, synapse: distributed_training.protocol.IsAlive
    ) -> distributed_training.protocol.IsAlive:
        bt.logging.info("Responded to be Active")
        synapse.completion = "True"
        synapse.epoch = self.local_progress.epoch
        return synapse

    async def all_reduce(
        self, synapse: distributed_training.protocol.AllReduce
    ) -> distributed_training.protocol.AllReduce:
        bt.logging.info("Received All Reduce Call")

        # Wait for model to load if it is currently loading
        while self.model_loading_manager.is_loading:
            time.sleep(1)

        failed_gradient_all_reduce = False

        # Set to True to avoid state loading during allreduce
        self.model_loading_manager.set_loading_state(True)

        # Update the gradient averaging kwargs
        if synapse.next_chunk_timeout is not None:
            self.grad_averager.next_chunk_timeout = synapse.next_chunk_timeout
            self.grad_averager.allreduce_kwargs[
                "sender_timeout"
            ] = self.grad_averager.next_chunk_timeout
            self.grad_averager.allreduce_kwargs["reducer_timeout"] = (
                self.grad_averager.next_chunk_timeout * 2
            )
        if synapse.all_reduce_timeout is not None:
            self.grad_averager._allreduce_timeout = synapse.all_reduce_timeout
        if synapse.min_group_size is not None:
            self.grad_averager.matchmaking_kwargs[
                "min_group_size"
            ] = synapse.min_group_size
        if synapse.request_timeout is not None:
            self.grad_averager.matchmaking_kwargs[
                "request_timeout"
            ] = synapse.request_timeout
        if synapse.min_matchmaking_time is not None:
            self.grad_averager.matchmaking_kwargs[
                "min_matchmaking_time"
            ] = synapse.min_matchmaking_time

        # # Update mapping of uids to peerids
        try:
            gradient_averaging_step = self.grad_averager.step(
                timeout=(synapse.timeout - 20),
                wait=False,
                gather=self.local_progress.samples_accumulated,
            )
            start_time = time.perf_counter()

            while (gradient_averaging_step.done() is False) and (
                (time.perf_counter() - start_time) <= synapse.timeout
            ):
                time.sleep(1)

            if gradient_averaging_step.done():
                with self.grad_averager.use_averaged_gradients():  # this will fill param.grads with aggregated gradients
                    bt.logging.info("Model Weights Before Optimizer Step")
                    current_model_weights_sample = copy.copy(
                        [layer for layer in self.model.parameters()][-2][-10:].tolist()
                    )
                    bt.logging.info(current_model_weights_sample)

                    bt.logging.info("Clipping Grads")
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                    if synapse.learning_rate is not None:
                        bt.logging.info(
                            f"Updating Optimizer Learning Rate To: {synapse.learning_rate}"
                        )
                        for param_group in self.opt.param_groups:
                            param_group["lr"] = synapse.learning_rate

                    bt.logging.info("Performing Optimizer Step")
                    self.opt.step()

                    # Reset gradient buffers
                    self.grad_averager.reset_accumulated_grads_()

                # Set back to false to allow state loading
                self.model_loading_manager.set_loading_state(False)

                bt.logging.info("Model Weights After Optimizer Step")
                new_model_weights_sample = copy.copy(
                    [layer for layer in self.model.parameters()][-2][-10:].tolist()
                )
                bt.logging.info(new_model_weights_sample)

                if new_model_weights_sample == current_model_weights_sample:
                    bt.logging.info(
                        "Averaging Failed. Model Weights Haven't Changed. Loading Latest Model State."
                    )
                    failed_gradient_all_reduce = True
                    load_state_from_peer(self, epoch=self.local_progress.epoch + 1)

                elif sum(np.isnan(new_model_weights_sample)) > 1:
                    bt.logging.info(
                        "Averaging Failed. Model Weights Corrupted With NaNs After Running The Optimizer Step. Loading Latest Model State."
                    )
                    failed_gradient_all_reduce = True
                    state_loaded = load_state_from_peer(
                        self, epoch=self.local_progress.epoch + 1
                    )
                    if not state_loaded:
                        state_loaded = load_state_from_peer(
                            self, epoch=self.local_progress.epoch
                        )

                else:
                    # Update local progress
                    self.local_progress.epoch += 1
                    self.local_progress.samples_accumulated = 0
                    synapse.completion = "True"

            else:
                bt.logging.info("Averaging Failed. Loading Latest Model State.")
                failed_gradient_all_reduce = True
                # Set back to false to allow state loading
                self.model_loading_manager.set_loading_state(False)
                load_state_from_peer(self)

        except Exception as e:
            bt.logging.info(
                f"Gradient Averaging Step Failed With Error: {e}. Loading Latest Model State."
            )
            failed_gradient_all_reduce = True
            self.global_progress.epoch = get_global_epoch(self)
            # Set back to false to allow state loading
            self.model_loading_manager.set_loading_state(False)
            load_state_from_peer(self, epoch=self.global_progress.epoch)
            synapse.completion = "False"

        if failed_gradient_all_reduce:
            gradient_averaging_step.cancel()
            bt.logging.info("Gradient Step Cancelled")
            with self.grad_averager.use_averaged_gradients():
                self.opt.zero_grad()
            bt.logging.info("Optimizer Gradients Zeroed")

        return synapse

    async def forward(
        self, synapse: distributed_training.protocol.Train
    ) -> distributed_training.protocol.Train:
        """
        Processes the incoming 'Train' synapse by performing a training run

        Args:
            synapse (template.protocol.Train): The synapse object containing the 'dataset_indices' data.

        Returns:
            template.protocol.Train: The synapse object with the 'loss' field set to models loss.
        """
        timeout: float = synapse.timeout
        start_time: float = time.perf_counter()

        self.global_progress.epoch = get_global_epoch(self)

        # Wait for model to load if it is currently loading
        while self.model_loading_manager.is_loading:
            time.sleep(1)

        # Load the latest model if self.local_progress.epoch != self.global_progress.epoch
        if (self.local_progress.epoch != self.global_progress.epoch) or (
            sum(
                np.isnan(
                    [layer for layer in self.model.parameters()][-2][-10:].tolist()
                )
            )
            > 1
        ):
            bt.logging.info(
                f"Local Epoch {self.local_progress.epoch} Behind Global Epoch {self.global_progress.epoch}. Loading Latest Model State."
            )
            load_state_from_peer(self, epoch=self.global_progress.epoch)

        # Start dataloader
        search_start = random.choice(
            range(
                len(self.dataset_indices)
                - self.config.neuron.training_examples_per_miner
                + 1
            )
        )
        start = self.dataset_indices.index(
            bitarray("0" * self.config.neuron.training_examples_per_miner), search_start
        )
        group = [
            i
            for i in range(
                start, start + self.config.neuron.training_examples_per_miner
            )
        ]

        self.dataset_indices[group] = True

        # Create Dataloader
        dataloader = DataLoader(
            batch_size=self.config.neuron.local_batch_size_train,
            sequence_length=1024,
            rows=group,
        )

        synapse.batch_size = self.config.neuron.local_batch_size_train

        total_loss = 0
        gradient_sum_list = []

        target_param = list(self.model.parameters())[synapse.gradient_test_index]

        # Training loop
        for index, batch in enumerate(dataloader):
            # Extract inputs and labels
            inputs = batch[0].to(self.device)
            labels = batch[1].to(self.device)

            # Zero Gradients
            self.opt.zero_grad()

            # Forward pass
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = self.model(input_ids=inputs, labels=labels)
                loss = outputs[1]

            # Accumulate Total Loss
            total_loss += loss.detach().item()

            # Backward Pass
            loss.backward()

            # Accumulate Gradients
            self.grad_averager.accumulate_grads_(batch_size=inputs.size(0))

            # Update Tracker
            self.local_progress.samples_accumulated += inputs.size(0)

            # Extract gradient for the test_layer_index
            gradient = target_param.grad.detach()

            gradient_sum_list.append(torch.sum(torch.abs(gradient)).item())

            # Log accumulation status
            bt.logging.info(f"Index: {index} | Loss: {loss.detach().item():.2f}")

        if synapse.gradient_test_index >= len(gradient):
            bt.logging.error(
                f"Request Received From A Validator Running {synapse.model_name} Whilst Current Miner Is Running {self.model.name_or_path}."
            )
            synapse.model_name = self.model.name_or_path
            return synapse

        # Store the list of gradient sums and projected gradients in the synapse
        synapse.gradient_sums = gradient_sum_list

        average_loss = total_loss / (index + 1)
        synapse.loss = average_loss
        synapse.dataset_indices = group

        if not self.config.neuron.dont_wandb_log:
            self.event.update(
                {
                    "loss": synapse.loss,
                    "local_epoch": self.local_progress.epoch,
                    "global_epoch": self.global_progress.epoch,
                    "steps": index,
                }
            )
            self.wandb.log(self.event)
            self.event = {}

        if time.perf_counter() - start_time > timeout:
            bt.logging.error(
                f"Timed out responding to request from {synapse.dendrite.hotkey}. Try decreasing config.neuron.training_examples_per_miner or upgrading to a faster GPU."
            )
        else:
            bt.logging.info(
                f"Succesfully responded to request from {synapse.dendrite.hotkey} in {time.perf_counter() - start_time} seconds."
            )

        return synapse

    def warmup(
        self,
    ):
        (self)

    async def blacklist_base(self, synapse) -> typing.Tuple[bool, str]:
        """
        Determines whether an incoming request should be blacklisted and thus ignored. Your implementation should
        define the logic for blacklisting requests based on your needs and desired security parameters.

        Blacklist runs before the synapse data has been deserialized (i.e. before synapse.data is available).
        The synapse is instead contructed via the headers of the request. It is important to blacklist
        requests before they are deserialized to avoid wasting resources on requests that will be ignored.

        Args:
            synapse (template.protocol.Train): A synapse object constructed from the headers of the incoming request.

        Returns:
            Tuple[bool, str]: A tuple containing a boolean indicating whether the synapse's hotkey is blacklisted,
                            and a string providing the reason for the decision.

        This function is a security measure to prevent resource wastage on undesired requests. It should be enhanced
        to include checks against the metagraph for entity registration, validator status, and sufficient stake
        before deserialization of synapse data to minimize processing overhead.

        Example blacklist logic:
        - Reject if the hotkey is not a registered entity within the metagraph.
        - Consider blacklisting entities that are not validators or have insufficient stake.

        In practice it would be wise to blacklist requests from entities that are not validators, or do not have
        enough stake. This can be checked via metagraph.S and metagraph.validator_permit. You can always attain
        the uid of the sender via a metagraph.hotkeys.index( synapse.dendrite.hotkey ) call.

        Otherwise, allow the request to be processed further.
        """
        hotkey = synapse.dendrite.hotkey
        synapse_type = type(synapse).__name__

        uid = None
        axon = None
        for _uid, _axon in enumerate(self.metagraph.axons):
            if _axon.hotkey == hotkey:
                uid = _uid
                axon = _axon
                break

        if uid is None:
            bt.logging.trace(
                f"Blacklisting unrecognized hotkey: {synapse.dendrite.hotkey}"
            )
            return (
                True,
                f"Blacklisted a non registered hotkey's {synapse_type} request from {hotkey}",
            )

        if self.config.blacklist.force_validator_permit and (
            not self.config.blacklist.allow_non_registered
        ):
            # Check stake if uid is recognize
            tao = self.metagraph.neurons[uid].stake.tao
            if tao < self.config.neuron.vpermit_tao_limit:
                return (
                    True,
                    f"Blacklisted a low stake {synapse_type} request: {tao} < {self.config.neuron.vpermit_tao_limit} from {hotkey}",
                )

        if synapse.dendrite.hotkey not in self.metagraph.hotkeys:
            # Ignore requests from unrecognized entities.
            bt.logging.trace(
                f"Blacklisting unrecognized hotkey {synapse.dendrite.hotkey}"
            )
            return True, "Unrecognized hotkey"

        bt.logging.trace(
            f"Not Blacklisting recognized hotkey {synapse.dendrite.hotkey}"
        )
        return False, "Hotkey recognized!"

    async def blacklist_is_alive(
        self, synapse: distributed_training.protocol.IsAlive
    ) -> typing.Tuple[bool, str]:
        blacklist = await self.blacklist_base(synapse)
        bt.logging.debug(blacklist[1])
        return blacklist

    async def blacklist_all_reduce(
        self, synapse: distributed_training.protocol.AllReduce
    ) -> typing.Tuple[bool, str]:
        blacklist = await self.blacklist_base(synapse)
        bt.logging.debug(blacklist[1])
        return blacklist

    async def blacklist_train(
        self, synapse: distributed_training.protocol.Train
    ) -> typing.Tuple[bool, str]:
        blacklist = await self.blacklist_base(synapse)
        bt.logging.info(blacklist[1])
        return blacklist

    async def priority_base(
        self, synapse: distributed_training.protocol.Train
    ) -> float:
        """
        The priority function determines the order in which requests are handled. More valuable or higher-priority
        requests are processed before others. You should design your own priority mechanism with care.

        This implementation assigns priority to incoming requests based on the calling entity's stake in the metagraph.

        Args:
            synapse (template.protocol.Train): The synapse object that contains metadata about the incoming request.

        Returns:
            float: A priority score derived from the stake of the calling entity.

        Miners may recieve messages from multiple entities at once. This function determines which request should be
        processed first. Higher values indicate that the request should be processed first. Lower values indicate
        that the request should be processed later.

        Example priority logic:
        - A higher stake results in a higher priority value.
        """
        caller_uid = self.metagraph.hotkeys.index(
            synapse.dendrite.hotkey
        )  # Get the caller index.
        prirority = float(
            self.metagraph.S[caller_uid]
        )  # Return the stake as the priority.
        bt.logging.trace(
            f"Prioritizing {synapse.dendrite.hotkey} with value: ", prirority
        )
        return prirority


# This is the main function, which runs the miner.
if __name__ == "__main__":
    with Miner() as miner:
        while True:
            bt.logging.info("Miner running...", time.time())
            time.sleep(5)
