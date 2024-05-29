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

import random
import re
import time
import typing
from ipaddress import ip_address

import bittensor as bt
import hivemind
import requests
import torch
import wandb
from bitarray import bitarray
from hivemind import utils
from hivemind.optim.progress_tracker import ProgressTracker
from hivemind.optim.state_averager import TrainingStateAverager
from transformers import AutoModelForCausalLM
import psutil
import copy
import numpy as np

# Bittensor Miner Template:
import template

# import base miner class which takes care of most of the boilerplate
from template.base.miner import BaseMinerNeuron
from template.data.dataset import SubsetFalconLoader
from template.utils.hivemind import (
    DTGradientAverager,
    DTStateAverager,
    load_state_from_peer,
    GlobalTrainingProgress,
    LocalTrainingProgress,
)
from template.utils.misc import (
    get_bandwidth,
    init_dht,
    load_wandb,
    setup_logging,
    warmup,
    update_global_tracker_state,
)
from huggingface_hub import list_repo_refs


class Miner(BaseMinerNeuron):
    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)

        # Init DHT
        init_dht(self)

        # Init device
        self.device = self.config.neuron.device

        # Init Model
        refs = list_repo_refs(self.config.neuron.model_name, repo_type="model")
        self.model_hf_tag = max([int(tag.name) for tag in refs.tags]) if refs.tags else None
        self.model = AutoModelForCausalLM.from_pretrained(self.config.neuron.model_name)

        # Move the model to the appropriate device
        self.model = self.model.to(self.device)

        # Set up a decentralized optimizer that will average with peers in background
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=self.config.neuron.lr)

        # Init Gradient Averager
        self.grad_averager = DTGradientAverager(
            self.model.parameters(),
            dht=self.dht,
            prefix=f"{self.config.neuron.run_id}_grad_averager",
            compression=hivemind.Uniform8BitQuantization(),
            # reuse_grad_buffers=True,
            accumulate_grads_on=torch.device(self.device),
            start=True,
            next_chunk_timeout=30.0,
        )

        # Init Tracker
        self.tracker = ProgressTracker(
            dht=self.dht,
            prefix=f"{self.config.neuron.run_id}",
            target_batch_size=self.config.neuron.global_batch_size_train,
            start=True,
        )

        # Init State Averager
        self.state_averager = DTStateAverager(
            optimizer=self.opt,
            initialize_optimizer=False,
            dht=self.dht,
            prefix=f"{self.config.neuron.run_id}_state_averager",
            state_compression=hivemind.Uniform8BitQuantization(),
            start=True,
            next_chunk_timeout=30.0,
        )

        # Init Tracker
        self.local_progress = LocalTrainingProgress(epoch=0, samples_accumulated=0)
        self.local_progress.epoch, self.local_progress.samples_accumulated = (
            self.model_hf_tag,
            0,
        )
        self.global_progress = GlobalTrainingProgress(epoch=0, samples_accumulated=0)
        self.global_progress.epoch, self.global_progress.samples_accumulated = 0, 0
        update_global_tracker_state(self)

        # Init UID
        self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)

        self.step_scheduled = False
        self.local_epoch, self.local_samples = 0, 0

        # Load dataset
        self.dataset_loader = ()
        dataset_length = 968000015
        self.dataset_indices = bitarray(dataset_length)

        # Init Wandb
        if not self.config.neuron.dont_wandb_log:
            self.wandb = load_wandb(
                self, self.config, self.wallet, "miner", str(self.dht.peer_id)
            )

        # Load state from peers if miner is not on latest epoch
        if (self.local_progress.epoch < self.global_progress.epoch) and (
            self.model_hf_tag < self.global_progress.epoch
        ):
            load_state_from_peer(self)

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
        self, synapse: template.protocol.IsAlive
    ) -> template.protocol.IsAlive:
        bt.logging.info("Responded to be Active")
        synapse.completion = "True"
        return synapse

    async def all_reduce(
        self, synapse: template.protocol.AllReduce
    ) -> template.protocol.IsAlive:
        bt.logging.info("Received All Reduce Call")
        try:
            with self.tracker.pause_updates():
                self.grad_averager.step(timeout=(synapse.timeout - 20))
                bt.logging.info("Model Weights Before Optimizer Step")
                current_model_weights_sample = copy.copy(
                    [layer for layer in self.model.parameters()][-1][-10:].tolist()
                )
                bt.logging.info(current_model_weights_sample)
                with self.grad_averager.use_averaged_gradients():  # this will fill param.grads with aggregated gradients
                    bt.logging.info("Performing Optimizer Step")
                    self.opt.step()

                bt.logging.info("Model Weights After Optimizer Step")
                new_model_weights_sample = copy.copy(
                    [layer for layer in self.model.parameters()][-1][-10:].tolist()
                )
                bt.logging.info(new_model_weights_sample)

                if new_model_weights_sample == current_model_weights_sample:
                    bt.logging.info("Averaging Failed. Model Weights Haven't Changed.")
                    load_state_from_peer(self, epoch = self.local_progress.epoch + 1)

                elif np.nan in new_model_weights_sample:
                    bt.logging.info(
                        "Averaging Failed. Model Weights Corrupted With Nans After Running The Optimizer Step."
                    )
                    load_state_from_peer(self, epoch = self.local_progress.epoch + 1)

                else:
                    self.grad_averager.reset_accumulated_grads_()  # prepare for next step
                    self.tracker.local_progress.epoch = self.tracker.update_epoch(
                        self.tracker.local_progress.epoch + 1
                    )
                    self.local_progress.epoch += 1
                    self.local_progress.samples_accumulated = 0
                    synapse.completion = "True"

        except Exception as e:
            bt.logging.info(f"Gradient averaging step failed with error {e}")
            update_global_tracker_state(self)
            load_state_from_peer(self, epoch=self.global_progress.epoch)
            synapse.completion = "False"

        return synapse

    async def forward(
        self, synapse: template.protocol.Train
    ) -> template.protocol.Train:
        """
        Processes the incoming 'Train' synapse by performing a training run

        Args:
            synapse (template.protocol.Train): The synapse object containing the 'dataset_indices' data.

        Returns:
            template.protocol.Train: The synapse object with the 'loss' field set to models loss.
        """
        update_global_tracker_state(self)
        if (self.tracker.local_progress.epoch < self.global_progress.epoch) and (
            self.model_hf_tag < self.global_progress.epoch
        ):
            load_state_from_peer(self, epoch=self.global_progress.epoch)

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
        dataloader = SubsetFalconLoader(
            batch_size=self.config.neuron.local_batch_size_train,
            sequence_length=1024,
            rows=group,
        )

        total_loss = 0
        # Train data for one epoch
        for index, batch in enumerate(dataloader):
            inputs = batch.to(self.device)

            # Forward pass
            outputs = self.model(input_ids=inputs, labels=inputs)

            # Normalize loss to account for batch accumulation
            loss = outputs.loss

            # Accumulate Total Loss
            total_loss += outputs.loss.detach().item()

            # Backward Pass
            loss.backward()

            # Copy gradients
            gradients = tuple(
                (
                    param.grad.detach().cpu().clone()
                    if param.grad is not None
                    else torch.zeros_like(param)
                )
                for param in self.model.parameters()
            )

            # Accumulate Gradients
            self.grad_averager.accumulate_grads_(batch_size=len(inputs))

            # Zero Gradients
            self.opt.zero_grad()

            # Update Tracker
            self.local_samples += 1
            self.local_progress.samples_accumulated += 1
            self.tracker.report_local_progress(self.local_epoch, self.local_samples)

            # Log accumulation status
            bt.logging.info(
                f"Index: {index} | Loss: {outputs.loss.detach().item():.2f} | Number of Peers: {self.tracker.global_progress.num_peers}"
            )
            if not self.config.neuron.dont_wandb_log:
                self.wandb.log(
                    {
                        "loss": outputs.loss.detach().item(),
                        "local_epoch": self.local_epoch,
                        "global_epoch": self.global_progress.epoch,
                    }
                )

        # Store summed random gradients in the synapse
        synapse.gradients = float(
            torch.sum(torch.abs(gradients[synapse.gradient_test_index]))
        )

        average_loss = total_loss / index
        synapse.loss = average_loss
        synapse.dataset_indices = group

        event = {}
        event.update(self.get_miner_info())
        try:
            event.update(get_bandwidth())
        except:
            bt.logging.info("Error getting bandwidth metrics")
        event.update({"steps": index})

        # bt.logging.debug(f"Events: {str(event)}")
        # bt.logging.info("EVENTS", "events", **event)

        if not self.config.neuron.dont_wandb_log:
            self.wandb.log(event)

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
        self, synapse: template.protocol.IsAlive
    ) -> typing.Tuple[bool, str]:
        blacklist = await self.blacklist_base(synapse)
        bt.logging.debug(blacklist[1])
        return blacklist

    async def blacklist_all_reduce(
        self, synapse: template.protocol.AllReduce
    ) -> typing.Tuple[bool, str]:
        blacklist = await self.blacklist_base(synapse)
        bt.logging.debug(blacklist[1])
        return blacklist

    async def blacklist_train(
        self, synapse: template.protocol.Train
    ) -> typing.Tuple[bool, str]:
        blacklist = await self.blacklist_base(synapse)
        bt.logging.info(blacklist[1])
        return blacklist

    async def priority_base(self, synapse: template.protocol.Train) -> float:
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
    setup_logging()
    with Miner() as miner:
        while True:
            bt.logging.info("Miner running...", time.time())
            time.sleep(5)
