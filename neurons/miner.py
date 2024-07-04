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
import time
import typing

import bittensor as bt
import hivemind
import torch
from bitarray import bitarray
from hivemind.optim.progress_tracker import ProgressTracker
from transformers import AutoModelForCausalLM
import copy
import numpy as np

# Bittensor Miner Template:
import template

# import base miner class which takes care of most of the boilerplate
from template.base.miner import BaseMinerNeuron
from template.data.dataset import SubsetFalconLoader
from template.utils.gradient_averager import (
    DTGradientAverager,
)
from template.utils.state_loader import load_state_from_peer, DTStateAverager

from template.utils.progress_tracker import (
    GlobalTrainingProgress,
    LocalTrainingProgress,
    update_global_tracker_state,
)
from template.utils.misc import (
    get_bandwidth,
    init_dht,
    load_wandb,
    setup_logging,
)
from huggingface_hub import list_repo_refs


class Miner(BaseMinerNeuron):
    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)

        # Init Logging
        setup_logging(ip=self.config.axon.ip, port=self.config.axon.port)

        # Init DHT
        init_dht(self)

        # Init Device & Model
        self.device = self.config.neuron.device
        refs = list_repo_refs(self.config.neuron.model_name, repo_type="model")
        self.model_hf_tag = (
            max([int(tag.name) for tag in refs.tags]) if refs.tags else None
        )
        if self.model_hf_tag is None:
            bt.logging.warning(
                f"Model Tag Is None. Make Sure You Are Using The Correct Model Name"
            )
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                self.config.neuron.model_name, revision=str(self.model_hf_tag)
            )
            if self.model_hf_tag
            else AutoModelForCausalLM.from_pretrained(self.config.neuron.model_name)
        )

        # Move the model to the appropriate device
        self.model = self.model.to(self.device)

        # Set up a decentralized optimizer that will average with peers in background
        from template.utils.optimizer import VerboseAdamW

        self.opt = VerboseAdamW(self.model.parameters(), lr=self.config.neuron.lr)

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

        # Init Local & Global Progress
        self.local_progress = LocalTrainingProgress(
            peer_id=self.dht.peer_id.to_bytes(),
            epoch=0,
            samples_accumulated=0,
            samples_per_second=0.0,
            time=0.0,
            client_mode=False,
        )
        self.local_progress.epoch, self.local_progress.samples_accumulated = (
            self.model_hf_tag if self.model_hf_tag is not None else 0,
            0,
        )
        self.global_progress = GlobalTrainingProgress(epoch=0, samples_accumulated=0)
        self.global_progress.epoch, self.global_progress.samples_accumulated = (
            self.model_hf_tag if self.model_hf_tag is not None else 0,
            0,
        )
        update_global_tracker_state(self)

        # Init UID
        self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)

        # Load dataset
        self.dataset_loader = ()
        dataset_length = SubsetFalconLoader.max_pages
        self.dataset_indices = bitarray(dataset_length)

        # Init Wandb
        if not self.config.neuron.dont_wandb_log:
            self.wandb = load_wandb(
                self, self.config, self.wallet, "miner", str(self.dht.peer_id)
            )

        # Load state from peers if miner is not on latest global epoch
        if (self.local_progress.epoch < self.global_progress.epoch) and (
            self.model_hf_tag < self.global_progress.epoch
        ):
            load_state_from_peer(self, epoch=self.global_progress.epoch)

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
    ) -> template.protocol.AllReduce:
        bt.logging.info("Received All Reduce Call")
        failed_gradient_all_reduce = False
        try:
            bt.logging.info(
                [layer for layer in self.model.parameters()][-1][-10:].tolist()
            )
            bt.logging.info(
                [group["params"][-1][-10:].tolist() for group in self.opt.param_groups][
                    0
                ]
            )
            self.grad_averager.step(timeout=(synapse.timeout - 20))
            with self.grad_averager.use_averaged_gradients():  # this will fill param.grads with aggregated gradients
                bt.logging.info(
                    [layer for layer in self.model.parameters()][-1][-10:].tolist()
                )
                bt.logging.info(
                    [
                        group["params"][-1][-10:].tolist()
                        for group in self.opt.param_groups
                    ][0]
                )
                bt.logging.info("Model Weights Before Optimizer Step")
                current_model_weights = copy.deepcopy(
                    [layer for layer in self.model.parameters()][-100][-10:].tolist()[0]
                )
                current_model_weights_sample = copy.copy(
                    [layer for layer in self.model.parameters()][-1][-10:].tolist()
                )
                bt.logging.info(current_model_weights_sample)
                bt.logging.info("Model Gradients Before Optimizer Step")
                # Copy gradients
                gradients = tuple(
                    (
                        param.grad.detach().cpu().clone()
                        if param.grad is not None
                        else torch.zeros_like(param)
                    )
                    for param in self.model.parameters()
                )
                bt.logging.info(gradients[-1][-10:])
                bt.logging.info("Performing Optimizer Step")
                self.opt.step()

            bt.logging.info("Model Weights After Optimizer Step")
            new_model_weights = copy.deepcopy(
                [layer for layer in self.model.parameters()][-100][-10:].tolist()[0]
            )
            new_model_weights_sample = copy.copy(
                [layer for layer in self.model.parameters()][-1][-10:].tolist()
            )
            bt.logging.info(new_model_weights_sample)

            if new_model_weights == current_model_weights:
                bt.logging.info("Averaging Failed. Model Weights Haven't Changed.")
                failed_gradient_all_reduce = True
                load_state_from_peer(self, epoch=self.local_progress.epoch + 1)

            elif sum(np.isnan(new_model_weights_sample)) > 1:
                bt.logging.info(
                    "Averaging Failed. Model Weights Corrupted With Nans After Running The Optimizer Step."
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
                self.grad_averager.reset_accumulated_grads_()  # prepare for next step
                self.local_progress.epoch += 1
                self.local_progress.samples_accumulated = 0
                synapse.completion = "True"

        except Exception as e:
            bt.logging.info(f"Gradient averaging step failed with error {e}")
            failed_gradient_all_reduce = True
            update_global_tracker_state(self)
            load_state_from_peer(self, epoch=self.global_progress.epoch)
            synapse.completion = "False"

        if failed_gradient_all_reduce:
            with self.grad_averager.use_averaged_gradients():
                self.opt.zero_grad()
            bt.logging.info("Optimizer Gradients Zeroed")

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
        if (self.local_progress.epoch < self.global_progress.epoch) and (
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
        index = 0
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

            # # Test Step
            bt.logging.info(
                [layer for layer in self.model.parameters()][-1][-10:].tolist()
            )
            bt.logging.info(
                [group["params"][-1][-10:].tolist() for group in self.opt.param_groups][
                    0
                ]
            )

            # Zero Gradients
            self.opt.zero_grad()

            # Update Tracker
            self.local_progress.samples_accumulated += 1

            # Log accumulation status
            bt.logging.info(
                f"Index: {index} | Loss: {outputs.loss.detach().item():.2f}"
            )
            bt.logging.info(gradients[-1][-5:])
            if not self.config.neuron.dont_wandb_log:
                self.wandb.log(
                    {
                        "loss": outputs.loss.detach().item(),
                        "local_epoch": self.local_progress.epoch,
                        "global_epoch": self.global_progress.epoch,
                    }
                )

        if synapse.gradient_test_index > len(gradients):
            bt.logging.error(
                f"Request received from a validator running {synapse.model_name} whilst current miner is running {self.model.name_or_path}."
            )
            synapse.model_name = self.model.name_or_path
            return synapse
        else:
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
    with Miner() as miner:
        while True:
            bt.logging.info("Miner running...", time.time())
            time.sleep(5)
