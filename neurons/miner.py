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
import distributed_training

# import base miner class which takes care of most of the boilerplate
from distributed_training.base.miner import BaseMinerNeuron
from distributed_training.data.dataset import SubsetFalconLoader
from distributed_training.utils.gradient_averager import (
    DTGradientAverager,
)
from distributed_training.utils.state_loader import (
    load_state_from_peer,
    DTStateAverager,
)

from distributed_training.utils.progress_tracker import (
    GlobalTrainingProgress,
    LocalTrainingProgress,
    update_global_tracker_state,
)
from distributed_training.utils.misc import (
    get_bandwidth,
    init_dht,
    load_wandb,
    setup_logging,
)
from distributed_training.utils.uids import map_uid_to_peerid
from torch_optimizer import Lamb
from distributed_training import __version__, __spec_version__


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
            ip=self.config.axon.ip
            if self.config.axon.ip != "[::]"
            else bt.utils.networking.get_external_ip(),
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
        update_global_tracker_state(self)

        # Init Device & Model
        self.device = self.config.neuron.device
        if self.global_progress.epoch is None:
            bt.logging.error(
                f"Model Tag Is None. Make Sure You Are Using The Correct Model Name"
            )
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                self.config.neuron.model_name,
                revision=str(self.global_progress.epoch),
                trust_remote_code=True,
            )
            if self.global_progress.epoch
            else AutoModelForCausalLM.from_pretrained(
                self.config.neuron.model_name, trust_remote_code=True
            )
        )

        # Move the model to the appropriate device
        self.model = self.model.to(self.device)

        # Init UID
        self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)

        # Set up a decentralized optimizer that will average with peers in background
        self.opt = Lamb(self.model.parameters(), lr=self.config.neuron.learning_rate)

        # Init Gradient Averager
        self.grad_averager = DTGradientAverager(
            self.model.parameters(),
            dht=self.dht,
            prefix=f"{self.config.neuron.run_id}_grad_averager",
            compression=hivemind.Uniform8BitQuantization(),
            accumulate_grads_on=torch.device(self.device),
            start=True,
            next_chunk_timeout=30.0,
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
        if self.local_progress.epoch != self.global_progress.epoch:
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
        failed_gradient_all_reduce = False

        # Update mapping of uids to peerids
        self.uids_to_peerids = await map_uid_to_peerid(self, range(0, self.metagraph.n))
        self.uids_to_peerids[self.uid] = self.dht.peer_id
        bt.logging.info(f"UID To PeerID Mapping: {self.uids_to_peerids}")
        try:
            self.grad_averager.step(
                timeout=(synapse.timeout - 20),
                weight=(
                    self.local_progress.samples_accumulated
                    / self.config.neuron.global_batch_size_train
                ),
            )
            with self.grad_averager.use_averaged_gradients():  # this will fill param.grads with aggregated gradients
                bt.logging.info("Model Weights Before Optimizer Step")
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
                if synapse.learning_rate is not None:
                    bt.logging.info(
                        f"Updating Optimizer Learning Rate To: {synapse.learning_rate}"
                    )
                    for param_group in self.opt.param_groups:
                        param_group["lr"] = synapse.learning_rate
                bt.logging.info("Performing Optimizer Step")
                self.opt.step()

            bt.logging.info("Model Weights After Optimizer Step")
            new_model_weights_sample = copy.copy(
                [layer for layer in self.model.parameters()][-1][-10:].tolist()
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
                self.grad_averager.reset_accumulated_grads_()  # prepare for next step
                self.local_progress.epoch += 1
                self.local_progress.samples_accumulated = 0
                synapse.completion = "True"

        except Exception as e:
            bt.logging.info(
                f"Gradient Averaging Step Failed With Error: {e}. Loading Latest Model State."
            )
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
        self, synapse: distributed_training.protocol.Train
    ) -> distributed_training.protocol.Train:
        """
        Processes the incoming 'Train' synapse by performing a training run

        Args:
            synapse (template.protocol.Train): The synapse object containing the 'dataset_indices' data.

        Returns:
            template.protocol.Train: The synapse object with the 'loss' field set to models loss.
        """
        update_global_tracker_state(self)
        if (self.local_progress.epoch != self.global_progress.epoch) or (
            sum(
                np.isnan(
                    [layer for layer in self.model.parameters()][-1][-10:].tolist()
                )
            )
            > 1
        ):
            bt.logging.info(
                "Local Epoch Behind Global Epoch. Loading Latest Model State."
            )
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
            # Extract inputs and labels
            inputs = batch[0].to(self.device)
            labels = batch[1].to(self.device)

            # Forward pass
            outputs = self.model(input_ids=inputs, labels=labels)

            # Normalize loss to account for batch accumulation
            logits, loss = outputs

            # Accumulate Total Loss
            total_loss += loss.detach().item()

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
            self.local_progress.samples_accumulated += 1

            # Log accumulation status
            bt.logging.info(f"Index: {index} | Loss: {loss.detach().item():.2f}")
            if not self.config.neuron.dont_wandb_log:
                self.wandb.log(
                    {
                        "loss": loss.detach().item(),
                        "local_epoch": self.local_progress.epoch,
                        "global_epoch": self.global_progress.epoch,
                    }
                )

        if synapse.gradient_test_index > len(gradients):
            bt.logging.error(
                f"Request Received From A Validator Running {synapse.model_name} Whilst Current Miner Is Running {self.model.name_or_path}."
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
