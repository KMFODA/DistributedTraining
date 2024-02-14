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

import re
import time
import typing
from functools import partial
from ipaddress import ip_address

import bittensor as bt
import hivemind
import requests
import torch
import wandb
from datasets import load_dataset
from hivemind import utils
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator,
    get_linear_schedule_with_warmup,
)

# Bittensor Miner Template:
import template
from bitarray import bitarray

# import base miner class which takes care of most of the boilerplate
from template.base.miner import BaseMinerNeuron
from template.utils.misc import load_wandb, setup_logging, get_bandwidth
from template.data.dataset import SubsetFalconLoader

import random
import asyncio


class Miner(BaseMinerNeuron):
    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)

        # Init device
        self.device = self.config.neuron.device

        # Init DHT and model
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

        # Init DHT
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
                )
                bt.logging.info(
                    f"Successfully initialised dht using initial_peer as {initial_peers_list[retries]}"
                )
                break
            except Exception as e:
                bt.logging.error(
                    f"Attempt {retries + 1} to init DHT using initial_peer as {initial_peers_list[retries]} failed with error: {e}"
                )
                retries += 1
                bt.logging.error(f"Retrying...")
        utils.log_visible_maddrs(self.dht.get_visible_maddrs(), only_p2p=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.config.neuron.model_name)

        # Add DHT address to wandb config
        self.config.neuron.initial_peers = self.config.neuron.initial_peers + [
            re.sub("ip4/?(.*?)/", f"ip{version}/{address}/", str(addr), flags=re.DOTALL)
            for addr in self.dht.get_visible_maddrs()
        ]

        # Move the model to the appropriate device
        self.model = self.model.to(self.device)

        # Set up a decentralized optimizer that will average with peers in background
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.config.neuron.lr)
        self.opt = hivemind.Optimizer(
            dht=self.dht,  # use a DHT that is connected with other peers
            run_id=self.config.neuron.run_id,  # unique identifier of this collaborative run
            scheduler=None,
            batch_size_per_step=self.config.neuron.local_batch_size_train*self.config.neuron.local_gradient_accumilation_steps_train,  # each call to opt.step adds this many samples towards the next epoch
            target_batch_size=self.config.neuron.global_batch_size_train,  # after peers collectively process this many samples, average weights and begin the next epoch
            optimizer=opt,  # wrap the SGD optimizer defined above
            use_local_updates=False,  # perform optimizer steps with local gradients, average parameters in background
            matchmaking_time=15.0,  # when averaging parameters, gather peers in background for up to this many seconds
            averaging_timeout=60.0,  # give up on averaging if not successful in this many seconds
            verbose=False,  # print logs incessently
            grad_compression=hivemind.Uniform8BitQuantization(),
            state_averaging_compression=hivemind.Uniform8BitQuantization(),
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.neuron.model_name)
        # Add the EOS token as PAD token to ensure our dataloader doesn't throw an error for sequences of unequal length
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load dataset
        self.dataset_loader = ()
        dataset_length = 968000015
        self.dataset_indices = bitarray(dataset_length)

        # Init Wandb
        if not self.config.neuron.dont_wandb_log:
            # self.wandb = load_wandb(self.config, self.wallet, "miner", str(self.dht.peer_id))
            self.wandb = load_wandb(self.config, self.wallet, "miner", str(1))

    # Define encoding function
    def encode(self, examples):
        return self.tokenizer(
            examples["text"], truncation=True, max_length=512, padding="max_length"
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
        self, synapse: template.protocol.IsAlive
    ) -> template.protocol.IsAlive:
        bt.logging.info("Responded to be Active")
        synapse.completion = "True"
        return synapse

    async def all_reduce(
        self, synapse: template.protocol.AllReduce
    ) -> template.protocol.IsAlive:
        bt.logging.info("Received All Reduce Call")
        # Sleep for 2 minutes and allow all-reduce to run in the background
        asyncio.sleep(120)
        synapse.completion = "True"
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
       
        search_start = random.choice(range(len(self.dataset_indices) -  self.config.neuron.training_examples_per_miner + 1))
        start = self.dataset_indices.index(bitarray('0'* self.config.neuron.training_examples_per_miner), search_start)
        group = [i for i in range(start,start +  self.config.neuron.training_examples_per_miner)]
        self.dataset_indices[group] = True

        # Create Dataloader
        dataloader = SubsetFalconLoader(
            batch_size=self.config.neuron.local_batch_size_train, sequence_length=1024, rows=group
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

            # Log training to hivemind
            self.opt.step()

            # Store gradients
            gradients = []
            for layer in self.model.parameters():
                gradients.append(layer.grad)

            # Zero gradients
            self.opt.zero_grad()

            bt.logging.info(f"Step {index} Loss: {outputs.loss.detach().item()}")
        
            if not self.config.neuron.dont_wandb_log:
                self.wandb.log({"loss": outputs.loss.detach().item(), "opt_local_epoch": self.opt.local_epoch})

        synapse.gradients = float(sum(gradients[synapse.gradient_test_index]))

        average_loss = total_loss / index
        synapse.loss = average_loss
        synapse.epoch = self.opt.tracker.local_progress.epoch
        synapse.dataset_indices = group

        event = {}
        event.update(self.get_miner_info())
        bt.logging.info('Logged miner info')
        event.update(get_bandwidth())
        bt.logging.info('Logged bandwidth info')
        event.update({'steps':index})
        bt.logging.info('Logged steps info')
        
        # bt.logging.debug(f"Events: {str(event)}")
        # bt.logging.info("EVENTS", "events", **event)

        if not self.config.neuron.dont_wandb_log:
            self.wandb.log(event)

        return synapse

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
