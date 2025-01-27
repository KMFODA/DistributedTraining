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
import random
import tempfile
import time
import typing
from enum import Enum

os.environ["NEST_ASYNCIO"] = "0"
import threading
from concurrent.futures import ThreadPoolExecutor

import bitsandbytes
import bittensor as bt
import numpy as np
import torch
from bitsandbytes.cextension import lib
from huggingface_hub import create_tag, upload_folder
from transformers import AutoTokenizer

import distributed_training
from distributed_training.averaging.avg_handler import AveragingHandler
from distributed_training.base.miner import BaseMinerNeuron
from distributed_training.data.dataset import DatasetLoader
from distributed_training.exceptions import (
    TrainingError,
    handle_error,
    log_and_handle_error,
)
from distributed_training.utils.chain import log_peerid_to_chain
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
from distributed_training.utils.state_loader import (
    load_model_optimizer_gradient_averager,
    load_state_from_peer,
)

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


class TrainingStatus(Enum):
    RUNNING = "running"
    ERROR = "error"
    STOPPED = "stopped"


# TODO Consider when/how we would do model loading when using diloco
# TODO I.e. if peers join in-between outer steps, then load the latest, but skip training to only sync the model, to then start training the new step
class Miner(BaseMinerNeuron):
    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)

        # Logging setup
        setup_logging(config=self.config)

        self._init_network_components()
        self._init_basic_components()
        self._init_model_components()

        # self._init_background_tasks()

    def _init_basic_components(self):
        """Initialize basic miner components and configurations."""

        # Device and ID setup
        self.device = self.config.neuron.device
        self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)

        # Progress tracking initialization
        self.local_progress = LocalTrainingProgress(
            peer_id=self.dht.peer_id.to_bytes(),
            epoch=0,
            samples_accumulated=0,
            samples_per_second=0.0,
            time=0.0,
            client_mode=False,
        )
        self.global_progress = GlobalTrainingProgress(epoch=0, samples_accumulated=0)
        self.global_progress.epoch = 10  # TODO Fix this
        self.local_progress.epoch = self.global_progress.epoch

        if self.global_progress.epoch is None:
            bt.logging.error(
                "Model Tag Is None. Make Sure You Are Using The Correct Model Name"
            )

        # Event tracking
        self.event = {}
        self.stop_event = threading.Event()
        
        # Initialize asyncio event loop
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        # Training control
        self.training_thread = None
        self.training_active = threading.Event()
        self.training_active.set()

        self.training_lock = asyncio.Lock()

        # Training status tracking
        self.training_status = TrainingStatus.STOPPED
        self.training_error = None
        self.training_thread = None

    def _init_model_components(self):
        """Initialize model-related components including tokenizer and optimizer settings."""
        # Tokenizer setup
        model_name = "distilgpt2"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Optimizer configurations
        self.learning_rate_maximum = 6e-4
        self.weight_decay = 0.1
        self.all_reduce_timeout = 360
        self.num_inner_steps = 500
        self.offload_optimizer = True  # DiLoCo Optimizer requires optimizer offloading

        # Model loading settings
        self.model_upload_retry_limit = 3
        self.model_upload_retry_delay = 6
        self.config.neuron.hf_repo_id = "kmfoda/gpt2-1b-miner-1"

        # Initialize model and its components
        # self.model_loading_manager = ModelLoadingManager() # TODO We dont need this anymore, right?
        load_model_optimizer_gradient_averager(self, self.global_progress.epoch)

        # Load initial state if needed # TODO This check should see if after loading states we are still on the same epoch
        # if self.local_progress.epoch != self.global_progress.epoch:
        #     load_state_from_peer(self, epoch=self.global_progress.epoch)

        # Initialize AveragingHandler for allreduce
        self.avg_handler = AveragingHandler(
            self.model, self.outer_optimizer, self.grad_averager, self.state_averager
        )

        # Initialize thread pool for background uploads
        self.upload_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="model_upload"
        )
        self.current_upload_future = None

    def _init_network_components(self):
        """Initialize network and DHT-related components."""
        # DHT initialization
        init_dht(self)

        # UID to PeerID mapping
        self.uids_to_peerids = {uid: None for uid in self.metagraph.uids.tolist()}

        # Wandb initialization if enabled
        if not self.config.neuron.dont_wandb_log:
            self.wandb = load_wandb(
                self, self.config, self.wallet, "miner", str(self.dht.peer_id)
            )

        # Log PeerID to chain
        bt.logging.info("Logging PeerID to chain")
        log_peerid_to_chain(self)

    def _init_background_tasks(self):
        """Initialize and start background tasks."""
        self.update_model_thread = threading.Thread(
            target=self.load_latest_model, daemon=True
        )
        self.update_model_thread.start()

    def upload_model(self, epoch, batch_size):
        """Unified function to save and upload both model and optimizer state"""
        attempt = 0
        while attempt < self.model_upload_retry_limit:
            try:
                with tempfile.TemporaryDirectory() as tmp_folder:
                    bt.logging.info(f"Saving model state locally for epoch {epoch}")
                    self.model.save_pretrained(tmp_folder)

                    bt.logging.info(
                        f"Uploading model and optimizer states to repo: {self.config.neuron.hf_repo_id}"
                    )
                    commit_message = f"Block {epoch}. Batch Size {batch_size}."
                    upload_folder(
                        folder_path=tmp_folder,
                        repo_id=self.config.neuron.hf_repo_id,
                        repo_type="model",
                        commit_message=commit_message,
                    )

                    # Create a tag for this version
                    create_tag(
                        self.config.neuron.hf_repo_id,
                        repo_type="model",
                        tag=str(epoch),
                        tag_message=commit_message,
                    )

                    bt.logging.info(
                        f"Successfully pushed new model and optimizer state with tag {epoch} to repo: {self.config.neuron.model_name}"
                    )
                    return True

            except Exception as e:
                attempt += 1
                bt.logging.warning(
                    f"Failed to upload state to HF hub, Retrying. Attempt {attempt}/{self.model_upload_retry_limit}. Error: {str(e)}"
                )
                if attempt < self.model_upload_retry_limit:
                    time.sleep(self.model_upload_retry_delay)
                else:
                    bt.logging.error(
                        "Maximum retry limit reached. Unable to upload state to HF Hub."
                    )
                    raise
        return False

    def start_background_upload(self, epoch, batch_size):
        """Starts a background upload of the model state, managing ongoing uploads."""
        # If there's an ongoing upload, check if it's done
        if self.current_upload_future and not self.current_upload_future.done():
            bt.logging.info("Previous upload still in progress, skipping new upload")
            return

        # Start new upload
        self.current_upload_future = self.upload_executor.submit(
            self.upload_model, epoch, batch_size
        )

        # Optional: Add callback to handle completion
        def upload_completed(future):
            try:
                future.result()  # This will raise any exceptions that occurred
                bt.logging.info("Validation state upload completed successfully")
            except Exception as e:
                bt.logging.error(f"Validation state upload failed: {str(e)}")

        self.current_upload_future.add_done_callback(upload_completed)

    def load_latest_model(self):
        while not self.stop_event.is_set():
            # Skip checking if we're currently loading
            if (self.model_loading_manager.is_loading) or (
                hasattr(self, "model") is False
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

    def start_continuous_training(self):
        """Starts continuous training in a background thread"""
        if self.training_thread is None or not self.training_thread.is_alive():
            self.training_status = TrainingStatus.RUNNING
            self.training_error = None
            self.training_thread = threading.Thread(
                target=self.continuous_training_loop,
                daemon=True,
                name="training_thread",
            )
            bt.logging.info(
                ":white_heavy_check_mark:Starting continuous training thread"
            )
            self.training_thread.start()

    def pause_training(self):
        """Pauses the continuous training loop"""
        self.training_active.clear()
        bt.logging.info("Pausing continuous training")

    def resume_training(self):
        """Resumes the continuous training loop"""
        self.training_active.set()
        bt.logging.info("Resuming continuous training")

    async def get_training_batch(self):
        """Gets a batch of training data"""
        block = self.block
        print("HERE..")
        pages = await DatasetLoader.next_pages(
            offset=block,
            n_pages=5,
            seed=self.uid if not self.config.random else random.randint(0, 1000),
        )
        random.shuffle(pages)
        print("HERE2..")
        dataset = await DatasetLoader.create(
            batch_size=self.config.neuron.local_batch_size_train,
            sequence_length=1024,
            pages_info=pages,
            tokenizer=self.tokenizer,
        )
        return dataset

    @handle_error(error_types=(TrainingError, Exception))
    def continuous_training_loop(self):
        """Main continuous training loop"""
        inner_step_counter = 0

        while not self.stop_event.is_set():
            try:
                # Wait if training is paused
                self.training_active.wait()

                # Get training batch using asyncio
                bt.logging.info("[magenta]Getting training batch..")
                dataset = asyncio.run_coroutine_threadsafe(
                    self.get_training_batch(), self.loop
                ).result()

                total_loss = 0
                batch_count = 0
                bt.logging.info("[magenta]Started for loop..")
                for batch in dataset:
                    if not self.training_active.is_set():
                        # Clean up gradients if interrupted
                        self.inner_optimizer.zero_grad()
                        break

                    # Convert batch to tensor and move to device
                    inputs = torch.tensor(batch).to(self.device)
                    labels = torch.tensor(batch).to(self.device)

                    # Forward pass
                    bt.logging.info("[magenta]Processing batch..")
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        outputs = self.model(input_ids=inputs, labels=labels)
                        loss = outputs[1]

                    # Backward pass
                    loss.backward()

                    # Update local progress
                    self.local_progress.samples_accumulated += inputs.size(0)
                    total_loss += loss.detach().item()
                    batch_count += 1
                    inner_step_counter += 1

                    # Log progress
                    if batch_count % 5 == 0:
                        bt.logging.info(
                            f":arrow_right:Inner Step: {inner_step_counter} | Average Loss: {total_loss / batch_count:.4f}"
                        )

                    # Upload to HuggingFace every 20 inner steps
                    if inner_step_counter % 20 == 0:
                        self.start_background_upload(
                            epoch=self.local_progress.epoch,
                            batch_size=self.config.neuron.local_batch_size_train,
                        )

                    # Optimizer step
                    self.inner_optimizer.step()
                    self.inner_optimizer.zero_grad()

                # Log wandb metrics if enabled
                if not self.config.neuron.dont_wandb_log:
                    self.event.update(
                        {
                            "loss": total_loss / batch_count if batch_count > 0 else 0,
                            "local_epoch": self.local_progress.epoch,
                            "global_epoch": self.global_progress.epoch,
                            "inner_steps": inner_step_counter,
                        }
                    )

            except TrainingError as e:
                self.training_status = TrainingStatus.ERROR
                bt.logging.error(f"Error in continuous training loop: {str(e)}")
                time.sleep(1)
                break

        if self.training_status != TrainingStatus.ERROR:
            self.training_status = TrainingStatus.STOPPED
        bt.logging.warning(
            f"Training thread exited. Status: {self.training_status.value}"
        )
        if self.training_error:
            bt.logging.error(f"Final error: {self.training_error}")

    async def all_reduce(
        self, synapse: distributed_training.protocol.AllReduce
    ) -> distributed_training.protocol.AllReduce:
        """Handle incoming all_reduce requests by pausing continuous training"""
        try:
            async with self.training_lock:
                # Ensure training is paused
                self.pause_training()
                bt.logging.info(
                    ":warning: Pausing continuous training for all_reduce query :warning:"
                )

                # Wait for running training process to finish # TODO Wait for training_thread == WAIT instead
                await asyncio.sleep(2)

                # Run allreduce with proper timeout
                result = await self.avg_handler.run_miner_allreduce(
                    synapse, timeout=synapse.timeout
                )
                return result

        except Exception as e:
            log_and_handle_error(e, "all_reduce operation failed")
            raise

        finally:
            # Resume training when done
            self.resume_training()
            bt.logging.succes("Resuming continuous training after all_reduce")

    async def blacklist_base(self, synapse) -> typing.Tuple[bool, str]:
        """
        Determines whether an incoming request should be blacklisted and thus ignored. Your implementation should
        define the logic for blacklisting requests based on your needs and desired security parameters.

        Blacklist runs before the synapse data has been deserialized (i.e. before synapse.data is available).
        The synapse is instead contructed via the headers of the request. It is important to blacklist
        requests before they are deserialized to avoid wasting resources on requests that will be ignored.

        Args:
            synapse (template.protocol.AllReduce): A synapse object constructed from the headers of the incoming request.

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


# This is the main function, which runs the miner.
if __name__ == "__main__":
    with Miner() as miner:
        while True:
            bt.logging.info("Miner running...", time.time())
            bt.logging.info(f"{miner.training_status.value}")
            time.sleep(5)
