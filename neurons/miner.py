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
import gc
import os
import queue
import random
import tempfile
import time
import typing
from enum import Enum

os.environ["NEST_ASYNCIO"] = "0"
import threading
from concurrent.futures import ThreadPoolExecutor

import bittensor as bt
import torch
from hivemind.averaging.averager import compute_schema_hash
from huggingface_hub import (
    create_repo,
    create_tag,
    delete_tag,
    list_repo_refs,
    repo_exists,
    upload_folder,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

import distributed_training
from distributed_training.averaging.avg_handler import AllReduceError, AveragingHandler
from distributed_training.base.miner import BaseMinerNeuron
from distributed_training.data.dataset import DatasetLoader
from distributed_training.utils.chain import log_peerid_to_chain
from distributed_training.utils.misc import (
    get_bandwidth,
    init_dht,
    load_wandb,
    setup_logging,
)
from distributed_training.utils.progress_tracker import (
    GlobalTrainingProgress,
    LocalTrainingProgress,
    get_global_epoch,
    get_local_epoch,
)
from distributed_training.utils.state_loader import (
    FastModelLoader,
    cleanup_old_cache,
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


class TrainingStatus(Enum):
    ERROR = "❗ | Error"
    RUNNING = "🏋️ | Training"
    STOPPED = "😴 | Stopped"
    AVERAGING = "🔄 | Averaging"


class Miner(BaseMinerNeuron):
    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)
        self._update_wandb_project()
        self._init_basic_components()
        self._init_model_components()
        self._init_network_components()

    def _update_wandb_project(self):
        suffix = "_miners" if self.neuron_type == "MinerNeuron" else "_validators"
        self.config.neuron.wandb_project += suffix

    def _init_basic_components(self):
        """Initialize basic miner components and configurations."""
        setup_logging(config=self.config)

        # Core setup
        self.device = self.config.neuron.device
        self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        init_dht(self)

        # Progress tracking
        self._init_progress_tracking()

        # Wandb setup
        if not self.config.neuron.dont_wandb_log:
            self.wandb = load_wandb(
                self, self.config, self.wallet, "miner", str(self.dht.peer_id)
            )

        # Training components
        self._init_training_components()

    def _init_progress_tracking(self):
        self.local_progress = LocalTrainingProgress(
            peer_id=self.dht.peer_id.to_bytes(),
            epoch=0,
            samples_accumulated=0,
            samples_per_second=0.0,
            time=0.0,
            client_mode=False,
            inner_step=0,
            loss=0.0,
        )
        self.global_progress = GlobalTrainingProgress(epoch=0, samples_accumulated=0)
        self.global_progress.epoch = get_global_epoch(self)
        self.local_progress.epoch = get_local_epoch(self)

        if self.global_progress.epoch is None:
            bt.logging.error(
                "Model Tag Is None. Make Sure You Are Using The Correct Model Name"
            )

    def _init_training_components(self):
        # Event tracking
        self.event = {}
        self.stop_event = threading.Event()

        # Training control
        self.training_thread = None
        self.training_active = threading.Event()
        self.training_active.set()

        # Queue and executor
        self.training_queue = queue.Queue()
        self.training_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="training_worker"
        )

        # Async components
        self.training_loop = asyncio.new_event_loop()
        self.training_lock = asyncio.Lock()

        # Status tracking
        self.training_status = TrainingStatus.STOPPED
        self.training_error = None

    def _init_model_components(self):
        """Initialize model-related components including tokenizer and optimizer settings."""
        self._init_tokenizer()
        self._setup_model_params()
        self._load_model()
        self._setup_training_params()

    def _init_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained("distilgpt2", use_fast=False)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def _setup_model_params(self):
        # Optimizer settings
        self.learning_rate_maximum = 6e-4
        self.weight_decay = 0.1
        self.num_inner_steps = 500
        self.offload_optimizer = True

        # Upload settings
        self.model_upload_retry_limit = 3
        self.model_upload_retry_delay = 6

    def _load_model(self):
        # Initialize loader
        self.loader = FastModelLoader(self.config.neuron.local_model_name)

        # Load model and components
        load_model_optimizer_gradient_averager(
            self, self.config.neuron.local_model_name, self.local_progress.epoch
        )
        cleanup_old_cache(self, repo_id=self.config.neuron.local_model_name)

        # Setup upload executor
        self.upload_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="model_upload"
        )
        self.current_upload_future = None

        # Sync and initialize handlers
        self._sync_with_global_model()

    def _setup_training_params(self):
        self.local_batch_size_train = self.config.neuron.local_batch_size_train
        self.local_batch_size_train_effective = (
            self.config.neuron.local_batch_size_train_effective
        )
        self.logging_interval = 5
        self.number_of_local_steps = (
            self.config.neuron.local_batch_size_train_effective
            // self.config.neuron.local_batch_size_train
        )

        self.running_loss = 0.0
        self.batch_count = 0

    def _init_network_components(self):
        """Initialize network and P2P components"""
        bt.logging.info("Logging PeerID to chain")
        log_peerid_to_chain(self)

    def _sync_with_global_model(self):
        global_model = AutoModelForCausalLM.from_pretrained(
            self.config.neuron.global_model_name,
            revision=str(self.global_progress.epoch),
            trust_remote_code=True,
        )

        if self.config.neuron.global_model_name == self.config.neuron.local_model_name:
            bt.logging.warning(
                "Your local miner_hf_repo_id set to the global model_name. This will harm your incentive. Set miner_hf_repo_id to a unique huggingface repo id."
            )

        self.model.to("cpu")
        should_sync_model = (
            (self.local_progress.epoch is None)
            or (self.local_progress.epoch != self.global_progress.epoch)
            or (
                compute_schema_hash(global_model.parameters())
                != compute_schema_hash(self.model.parameters())
            )
        )
        self.model.to(self.device)
        if should_sync_model:
            del global_model
            gc.collect()
            torch.cuda.empty_cache()
            load_state_from_peer(self, epoch=self.global_progress.epoch)
            self.start_background_upload(
                epoch=self.global_progress.epoch, inner_step=0, batch_size=0
            )
        else:
            del global_model
            gc.collect()
            torch.cuda.empty_cache()

    def upload_model(self, epoch, inner_step, batch_size):
        """Unified function to save and upload both model and optimizer state"""
        if not repo_exists(self.config.neuron.local_model_name, repo_type="model"):
            try:
                create_repo(
                    self.config.neuron.local_model_name,
                    repo_type="model",
                    private=False,
                )
                bt.logging.info(
                    f"Created new repository: {self.config.neuron.local_model_name}"
                )
            except Exception as e:
                bt.logging.error(f"Failed to create repository: {str(e)}")
                raise

        attempt = 0
        while attempt < self.model_upload_retry_limit:
            # Check if training is paused (i.e. all_reduce is happening)
            if not self.training_active.is_set():
                bt.logging.info("Upload Cancelled Due To AllReduce Operation")
                return False
            try:
                with tempfile.TemporaryDirectory() as tmp_folder:
                    bt.logging.info(
                        f":memory: Saving model state locally for epoch {epoch}"
                    )
                    self.model.config.inner_step = self.local_progress.inner_step
                    self.model.save_pretrained(tmp_folder)

                    bt.logging.info(
                        f":upload: Uploading model and optimizer states to repo: {self.config.neuron.local_model_name}"
                    )
                    commit_message = f"Outer Step {epoch}. Inner Step {inner_step}. Batch Size {batch_size}"
                    upload_folder(
                        folder_path=tmp_folder,
                        repo_id=self.config.neuron.local_model_name,
                        repo_type="model",
                        commit_message=commit_message,
                    )
                    refs = list_repo_refs(
                        self.config.neuron.local_model_name, repo_type="model"
                    )
                    for tag in refs.tags:
                        if (tag.name == "None") or (int(tag.name) >= epoch):
                            # Update tag for this version
                            delete_tag(
                                self.config.neuron.local_model_name,
                                repo_type="model",
                                tag=tag.name,
                            )
                            time.sleep(5)
                    # Create new tag for this version
                    create_tag(
                        self.config.neuron.local_model_name,
                        repo_type="model",
                        tag=str(epoch),
                        tag_message=commit_message,
                    )
                    # Cleanup old cache
                    cleanup_old_cache(
                        self,
                        repo_id=self.config.neuron.local_model_name,
                        current_revision=None,
                    )

                    bt.logging.info(
                        f"Successfully pushed new model state with tag {epoch} to repo: {self.config.neuron.local_model_name}"
                    )

                    # Reset block_list
                    self.model.config.block_list = []

                    return True

            except Exception as e:
                attempt += 1
                bt.logging.warning(
                    f":error: Failed to upload state to HF hub, Retrying. Attempt {attempt}/{self.model_upload_retry_limit}. Error: {str(e)}"
                )
                if attempt < self.model_upload_retry_limit:
                    time.sleep(self.model_upload_retry_delay)
                else:
                    bt.logging.error(
                        "Maximum retry limit reached. Unable to upload state to HF Hub."
                    )
                    raise
        return False

    def start_background_upload(self, epoch, inner_step, batch_size):
        """Starts a background upload of the model state, managing ongoing uploads."""
        # If there's an ongoing upload, check if it's done
        if self.current_upload_future and not self.current_upload_future.done():
            bt.logging.info("Previous upload still in progress, skipping new upload")
            return

        # Start new upload
        self.current_upload_future = self.upload_executor.submit(
            self.upload_model, epoch, inner_step, batch_size
        )

        # Optional: Add callback to handle completion
        def upload_completed(future):
            try:
                future.result()  # This will raise any exceptions that occurred
                bt.logging.info("Model state upload completed successfully")
            except Exception as e:
                bt.logging.error(f"Model state upload failed: {str(e)}")

        self.current_upload_future.add_done_callback(upload_completed)

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
        """Starts continuous training using the ThreadPoolExecutor"""
        if self.training_status != TrainingStatus.RUNNING:
            self.training_status = TrainingStatus.RUNNING
            self.training_error = None
            self.training_executor.submit(self._training_worker)
            bt.logging.info(
                ":white_heavy_check_mark: Starting continuous training worker"
            )

    def pause_training(self):
        """Pauses the continuous training loop"""
        self.training_active.clear()
        self.training_status = TrainingStatus.AVERAGING
        bt.logging.info(":warning:  Pausing continuous training for AllReduce query")

    def resume_training(self):
        """Resumes the continuous training loop"""
        self.training_active.set()
        self.training_status = TrainingStatus.RUNNING
        bt.logging.info(":white_heavy_check_mark: Resuming continuous training..")

    async def fetch_training_data(self):
        """Async function to fetch training data"""
        try:
            pages = await DatasetLoader.next_pages(
                offset=self.current_block,
                n_pages=5,
                seed=self.uid,
            )
            random.seed(self.uid)
            random.shuffle(pages)

            dataset = await DatasetLoader.create(
                batch_size=self.config.neuron.local_batch_size_train,
                sequence_length=1024,
                pages_info=pages,
                tokenizer=self.tokenizer,
            )

            return dataset
        except Exception as e:
            bt.logging.error(f"Error fetching training data: {str(e)}")
            raise

    def _training_worker(self):
        """Worker function that runs in the ThreadPoolExecutor"""

        asyncio.set_event_loop(self.training_loop)

        while not self.stop_event.is_set():
            try:
                # Wait if training is paused
                self.training_active.wait()

                # Periodic model upload
                if (
                    len(self.model.config.block_list)
                    >= self.config.neuron.target_n_blocks
                ):
                    self.start_background_upload(
                        epoch=self.local_progress.epoch,
                        inner_step=self.local_progress.inner_step,
                        batch_size=self.local_progress.samples_accumulated,
                    )

                bt.logging.info(":pages: Fetching fineweb-edu pages")
                dataset = self.training_loop.run_until_complete(
                    self.fetch_training_data()
                )
                self.model.config.block_list.append(self.current_block)

                self._process_training_batch(dataset)
            except Exception as e:
                bt.logging.warning(f"Training Loop Failed with error: {e}")
                self.training_status = TrainingStatus.ERROR
                self.training_error = str(e)
                break

        self.training_status = TrainingStatus.STOPPED

    def _process_training_batch(self, dataset):
        """Process a single training batch"""

        for inputs, labels in dataset:
            if not self.training_active.is_set():
                self.inner_optimizer.zero_grad()
                break

            # Move to device
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, loss = self.model(input_ids=inputs, labels=labels)
                scaled_loss = loss / self.number_of_local_steps

            scaled_loss.backward()

            self.running_loss += loss.item()
            self.batch_count += 1
            self.local_progress.loss = self.running_loss / self.batch_count

            self.local_progress.samples_accumulated += self.local_batch_size_train

            if (
                self.local_progress.samples_accumulated
                % (self.logging_interval * self.local_batch_size_train)
                == 0
            ):
                bt.logging.info(
                    f":training:  Outer Step: {self.local_progress.epoch} | "
                    f"Inner Step: {self.local_progress.inner_step} | "
                    f"Average Loss: {self.local_progress.loss:.4f} | "
                    f"Micro Batches: [{self.local_progress.samples_accumulated}/{self.local_batch_size_train_effective}]"
                )

            if (
                self.local_progress.samples_accumulated
                >= self.local_batch_size_train_effective
            ):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.event.update(
                    {
                        "outer_step": self.local_progress.epoch,
                        "inner_step": self.local_progress.inner_step,
                        "loss": self.local_progress.loss,
                        "samples_accumulated": self.local_progress.samples_accumulated,
                    }
                )
                self.inner_optimizer.step()
                self.inner_optimizer.zero_grad()
                self.local_progress.inner_step += 1

                self.running_loss = 0.0
                self.batch_count = 0

                self.local_progress.samples_accumulated = 0

    async def all_reduce(
        self, synapse: distributed_training.protocol.AllReduce
    ) -> distributed_training.protocol.AllReduce:
        """Handle incoming all_reduce requests by pausing continuous training"""
        bt.logging.info("Received All Reduce Call")
        start_time = time.perf_counter()
        try:
            async with self.training_lock:
                # Cancel any ongoing upload
                if self.current_upload_future and not self.current_upload_future.done():
                    bt.logging.info(
                        "Cancelling Ongoing Model Upload For AllReduce Operation"
                    )
                    self.current_upload_future.cancel()

                # # Ensure training is paused
                self.pause_training()

                try:
                    # Run allreduce with proper timeout
                    synapse = await self.avg_handler.run_miner_allreduce(
                        synapse,
                        self.local_progress,
                        start_time,
                        # bandwidth
                    )
                    if not synapse.completion:
                        raise AllReduceError("AllReduce Failed, Loading Latest State")
                except Exception as e:
                    bt.logging.info(f"All Reduce Failed with error: {e}")
                    synapse.completion = False

        except Exception as e:
            synapse.completion = False
            raise AllReduceError(f"Unexpected error during AllReduce: {str(e)}") from e

        finally:
            # Reset inner_step
            self.local_progress.inner_step = 0
            # Update epoch if all_reduce was succsefull
            if synapse.completion is True:
                self.local_progress.epoch += 1
                bt.logging.info("AllReduce Operation Finished Succesfully")

            wait_time = (
                synapse.timeout
                + self.upload_state_duration
                + time.perf_counter()
                - start_time
            )
            bt.logging.info(
                f"Waiting {int(wait_time)} seconds until all nodes complete the all_reduce"
            )

            # Wait for the master validator to upload new global model
            while (time.perf_counter() - start_time) <= (
                synapse.timeout + self.upload_state_duration
            ):
                time.sleep(1)
            # Check if master validator has failed to all_reduce
            self.global_progress.epoch = get_global_epoch(self)
            if self.local_progress.epoch != self.global_progress.epoch:
                bt.logging.info(
                    f"Global Epoch Wasn't Updated After All Reduce. Resetting To Current Global Epoch: {self.global_progress.epoch}"
                )
                self.all_reduce_success_status = False
            else:
                # Resume training when done
                self.resume_training()

            return synapse

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
    Miner().run()
