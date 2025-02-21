import copy
import gc
import os
import subprocess
import sys
import tempfile
import threading
import time
from functools import partial
from pathlib import Path
from typing import Optional

import bittensor as bt
import hivemind
import psutil
import torch
from hivemind.compression import deserialize_torch_tensor
from hivemind.proto import averaging_pb2
from hivemind.utils import get_logger
from hivemind.utils.asyncio import aiter_with_timeout
from hivemind.utils.streaming import combine_from_streaming
from huggingface_hub import (
    create_tag,
    hf_hub_download,
    list_repo_refs,
    list_repo_files,
    scan_cache_dir,
    upload_folder,
)
from huggingface_hub.utils import (
    HfHubHTTPError,
    RepositoryNotFoundError,
    EntryNotFoundError,
)
from transformers import AutoModelForCausalLM

from distributed_training.averaging.averagers import DTGradAverager, DTStateAverager
from distributed_training.utils.progress_tracker import get_global_epoch

hivemind_logger = get_logger(__name__)


class ModelLoadingManager:
    def __init__(self):
        self.loading_lock = threading.Lock()
        self._is_loading = False
        self._last_loaded_epoch = None

    @property
    def is_loading(self):
        with self.loading_lock:
            return self._is_loading

    @property
    def last_loaded_epoch(self):
        with self.loading_lock:
            return self._last_loaded_epoch

    def set_loading_state(self, is_loading, epoch=None):
        with self.loading_lock:
            self._is_loading = is_loading
            if not is_loading and epoch is not None:
                self._last_loaded_epoch = epoch


class FastModelLoader:
    def __init__(self, model_name: str, cache_dir: str = None):
        """
        Initialize the fast model loader with HF downloader integration.

        Args:
            model_name (str): The HuggingFace model name (e.g., 'organization/model-name')
            cache_dir (str, optional): Directory to store downloaded files. Defaults to HF cache.
        """
        self.model_name = model_name
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/huggingface/hub")
        self._downloaded_files = {}  # Cache of downloaded files

    def download_files(self, revision: str = None, files: list = None):
        """
        Download files using hfdownloader.

        Args:
            revision (str, optional): Git revision/epoch number
            files (list, optional): List of specific files to download with patterns

        Returns:
            str: Path to downloaded files
        """
        # Generate cache key
        cache_key = f"{revision}_{','.join(files) if files else 'default'}"

        # Check if we already downloaded these files
        if cache_key in self._downloaded_files:
            return self._downloaded_files[cache_key]

        model_path = os.path.join(self.cache_dir, self.model_name.replace("/", "_"))
        os.makedirs(model_path, exist_ok=True)

        cmd = [
            "hfdownloader",
            "-r",
            self.model_name,
            "download",
            "-c",
            "10",
            "-y",
        ]

        if revision:
            cmd.extend(["-b", revision])

        # Add file patterns if specified, otherwise default to both model and optimizer
        if files:
            for file_pattern in files:
                cmd.extend(["-f", f"{file_pattern}"])
        else:
            cmd.extend(
                [
                    "-f",
                    "*.safetensors",
                    "-f",
                    "optimizer.pt",
                    "--skip-verify",
                ]
            )

        bt.logging.debug(f"Executing hfdownloader command: {' '.join(cmd)}")

        try:
            process = subprocess.Popen(
                cmd,
                cwd=model_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True,
                env={
                    **os.environ,
                    "PYTHONUNBUFFERED": "1",
                },  # Force Python unbuffered output
            )

            # Use select to handle both stdout and stderr
            import select

            outputs = [process.stdout, process.stderr]
            while True:
                # Wait for output on either stdout or stderr
                readable, _, _ = select.select(outputs, [], [])

                for output in readable:
                    line = output.readline()
                    if line:
                        # Don't buffer the print
                        print(line.rstrip(), flush=True)

                # Check if process has finished
                if process.poll() is not None:
                    break

            # Get any remaining output
            remaining_stdout, remaining_stderr = process.communicate()
            if remaining_stdout:
                print(remaining_stdout.rstrip(), flush=True)
            if remaining_stderr:
                print(
                    f"Error: {remaining_stderr.rstrip()}", file=sys.stderr, flush=True
                )

            if process.returncode != 0:
                raise RuntimeError(
                    f"hfdownloader failed with return code {process.returncode}"
                )

        except Exception as e:
            bt.logging.error(f"Download failed! Error: {str(e)}")
            raise RuntimeError(f"hfdownloader failed: {str(e)}")

        return model_path

    def load_model_and_optimizer(self, epoch: int = None):
        """
        Load both model and optimizer states in a single download operation.

        Args:
            epoch (int, optional): Epoch number for specific revision

        Returns:
            tuple: (model_state_dict, optimizer_state_dict)
        """
        revision = str(epoch) if epoch is not None else None

        # Download both model and optimizer files in one go
        model_path = self.download_files(revision=revision)

        # Load model state
        model_files = list(Path(model_path).rglob("*.safetensors"))
        if not model_files:
            raise FileNotFoundError(f"No model files found in {model_path}")

        bt.logging.info(f"Loading model state from: {[f.name for f in model_files]}")

        state_dict = {}
        for model_file in model_files:
            from safetensors.torch import load_file

            state = load_file(model_file)
            state_dict.update(state)

        # Load optimizer state
        optimizer_file = Path(model_path) / "optimizer.pt"
        if not optimizer_file.exists():
            raise FileNotFoundError(f"Optimizer state not found at {optimizer_file}")

        bt.logging.info(f"Loading optimizer state from: {optimizer_file}")
        optimizer_state = torch.load(str(optimizer_file), map_location="cpu")

        return state_dict, optimizer_state

def check_model_exists(repo_id: str, revision: Optional[str] = None) -> bool:
    try:
        if revision and revision != "None":
            list_repo_files(repo_id, revision=revision)
        else:
            list_repo_files(repo_id)
        return True
    except (RepositoryNotFoundError, EntryNotFoundError):
        return False
    
def load_model_optimizer_gradient_averager(
    self, model_name, epoch, use_fast_loader=False
):
    bt.logging.debug(
        f"CPU Memory Before Loading State {psutil.virtual_memory().available / 10**9} GB"
    )
    # Delete existing model
    if hasattr(self, "model"):
        del self.model

        gc.collect()
        torch.cuda.empty_cache()

    if use_fast_loader:
        try:
            # Load both model and optimizer states
            model_state, optimizer_state = self.loader.load_model_and_optimizer(
                epoch=epoch
            )

            # Create model instance and load state
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, state_dict=model_state, trust_remote_code=True
            )

            # Move model to device
            self.model = self.model.to(self.device)
            self.model.config.block_list = []

            # Set inner step
            self.local_progress.inner_step = (
                self.model.config.inner_step
                if "inner_step" in self.model.config.__dict__
                else 0
            )

            # Handle optimizer initialization/loading
            if hasattr(self, "opt"):
                self.inner_optimizer.param_groups = self.model.parameters()
            else:
                self.inner_optimizer = torch.optim.AdamW(
                    self.model.parameters(),
                    lr=optimizer_state.get("learning_rate", self.learning_rate_maximum),
                    betas=(0.9, 0.95),
                    eps=1e-8,
                    weight_decay=0.1,
                )

            # Load optimizer state if available
            if "optimizer_state_dict" in optimizer_state:
                self.inner_optimizer.load_state_dict(
                    optimizer_state["optimizer_state_dict"]
                )

            del optimizer_state
            del model_state
            gc.collect()
            torch.cuda.empty_cache()

            bt.logging.info(
                f"Successfully loaded model and optimizer using fast loader for epoch {epoch}"
            )

        except Exception as e:
            bt.logging.error(
                f"Fast loader failed: {str(e)}. Falling back to standard loading."
            )
            use_fast_loader = False  # TODO Set up fall back on using already downloaded model_state/opt_state, if either are missing

    if not use_fast_loader:

        if check_model_exists(model_name, revision=str(epoch)):
            try:
                self.model = (
                    AutoModelForCausalLM.from_pretrained(
                        model_name, revision=str(epoch), trust_remote_code=True
                    )
                    if epoch
                    else AutoModelForCausalLM.from_pretrained(
                        model_name, trust_remote_code=True
                    )
                )
                bt.logging.info(f"Successfully loaded model from {model_name} with revision {epoch}")
                
            except Exception as e:
                bt.logging.warning(f"Failed to load model despite repo existing: {str(e)}")
                
                bt.logging.info("Fallback to loading from global repo")
                self.model = (
                        AutoModelForCausalLM.from_pretrained(
                            self.config.neuron.model_name, revision=str(epoch), trust_remote_code=True
                        )
                        if epoch
                        else AutoModelForCausalLM.from_pretrained(
                            self.config.neuron.model_name, trust_remote_code=True
                        )
                    )
                bt.logging.info("Successfully loaded global model")
       
        self.model = self.model.to(self.device)
        self.model.config.block_list = []
        self.local_progress.inner_step = (
            self.model.config.inner_step
            if "inner_step" in self.model.config.__dict__
            else 0
        )

        try:
            optimizer_state = torch.load(
                hf_hub_download(
                    repo_id=model_name,
                    filename="optimizer.pt",
                    revision=str(epoch),
                ),
                weights_only=True,
                map_location="cpu",
            )

            if hasattr(self, "opt"):
                self.inner_optimizer.param_groups = self.model.parameters()
            else:
                self.inner_optimizer = torch.optim.AdamW(
                    self.model.parameters(),
                    lr=optimizer_state["learning_rate"],
                    betas=(0.9, 0.95),
                    eps=1e-8,
                    weight_decay=0.1,
                )

            # Load optimizer state if available
            if "optimizer_state_dict" in optimizer_state:
                self.inner_optimizer.load_state_dict(
                    optimizer_state["optimizer_state_dict"]
                )

            del optimizer_state
            gc.collect()
            torch.cuda.empty_cache()

            bt.logging.info(f"Successfully loaded optimizer state for epoch {epoch}")

        except Exception as e:
            bt.logging.warning(
                f"No optimizer state found or failed to load: {str(e)}. Initializing fresh optimizer."
            )
            self.inner_optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate_maximum,
                betas=(0.9, 0.95),
                eps=1e-8,
                weight_decay=0.1,
            )

    # Clean up
    gc.collect()
    torch.cuda.empty_cache()

    # Set outer optimizer
    self.outer_optimizer = partial(torch.optim.SGD, lr=0.7, momentum=0.9, nesterov=True)

    # Delete existing gradient averager
    if hasattr(self, "state_averager"):
        for i in self.state_averager.main_parameters:
            del i
        gc.collect()
        torch.cuda.empty_cache()

        # Reset gradient buffers and parameters
        self.state_averager.main_parameters = tuple(self.model.parameters())
        (
            param_groups,
            self.state_averager.main_parameters,
            self.state_averager.parameter_names,
        ) = self.state_averager._check_params(
            self.outer_optimizer, tuple(self.model.parameters()), parameter_names=None
        )
        self.state_averager._averaged_parameters = (
            self.state_averager._make_averaged_parameters(
                self.state_averager.main_parameters
            )
        )
        (
            self.state_averager.optimizer,
            self.state_averager.scheduler,
        ) = self.state_averager._init_components(
            param_groups,
            self.outer_optimizer,
            scheduler_or_factory=None,
            initialize_optimizer=True,
        )

    else:
        self.state_averager = DTStateAverager(
            dht=self.dht,
            prefix=f"{self.config.neuron.run_id}_state_averager",
            optimizer=self.outer_optimizer,
            params=self.model.parameters(),
            initialize_optimizer=True,
            offload_optimizer=self.offload_optimizer,
            custom_gradients=self.offload_optimizer,
            min_group_size=self.config.neuron.min_group_size,
            min_matchmaking_time=30.0,
            request_timeout=10.0,
            next_chunk_timeout=45.0,
            allreduce_timeout=self.allreduce_timeout - 30.0 - 15.0,
            start=True,
        )

    # Delete existing gradient averager
    if hasattr(self, "grad_averager"):
        for i in self.grad_averager.main_parameters:
            del i
        gc.collect()
        torch.cuda.empty_cache()

        # Reset gradient buffers and parameters
        self.grad_averager.main_parameters = tuple(self.model.parameters())
        self.grad_averager.offloaded_optimizer = self.state_averager.optimizer
        self.grad_averager._averaged_tensors = tuple(
            grad for grad in self.grad_averager._grads_from_optimizer()
        )

    else:
        # Load a new gradient averager
        self.grad_averager = DTGradAverager(
            dht=self.dht,
            main_parameters=self.state_averager.main_parameters,
            offloaded_optimizer=self.state_averager.optimizer,
            prefix=f"{self.config.neuron.run_id}_grad_averager",
            compression=hivemind.Uniform8BitQuantization(),
            state_compression=hivemind.Uniform8BitQuantization(),
            min_group_size=self.config.neuron.min_group_size,
            min_matchmaking_time=30.0,
            request_timeout=10.0,
            next_chunk_timeout=45.0,
            allreduce_timeout=self.allreduce_timeout - 30.0 - 15.0,
            start=True,
        )

    bt.logging.debug(
        f"CPU Memory After Loading State {psutil.virtual_memory().available / 10**9} GB"
    )


def load_state_from_peer(self, epoch=None):
    try:
        state_loaded = False
        if epoch is None:
            self.global_progress.epoch = get_global_epoch(self)
            epoch = self.global_progress.epoch

        bt.logging.debug("Model Weights Before Loading State")
        current_model_weights_sample = copy.copy(
            [layer for layer in self.model.parameters()][-2][-10:].tolist()
        )
        bt.logging.debug(current_model_weights_sample)

        bt.logging.debug(f"Old Model Tag: {self.local_progress.epoch}")

        if self.global_progress.epoch is not None:
            bt.logging.debug(
                f"Latest Model State Found On The HF Hub With The Tag: {self.global_progress.epoch}. Loading That Model State."
            )

            # Load model state with max retries
            MAX_ATTEMPTS = 3
            attempt = 0

            while attempt < MAX_ATTEMPTS:
                try:
                    load_model_optimizer_gradient_averager(
                        self, self.config.neuron.model_name, epoch
                    )
                    break

                except Exception as e:
                    attempt += 1
                    if attempt == MAX_ATTEMPTS:
                        raise Exception(
                            f"Failed to load model after {MAX_ATTEMPTS} attempts: {str(e)}"
                        )
                    bt.logging.warning(
                        f"Failed to load model, retrying. Attempt {attempt}/{MAX_ATTEMPTS}. Error {str(e)}"
                    )

            state_loaded = True

            bt.logging.debug("Model Weights After Loading State")
            new_model_weights_sample = copy.copy(
                [layer for layer in self.model.parameters()][-2][-10:].tolist()
            )
            bt.logging.debug(new_model_weights_sample)

            self.local_progress.epoch = self.global_progress.epoch
            self.local_progress.inner_step = 0
            self.local_progress.samples_accumulated = 0
            bt.logging.debug(f"New Model Tag: {self.global_progress.epoch}")

            # Clean up old cache
            try:
                cleanup_old_cache(self)
            except Exception as e:
                bt.logging.warning(f"Failed to cleanup cache: {str(e)}")

        else:
            bt.logging.debug(f"Model With Tag: {epoch} Does Not Exist")

        return state_loaded

    except Exception as e:
        bt.logging.error(f"Error loading state: {str(e)}")
        return False


# TODO Remove this if score_bandwidth is deprecated
async def load_state_from_miner(self, peer, timeout: Optional[float] = None):
    metadata = None
    hivemind_logger.info(f"Downloading parameters from peer {peer}")
    try:
        stub = self.grad_averager.get_stub(
            self._p2p,
            peer,
            namespace=self.grad_averager.matchmaking_kwargs["prefix"],
        )
        stream = await stub.rpc_download_state_partial(averaging_pb2.DownloadRequest())
        current_tensor_parts, tensors = [], []

        # TODO merge this with hivemind.compression.deserialize_tensor_stream
        async for message in aiter_with_timeout(stream, timeout=timeout):
            if message.metadata:
                metadata = self.grad_averager.serializer.loads(message.metadata)
            if message.tensor_part.dtype and current_tensor_parts:
                # tensor_part.dtype indicates the start of the new tensor, so we should wrap up this one
                tensors.append(
                    deserialize_torch_tensor(
                        combine_from_streaming(current_tensor_parts)
                    )
                )
                current_tensor_parts = []
            current_tensor_parts.append(message.tensor_part)
        if current_tensor_parts:
            tensors.append(
                deserialize_torch_tensor(combine_from_streaming(current_tensor_parts))
            )

        if not metadata:
            hivemind_logger.exception(f"Peer {peer} did not send its state")
            return

        hivemind_logger.info(f"Finished downloading state from {peer}")
        return metadata, tensors
    except Exception as e:
        hivemind_logger.exception(f"Failed to download state from {peer} - {repr(e)}")
        return None, None


def cleanup_old_cache(self, repo_id=None, current_revision=None):
    """Helper method to clean up old cache files"""
    if repo_id is None:
        repo_id = self.config.neuron.model_name
        current_revision = self.model.config._commit_hash

    cache_info = scan_cache_dir()
    bt.logging.info("Cache clearing warnings:")
    bt.logging.info(f"{cache_info.warnings}")

    for repo in cache_info.repos:
        if repo.repo_id == repo_id:
            revisions = sorted(
                repo.revisions, key=lambda r: r.last_modified, reverse=True
            )

            bt.logging.info(
                f"Found {len(revisions)} model revisions in .cache folder. Proceeding to delete all non-current revision."
            )
            for revision in revisions:
                if (current_revision is not None) and (
                    revision.commit_hash == current_revision
                ):
                    bt.logging.info(
                        f"Skipping cache for current revision {revision.commit_hash}"
                    )
                    continue
                else:
                    bt.logging.info(
                        f"Deleting cache for revision {revision.commit_hash}"
                    )
                    cache_info.delete_revisions(revision.commit_hash).execute()
            break


def upload_new_state(self, epoch: int, results: dict, block: int = None):
    attempt = 0
    while attempt < self.model_upload_retry_limit:
        try:
            bt.logging.info(
                f"Pushing new model and optimizer state to HF Hub with tag {epoch}"
            )

            # Save and upload both model and optimizer state
            upload_success = save_and_upload_state(
                self, epoch=epoch, results=results, block=block
            )

            if upload_success:
                # Verify the upload
                updated_refs = list_repo_refs(
                    self.config.neuron.model_name,
                    repo_type="model",
                )
                new_tag = max([int(tag.name) for tag in updated_refs.tags])
                bt.logging.info(f"Successfully pushed new model with tag {new_tag}")
                # Wait to allow out of sync miners to download new model state
                time.sleep(self.load_state_timeout)
                break

        except HfHubHTTPError as e:
            attempt += 1
            bt.logging.info(f"{e}. Loading State from Peer.")
            state_loaded = load_state_from_peer(self, epoch=self.global_progress.epoch)
            if state_loaded:
                break
        except Exception:
            attempt += 1
            bt.logging.warning(
                f"Failed To Upload Model To HF hub, Retrying. Attempt {attempt}/{self.model_upload_retry_limit}."
            )
            if attempt < self.model_upload_retry_limit:
                time.sleep(self.model_upload_retry_delay)
            else:
                bt.logging.error(
                    "Maximum Retry Limit Reached. Unable To Upload Model To HF Hub."
                )
                raise
    return upload_success


def save_and_upload_state(self, epoch: int, results: dict, block: int = None):
    """Unified function to save and upload both model and optimizer state"""
    batch_size = sum(
        [result for result in results["gathered"].values() if result is not None]
    )
    participating_peers = results["participating_peers"]
    failed_peers = results["failed_peers"]
    attempt = 0
    while attempt < self.model_upload_retry_limit:
        try:
            with tempfile.TemporaryDirectory() as tmp_folder:
                bt.logging.info(
                    f"Preparing model and optimizer state for epoch {epoch}"
                )
                if block is not None:
                    self.model.config.last_allreduce_block = block
                self.model.save_pretrained(tmp_folder)
                # Save optimizer state
                optimizer_state = {
                    "optimizer_state_dict": self.state_averager.optimizer.state_dict(),
                    "learning_rate": self.learning_rate_maximum,
                    "epoch": epoch,
                }
                torch.save(optimizer_state, os.path.join(tmp_folder, "optimizer.pt"))

                bt.logging.info(
                    f"Uploading model and optimizer states to repo: {self.config.neuron.model_name}"
                )

                # Upload everything in one go
                commit_message = f"Epoch {epoch}. Batch Size {batch_size}. Peers {len(participating_peers) - len(failed_peers)}."
                upload_folder(
                    folder_path=tmp_folder,
                    repo_id=self.config.neuron.model_name,
                    repo_type="model",
                    commit_message=commit_message,
                )

                # Create a tag for this version
                create_tag(
                    self.config.neuron.model_name,
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
