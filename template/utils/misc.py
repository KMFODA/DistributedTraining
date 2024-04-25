# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 Opentensor Foundation

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
import functools
import logging
import re
import time
from functools import lru_cache, update_wrapper
from ipaddress import ip_address
from math import floor
from typing import Any, Callable, List

import bittensor as bt
import hivemind
import requests
import speedtest
import torch
import wandb
from hivemind import utils
from hivemind.utils.logging import use_hivemind_log_handler
from loguru import logger as bt_logger

from template.protocol import Train
from template.utils.chain_storage import run_in_subprocess
from datetime import datetime


# LRU Cache with TTL
def ttl_cache(maxsize: int = 128, typed: bool = False, ttl: int = -1):
    """
    Decorator that creates a cache of the most recently used function calls with a time-to-live (TTL) feature.
    The cache evicts the least recently used entries if the cache exceeds the `maxsize` or if an entry has
    been in the cache longer than the `ttl` period.

    Args:
        maxsize (int): Maximum size of the cache. Once the cache grows to this size, subsequent entries
                       replace the least recently used ones. Defaults to 128.
        typed (bool): If set to True, arguments of different types will be cached separately. For example,
                      f(3) and f(3.0) will be treated as distinct calls with distinct results. Defaults to False.
        ttl (int): The time-to-live for each cache entry, measured in seconds. If set to a non-positive value,
                   the TTL is set to a very large number, effectively making the cache entries permanent. Defaults to -1.

    Returns:
        Callable: A decorator that can be applied to functions to cache their return values.

    The decorator is useful for caching results of functions that are expensive to compute and are called
    with the same arguments frequently within short periods of time. The TTL feature helps in ensuring
    that the cached values are not stale.

    Example:
        @ttl_cache(ttl=10)
        def get_data(param):
            # Expensive data retrieval operation
            return data
    """
    if ttl <= 0:
        ttl = 65536
    hash_gen = _ttl_hash_gen(ttl)

    def wrapper(func: Callable) -> Callable:
        @lru_cache(maxsize, typed)
        def ttl_func(ttl_hash, *args, **kwargs):
            return func(*args, **kwargs)

        def wrapped(*args, **kwargs) -> Any:
            th = next(hash_gen)
            return ttl_func(th, *args, **kwargs)

        return update_wrapper(wrapped, func)

    return wrapper


def _ttl_hash_gen(seconds: int):
    """
    Internal generator function used by the `ttl_cache` decorator to generate a new hash value at regular
    time intervals specified by `seconds`.

    Args:
        seconds (int): The number of seconds after which a new hash value will be generated.

    Yields:
        int: A hash value that represents the current time interval.

    This generator is used to create time-based hash values that enable the `ttl_cache` to determine
    whether cached entries are still valid or if they have expired and should be recalculated.
    """
    start_time = time.time()
    while True:
        yield floor((time.time() - start_time) / seconds)


# 12 seconds updating block.
@ttl_cache(maxsize=1, ttl=12)
def ttl_get_block(self) -> int:
    """
    Retrieves the current block number from the blockchain. This method is cached with a time-to-live (TTL)
    of 12 seconds, meaning that it will only refresh the block number from the blockchain at most every 12 seconds,
    reducing the number of calls to the underlying blockchain interface.

    Returns:
        int: The current block number on the blockchain.

    This method is useful for applications that need to access the current block number frequently and can
    tolerate a delay of up to 12 seconds for the latest information. By using a cache with TTL, the method
    efficiently reduces the workload on the blockchain interface.

    Example:
        current_block = ttl_get_block(self)

    Note: self here is the miner or validator instance
    """
    return self.subtensor.get_current_block()

class AsyncDendritePool:
    def __init__(self, wallet, metagraph):
        self.metagraph = metagraph
        self.dendrite = bt.dendrite(wallet=wallet)
    
    async def async_forward(
            self,
            uids: List[int],
            queries: List[Train], 
            timeout: float = 150.0
    ):

        def call_single_uid(uid, query):
            return self.dendrite(
                self.metagraph.axons[uid],
                synapse=query,
                timeout=timeout
            )
        
        async def query_async():
            corutines = [call_single_uid(uid, query) for uid, query in zip(uids, queries)]
            return await asyncio.gather(*corutines)
        
        return await query_async()
    

def load_wandb(self, config, wallet, neuron_type, peer_id):

    #signature = wallet.hotkey.sign(config.neuron.run_id).hex() #Extra for verification if needed
    run_name = f"{config.neuron.run_id}_{neuron_type}_UID{self.uid}_{peer_id}" #+ signature 
    wandb_run = wandb.init(
        id = run_name,
        name=run_name,
        anonymous="allow",
        project=config.neuron.wandb_project,
        entity=config.neuron.wandb_entity,
        config=config,
        allow_val_change=True,
    )

    signature = wallet.hotkey.sign(config.neuron.run_id.encode()).hex()
    config.signature = signature
    wandb_run.config.update(config, allow_val_change=True)
    return wandb_run

class BittensorLogHandler(logging.Handler):
    def emit(self, record):
        log_entry = self.format(record)
        
        if record.levelno >= logging.CRITICAL:
            bt_logger.critical(log_entry)
        elif record.levelno >= logging.ERROR:
            bt_logger.error(log_entry)
        elif record.levelno >= logging.WARNING:
            bt_logger.warning(log_entry)
        elif record.levelno >= logging.INFO:
            bt_logger.info(log_entry)
        elif record.levelno >= logging.DEBUG:
            bt_logger.debug(log_entry)
        else:
            bt_logger.trace(log_entry)

def setup_logging(level=logging.INFO):
    # Function to force hivemind to log via bittensor
    _ = bt.logging()

    bt_logger_ = logging.getLogger('bittensor')
    bt_logger_.propagate = False

    use_hivemind_log_handler("nowhere")

    root_logger = logging.getLogger()
    root_logger.setLevel(level) # Set this to logging.DEBUG to check hivemind debug messages -> Careful, it's a lot of output

    bt_handler = BittensorLogHandler()
    formatter = logging.Formatter('%(message)s')
    bt_handler.setFormatter(formatter)
    root_logger.addHandler(bt_handler)

    # Create a file handler that logs debug and higher level messages
    
    fh = logging.FileHandler(f"logs_{datetime.now().strftime('mylogfile_%H_%M_%d_%m_%Y')}.txt")
    fh.setLevel(logging.DEBUG)

    # Create a formatter and set the formatter for the handler.
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)

    # Add the FileHandler to the root logger
    root_logger.addHandler(fh)

def get_bandwidth():
    # Get speedtest results
    s = speedtest.Speedtest()
    s.get_servers()
    s.get_best_server()
    s.download()
    s.upload()
    results = s.results.dict()
    
    # Copy key metrics to a formatted badnwidth_dict
    bandwidth_dict = {}
    keys = ["download", "upload", "ping"]
    for key in keys:
        bandwidth_dict[key] = float(f"{results[key]/ 1e6:.2f}")

    return bandwidth_dict

def init_dht(self):
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
            if "dht_addresses" not in ru.config["neuron"].keys():
                continue
            else:
                for peer in ru.config["neuron"]["dht_addresses"]:
                    if peer not in initial_peers_list:
                        initial_peers_list.append(peer)

    # Init DHT
    retries = 0
    buffer = 2
    max_retries = buffer * len(initial_peers_list)
    successful_connection = False
    while (retries <= max_retries) and (successful_connection is False):
        if (retries == max_retries) and (successful_connection is False):
            raise Exception("Max retries reached, operation failed.")
        for i in range(0, buffer):
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
                successful_connection = True
                break
            except Exception as e:
                bt.logging.error(
                    f"Attempt {retries + 1} to init DHT using initial_peer as {initial_peers_list[retries]} failed with error: {e}"
                )
                retries += 1
                time.sleep(5)
                bt.logging.error(f"Retrying...")

    utils.log_visible_maddrs(self.dht.get_visible_maddrs(), only_p2p=True)

    # Commit Peer Id to Subtensor
    # self.subtensor.commit(self.wallet, self.config.netuid, self.dht.peer_id.to_base58())
    # Wrap calls to the subtensor in a subprocess w ith a timeout to handle potential hangs.
    partial = functools.partial(
        self.subtensor.commit,
        self.wallet,
        self.config.netuid,
       self.dht.peer_id.to_base58(),
    )
    # try:
    #     run_in_subprocess(partial, 60)
    # except Exception as e:
    #     bt.logging.info(f"Error submitting Peer ID to chaing {Exception} retrying in 2 minutes")

    # Add DHT address to wandb config
    self.config.neuron.dht_addresses = [re.sub("ip4/?(.*?)/", f"ip{version}/{address}/", str(addr), flags=re.DOTALL) for addr in self.dht.get_visible_maddrs()]

# From: https://github.com/unconst/gradient/blob/main/neurons/validator.py#L53
def compute_losses(model: torch.nn.Module, batches: List[torch.Tensor], device: str = 'cpu') -> float:
    """
    Computes and returns the average loss of a model evaluated over a given set of batches.

    This function iterates through each batch, feeds it to the model, and accumulates the loss to compute
    the average loss across all batches. This is useful for evaluating the model's performance on a dataset.

    Args:
        model (torch.nn.Module): The model to be evaluated.
        batches (List[torch.Tensor]): A list of batches to evaluate the model on. Each batch is a torch.Tensor.
        device (str, optional): The device (e.g., 'cpu' or 'cuda') on which to perform the computations. Defaults to 'cpu'.

    Returns:
        float: The average loss computed over all the batches.

    Note:
        This function does not compute gradients and is typically used for model evaluation.

    Raises:
        ValueError: If `batches` is empty, raising a ValueError to indicate that no batches were provided for evaluation.
    """
    # Ensure there are batches to compute the loss on
    if not batches:
        bt.logging.error("No batches provided for loss computation.")
        raise ValueError("No batches provided for loss computation.")

    # Initialize total_loss to accumulate losses over batches
    total_loss: float = 0.0

    # Calculate the number of batches for averaging the loss later
    num_batches: int = len(batches)

    # Disable gradient computations for efficiency and to prevent model updates
    with torch.no_grad():
        for batch in batches:
            try:
                # Move the batch to the specified device (e.g., CPU or GPU)
                inputs: torch.Tensor = batch.to(device)
                # Forward pass: Compute the model's output and loss for the given inputs
                outputs = model(inputs, labels=inputs)
                # Accumulate the loss
                total_loss += outputs.loss.item()
            except Exception as e:
                bt.logging.error(f"Error during loss computation for a batch: {e}")
                raise Exception(f"Error during loss computation for a batch: {e}")

    # Compute the average loss across all batches
    try:
        average_loss: float = total_loss / num_batches
        # TODO Should we add same loss/ppl calculation as in sn9? https://github.com/RaoFoundation/pretraining/blob/d2faaec9737c8858cd22373140bd5db0ace02c4c/scripts/run_benchmarks.py#L37
        perplexity: float = torch.exp(torch.tensor(average_loss)).item()
    except ZeroDivisionError as e:
        bt.logging.error("Division by zero encountered while computing average loss. This should not happen.")
        raise ZeroDivisionError("Division by zero encountered while computing average loss.")

    # Log the computed average loss
    bt.logging.debug(f"Average loss computed successfully: {average_loss}")

    return average_loss, perplexity