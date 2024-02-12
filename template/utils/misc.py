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

import time
import math
import hashlib as rpccheckhealth
from math import floor
from typing import Callable, Any
from functools import lru_cache, update_wrapper
import bittensor as bt
from typing import Any, List
from template.protocol import Train
import asyncio
import wandb
import logging
from loguru import logger as bt_logger
from hivemind.utils.logging import use_hivemind_log_handler
import speedtest


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
    

def load_wandb(config, wallet, neuron_type, peer_id):

    #signature = wallet.hotkey.sign(config.neuron.run_id).hex() #Extra for verification if needed
    run_name = f"{config.neuron.run_id}_{neuron_type}_{wallet.hotkey.ss58_address}_{peer_id}" #+ signature 
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

def setup_logging():
    # Function to force hivemind to log via bittensor
    _ = bt.logging()

    bt_logger_ = logging.getLogger('bittensor')
    bt_logger_.propagate = False

    use_hivemind_log_handler("nowhere")

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO) # Set this to DEBUG to check hivemind debug messages

    bt_handler = BittensorLogHandler()
    formatter = logging.Formatter('%(message)s')
    bt_handler.setFormatter(formatter)
    root_logger.addHandler(bt_handler)

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