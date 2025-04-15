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
import logging
import os
import re
import shutil
import time
from functools import lru_cache, update_wrapper
from ipaddress import ip_address
from logging.handlers import RotatingFileHandler
from math import floor
from typing import Any, Callable, List

import bittensor as bt
import hivemind
import speedtest
from bittensor.utils.btlogging import format
from dotenv import load_dotenv
from hivemind import utils
from hivemind.utils.logging import use_hivemind_log_handler

import wandb
from distributed_training.protocol import AllReduce
from distributed_training import __run__, __version__

EVENTS_LEVEL_NUM = 38
DEFAULT_LOG_BACKUP_COUNT = 10

load_dotenv()


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
        self, uids: List[int], queries: List[AllReduce], timeout: float = 150.0
    ):
        def call_single_uid(uid, query):
            return self.dendrite(
                self.metagraph.axons[uid], synapse=query, timeout=timeout
            )

        async def query_async():
            corutines = [
                call_single_uid(uid, query) for uid, query in zip(uids, queries)
            ]
            return await asyncio.gather(*corutines)

        return await query_async()


def load_wandb(self, config, wallet, neuron_type, peer_id):
    run_name = f"{neuron_type[0].upper()}{'{:03}'.format(self.uid)}"

    tags = [peer_id, __version__, self.wallet.hotkey.ss58_address, f"run{__run__}"]

    run_id = "_".join([run_name] + tags[2:]).lower()

    wandb_run = wandb.init(
        id=run_id,
        name=run_name,
        anonymous="allow",
        resume="allow",
        tags=tags,
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
    """Handler that routes log messages through bittensor's logging system"""

    def __init__(self):
        super().__init__()
        self.bt_logger = logging.getLogger("bittensor")

    def emit(self, record):
        try:
            msg = self.format(record)
            level_map = {
                logging.CRITICAL: self.bt_logger.critical,
                logging.ERROR: self.bt_logger.error,
                logging.WARNING: self.bt_logger.warning,
                logging.INFO: self.bt_logger.info,
                logging.DEBUG: self.bt_logger.debug,
                logging.TRACE: self.bt_logger.trace,
            }

            log_method = level_map.get(record.levelno, self.bt_logger.info)
            log_method(msg)

        except Exception:
            self.handleError(record)


def hive_log_filter(record):
    if (
        (record.name != "hivemind.dht.protocol")
        and (record.name != "hivemind.optim.progress_tracker")
        and (record.name != "hivemind.p2p.p2p_daemon_bindings.control")
    ):
        return True
    else:
        return False


def setup_logging(
    local_logfile="logs_mylogfile.txt",
    config=None,  # Add config parameter
):
    """Sets up comprehensive logging including bittensor, hivemind, and events logging"""

    # Initialize bittensor logging with extra emoji use
    format.emoji_map.update(
        {
            ":rocket:": "🚀",
            ":lock:": "🔒",
            ":unlock:": "🔓",
            ":lightning:": "⚡",
            ":error:": "❗",
            ":info:": "ℹ️",
            ":idle:": "😴",
            ":network:": "🌐",
            ":memory:": "💾",
            ":training:": "🏋️",
            ":progress:": "📈",
            ":wait:": "⏳",
            ":clock:": "⏱️",
            ":signal:": "📶",
            ":upload:": "🔼",
            ":broadcast:": "📡",
            ":sync:": "🔄",
            ":send:": "📤",
            ":receive:": "📥",
            ":pages:": "📑",
        }
    )
    # Change formatting of bt.debug messages
    bt_level = logging.INFO
    if config and hasattr(config, "logging"):
        if config.logging.debug:
            bt_level = logging.DEBUG
        elif config.logging.trace:
            bt_level = logging.TRACE
        elif config.logging.info:
            bt_level = logging.INFO

    if bt_level > logging.DEBUG:
        from bittensor.utils.btlogging.format import (
            LOG_FORMATS,
            Fore,
            Style,
            log_level_color_prefix,
        )

        for level, color in log_level_color_prefix.items():
            LOG_FORMATS[level] = (
                f"{Fore.BLUE}%(asctime)s{Fore.RESET} | "
                f"{Style.BRIGHT}{color}%(levelname)s{Style.RESET_ALL} | "
                f"%(message)s"
            )

    # Initialize bittensor logging
    if config:
        bt.logging(config=config)
    else:
        bt.logging()

    # Disable third party loggers in bittensor's queue system
    bt.logging.debug("Disabling third party loggers from bittensor queue...")
    bt.logging.disable_third_party_loggers()

    bt_level = logging.INFO
    if config and hasattr(config, "logging"):
        if config.logging.debug:
            bt_level = logging.DEBUG
        elif config.logging.trace:
            bt_level = logging.TRACE
        elif config.logging.info:
            bt_level = logging.INFO

    # if bt_level > logging.DEBUG:
    #     from bittensor.utils.btlogging.format import LOG_FORMATS, Fore, Style

    #     for level in LOG_FORMATS:
    #         # Simplify bt formatting for logging.INFO
    #         LOG_FORMATS[
    #             level
    #         ] = f"{Fore.BLUE}%(asctime)s{Fore.RESET} | {Style.BRIGHT}%(levelname)s{Style.RESET_ALL} | %(message)s"

    # Setup bittensor logger
    bt_logger = logging.getLogger("bittensor")
    bt_logger.setLevel(bt_level)
    bt_logger.propagate = False

    use_hivemind_log_handler("nowhere")

    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Setup BittensorLogHandler
    bt_handler = BittensorLogHandler()
    formatter = logging.Formatter("%(message)s")
    bt_handler.setFormatter(formatter)
    root_logger.addHandler(bt_handler)

    # Handle local file logging
    if os.path.exists(local_logfile):
        shutil.copyfile(local_logfile, local_logfile.replace(".txt", "_archive.txt"))
        os.remove(local_logfile)

    # Setup hivemind logger
    hivemind_logger = logging.getLogger("hivemind")
    hivemind_logger.handlers.clear()
    hivemind_logger.setLevel(logging.DEBUG)
    hivemind_logger.propagate = False

    file_handler = logging.FileHandler(local_logfile)
    file_handler.setLevel(logging.DEBUG)
    file_handler.addFilter(hive_log_filter)
    hivemind_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(hivemind_formatter)
    hivemind_logger.addHandler(file_handler)
    hivemind_logger.propagate = False

    # Get all existing loggers and ensure they don't propagate
    for name, logger in logging.root.manager.loggerDict.items():
        if isinstance(logger, logging.Logger):
            if name not in ["bittensor"]:
                logger.propagate = False
                if not any(isinstance(h, RotatingFileHandler) for h in logger.handlers):
                    logger.addHandler(file_handler)


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
        bandwidth_dict[f"all_reduce/{key}"] = float(f"{results[key] / 1e6:.2f}")

    return bandwidth_dict


def init_dht(self):
    # Init DHT and model
    if self.config.dht.ip:
        version = "4"
        address = self.config.dht.ip
        announce_maddrs = [f"/ip{version}/{address}/tcp/{self.config.dht.port}"]
    else:
        address = bt.utils.networking.get_external_ip()
        bt.logging.info(f"Received public IP address of this machine: {address}")
        version = ip_address(address).version
        announce_maddrs = [f"/ip{version}/{address}/tcp/{self.config.dht.port}"]

    # Init list of available DHT addresses from wandb
    api = wandb.Api()
    initial_peers_list = self.config.neuron.initial_peers

    validator_runs = api.runs(
        f"{self.config.neuron.wandb_entity}/{self.config.neuron.wandb_project.replace('_validators','').replace('_miners','')}_validators"
    )
    for ru in validator_runs:
        if ru.state == "running":
            if "dht_addresses" not in ru.config["neuron"].keys():
                continue
            else:
                for peer in ru.config["neuron"]["dht_addresses"]:
                    if peer not in initial_peers_list:
                        initial_peers_list.append(peer)

    miner_runs = api.runs(
        f"{self.config.neuron.wandb_entity}/{self.config.neuron.wandb_project.replace('_validators','').replace('_miners','')}_miners"
    )
    for ru in miner_runs:
        if ru.state == "running":
            if "dht_addresses" not in ru.config["neuron"].keys():
                continue
            else:
                for peer in ru.config["neuron"]["dht_addresses"]:
                    if peer not in initial_peers_list:
                        initial_peers_list.append(peer)

    # Init DHT
    retries = 0
    buffer = 5
    max_retries = buffer * len(initial_peers_list)
    successful_connection = False
    while successful_connection is False:
        if (retries == max_retries) and (successful_connection is False):
            raise Exception("Max retries reached, operation failed.")
        for attempt in range(0, buffer):
            for initial_peer in initial_peers_list:
                try:
                    # Init DHT
                    self.dht = hivemind.DHT(
                        host_maddrs=[
                            f"/ip4/0.0.0.0/tcp/{self.config.dht.port}",
                            f"/ip4/0.0.0.0/udp/{self.config.dht.port}/quic",
                        ],
                        initial_peers=[initial_peer],
                        announce_maddrs=announce_maddrs,
                        start=True,
                    )
                    bt.logging.info(
                        f"Successfully initialised dht using initial_peer as {initial_peer}"
                    )
                    successful_connection = True
                    utils.log_visible_maddrs(
                        self.dht.get_visible_maddrs(), only_p2p=True
                    )
                    # Add DHT address to wandb config
                    self.config.neuron.dht_addresses = [
                        re.sub(
                            "ip4/?(.*?)/",
                            f"ip{version}/{address}/",
                            str(addr),
                            flags=re.DOTALL,
                        )
                        for addr in self.dht.get_visible_maddrs()
                    ]
                    return
                except Exception as e:
                    bt.logging.error(
                        f"Attempt {retries + 1} to init DHT using initial_peer as {initial_peer} failed with error: {e}"
                    )
                    retries += 1
                    time.sleep(5)
                    bt.logging.error("Retrying...")
