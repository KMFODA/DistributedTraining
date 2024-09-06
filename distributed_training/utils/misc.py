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
from math import floor
from typing import Callable, Any
from functools import lru_cache, update_wrapper
import bittensor as bt
from typing import Any, List
from distributed_training.protocol import Train
import asyncio
import wandb
import logging
from loguru import logger as bt_logger
from hivemind.utils.logging import use_hivemind_log_handler
import speedtest
import hivemind
import logging_loki
from logging.handlers import QueueHandler, QueueListener
from multiprocessing import Queue
import json
from hivemind import utils
import re
from ipaddress import ip_address

import os
import shutil
import random
from distributed_training.data.dataset import DataLoader
from bitarray import bitarray
import wandb
from logtail import LogtailHandler
import os
from dotenv import load_dotenv
import asyncio

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
        self, uids: List[int], queries: List[Train], timeout: float = 150.0
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
    run_name = f"{config.neuron.run_id}_{neuron_type}_UID{self.uid}_{peer_id}"
    wandb_run = wandb.init(
        id=run_name,
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


class IpFilter(logging.Filter):
    """
    This is a filter which injects contextual information into the log.
    """

    def __init__(self, ip, port):
        self.ip = ip
        self.port = port

    def filter(self, record):
        record.host = f"{self.ip}:{self.port}"
        return True


class JSONFormatter(logging.Formatter):
    def __init__(
        self,
        network,
        netuid,
        hotkey,
        version,
        spec_version,
        run_id,
        ip,
        port,
        uid,
        neuron_type,
    ):
        self.network = network
        self.netuid = netuid
        self.hotkey = hotkey
        self.version = version
        self.spec_version = spec_version
        self.run_id = run_id
        self.ip = ip
        self.port = port
        self.uid = uid
        self.neuron_type = neuron_type

    def format(self, record):
        try:
            # TODO Cleanup
            # Extract real message from the noisy msg line emitted by bittensor
            msg = "".join(record.getMessage().split(" - ")[1:])
        except Exception:
            msg = record.getMessage()

        log_record = {
            "level": record.levelname.lower(),
            "module": record.module,
            "func_name": record.funcName,
            "thread": record.threadName,
            "netuid": self.netuid,
            "network": self.network,
            "neuron_type": self.neuron_type,
            "hotkey": self.hotkey,
            "uid": self.uid,
            "ip": self.ip,
            "port": self.port,
            "message": msg,
            "filename": record.filename,
            "lineno": record.lineno,
            "version": self.version,
            "spec_version": self.spec_version,
        }
        return json.dumps(log_record)


class LogHandler(logging_loki.LokiHandler):
    def handleError(self, record):
        self.emitter.close()
        # When Loki endpoint gives error for some reason,
        # parent .handleError starts spamming error trace for each failure
        # so we are disabling this default behaviour
        # super().handleError(record)


class CustomLokiLoggingHandler(QueueHandler):
    def __init__(self, queue: Queue, **kwargs):
        super().__init__(queue)
        self.handler = LogHandler(**kwargs)  # noqa: WPS110
        self.listener = QueueListener(self.queue, self.handler)
        self.listener.start()


class LogHandler(logging_loki.LokiHandler):
    def handleError(self, record):
        self.emitter.close()
        # When Loki endpoints gives error for unexplained reasons,
        # parent .handleError starts spamming error trace for each failure
        # so we are disabling this default behaviour for now
        # super().handleError(record)


def logging_filter(record):
    if (record.name != "hivemind.dht.protocol") and (
        record.name != "hivemind.optim.progress_tracker"
    ):
        return True
    else:
        return False


def add_loki_logger_handler(
    logger,
    network,
    netuid,
    hotkey,
    version,
    spec_version,
    run_id,
    ip,
    port,
    uid,
    neuron_type,
):
    """Configure sending logs to loki server"""

    # Use LokiQueueHandler to upload logs in background
    loki_handler = CustomLokiLoggingHandler(
        Queue(-1),
        url="https://logs-prod-006.grafana.net/loki/api/v1/push",
        tags={"application": "distributed_training"},
        auth=(
            "944477",
            os.environ["LOKI_KEY"],
        ),
        version="1",
    )

    # Send logs to loki as JSON
    loki_handler.setFormatter(
        JSONFormatter(
            network,
            netuid,
            hotkey,
            version,
            spec_version,
            run_id,
            ip,
            port,
            uid,
            neuron_type,
        )
    )

    logger.addHandler(loki_handler)


def setup_logging(
    network,
    netuid,
    hotkey,
    version,
    spec_version,
    run_id,
    uid,
    neuron_type,
    level=logging.INFO,
    ip=None,
    port=None,
    local_logfile="/root/logs_mylogfile.txt",
):
    # Function to force hivemind to log via bittensor
    _ = bt.logging()

    logtail_handler = LogtailHandler(source_token=os.getenv("LOGTAIL_KEY"))
    formatter = logging.Formatter("%(host)s%(message)s")
    logtail_handler.setFormatter(formatter)
    logtail_handler.addFilter(IpFilter(ip=ip, port=port))

    bt_logger_ = logging.getLogger("bittensor")
    bt_logger_.propagate = False
    bt_logger_.addHandler(logtail_handler)
    add_loki_logger_handler(
        bt_logger_,
        network,
        netuid,
        hotkey,
        version,
        spec_version,
        run_id,
        ip,
        port,
        uid,
        neuron_type,
    )

    use_hivemind_log_handler("nowhere")

    root_logger = logging.getLogger()
    root_logger.setLevel(
        level
    )  # Set this to logging.DEBUG to check hivemind debug messages -> Careful, it's a lot of output

    bt_handler = BittensorLogHandler()
    formatter = logging.Formatter("%(message)s")
    bt_handler.setFormatter(formatter)
    root_logger.addHandler(bt_handler)
    root_logger.addHandler(logtail_handler)

    add_loki_logger_handler(
        root_logger,
        network,
        netuid,
        hotkey,
        version,
        spec_version,
        run_id,
        ip,
        port,
        uid,
        neuron_type,
    )

    # Create a file handler that logs debug and higher level messages
    if os.path.exists(local_logfile):
        # Archive any existing logfile
        shutil.copyfile(local_logfile, local_logfile.replace(".txt", "_archive.txt"))
        os.remove(local_logfile)

    hivemind_log_file = f"/root/logs_mylogfile.txt"
    hivemind_logger = logging.getLogger("hivemind")
    hivemind_logger.setLevel(logging.DEBUG)  # Capture all logs from hivemind
    file_handler = logging.FileHandler(hivemind_log_file)
    file_handler.setLevel(
        logging.DEBUG
    )  # Ensure file handler captures all levels for hivemind
    file_handler.addFilter(logging_filter)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    hivemind_logger.addHandler(file_handler)
    # hivemind_logger.addHandler(logtail_handler)
    hivemind_logger.propagate = (
        False  # Stop hivemind logs from propagating to the root logger
    )


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
    buffer = 5
    max_retries = buffer * len(initial_peers_list)
    successful_connection = False
    while successful_connection is False:
        if (retries == max_retries) and (successful_connection is False):
            raise Exception("Max retries reached, operation failed.")
        for initiaL_peer in initial_peers_list:
            for attempt in range(0, buffer):
                try:
                    # Init DHT
                    self.dht = hivemind.DHT(
                        host_maddrs=[
                            f"/ip4/0.0.0.0/tcp/{self.config.dht.port}",
                            f"/ip4/0.0.0.0/udp/{self.config.dht.port}/quic",
                        ],
                        initial_peers=[initiaL_peer],
                        announce_maddrs=announce_maddrs,
                        start=True,
                    )
                    bt.logging.info(
                        f"Successfully initialised dht using initial_peer as {initiaL_peer}"
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
                        f"Attempt {attempt + 1} to init DHT using initial_peer as {initiaL_peer} failed with error: {e}"
                    )
                    retries += 1
                    time.sleep(5)
                    bt.logging.error(f"Retrying...")


def warmup(self):
    """
    Processes the incoming 'Train' synapse by performing a training run

    Args:
        synapse (template.protocol.Train): The synapse object containing the 'dataset_indices' data.

    Returns:
        template.protocol.Train: The synapse object with the 'loss' field set to models loss.
    """

    # Load dataset
    self.dataset_loader = ()
    dataset_length = DataLoader.max_pages
    self.dataset_indices = bitarray(dataset_length)

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
        i for i in range(start, start + self.config.neuron.training_examples_per_miner)
    ]
    self.dataset_indices[group] = True

    # Create Dataloader
    dataloader = DataLoader(
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

        # Zero Gradients
        self.opt.zero_grad()

        # Update Tracker
        self.local_progress.samples_accumulated += 1

        # Log accumulation status
        bt.logging.info(f"Index: {index} | Loss: {outputs.loss.detach().item():.2f}")

        if not self.config.neuron.dont_wandb_log:
            self.wandb.log(
                {
                    "loss": outputs.loss.detach().item(),
                    "local_epoch": self.local_progress.local_epoch,
                    "global_epoch": self.global_progress.epoch,
                }
            )
