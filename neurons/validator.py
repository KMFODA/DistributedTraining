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


import os

os.environ["NEST_ASYNCIO"] = "0"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
import math
import threading

import bittensor as bt
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
from transformers import AutoTokenizer

from distributed_training.base.validator import BaseValidatorNeuron
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
from random import randrange
from distributed_training.utils.state_loader import (
    FastModelLoader,
    cleanup_old_cache,
    load_model_optimizer_gradient_averager,
    load_state_from_peer,
)
from distributed_training.utils.uids import map_uid_to_peerid, update_run_peerid_list
from distributed_training.validator import forward


class Validator(BaseValidatorNeuron):
    def __init__(self, config=None):
        super(Validator, self).__init__(config=config)
        self._update_wandb_project()
        self._init_basic_components()
        self._init_model_components()
        self._init_network_components()
        self._init_uid_components()
        self._randomly_reset_uid_tracker()

    def _update_wandb_project(self):
        suffix = "_validators" if self.neuron_type == "ValidatorNeuron" else "_miners"
        self.config.neuron.wandb_project += suffix

    def report_allreduce_operation(
        self,
        op_id,
        epoch,
        validator_uid,
        success_rate,
        duration,
        participating_miners_count,
        bandwidth=None,
    ):
        """Report AllReduce operation metrics to InfluxDB"""
        try:
            point = (
                Point("allreduce_operations")
                .tag("operation_id", str(op_id))
                .tag("epoch", str(epoch))
                .tag("validator_uid", str(validator_uid))
                .field("success_rate", float(success_rate))
                .field("duration", float(duration))
                .field("participating_miners", int(participating_miners_count))
            )

            if bandwidth is not None:
                point = point.field("bandwidth", float(bandwidth))

            self.influx_write_api.write(
                bucket=self.config.neuron.influxdb_bucket,
                org=self.config.neuron.influxdb_org,
                record=point,
            )
            bt.logging.info(
                f"Validator {validator_uid} reported AllReduce operation {op_id} metrics to InfluxDB"
            )
        except Exception as e:
            bt.logging.error(f"Error reporting AllReduce metrics: {e}")

    def report_scoring_metrics(self):
        """Send validator scoring metrics to InfluxDB"""
        try:
            points = []

            for uid, data in self.uid_tracker.items():
                point = (
                    Point("miner_scores")
                    .tag("validator_uid", str(self.uid))
                    .tag("miner_uid", str(uid))
                    .field("train_score", float(data["train_score"] or 0))
                    .field("all_reduce_score", float(data["all_reduce_score"] or 0))
                    .field("repo_valid_score", float(data["repo_valid_score"] or 0))
                    .field("total_score", float(data["total_score"] or 0))
                )
                points.append(point)

                if "loss" in data:
                    point = (
                        Point("miner_loss")
                        .tag("validator_uid", str(self.uid))
                        .tag("miner_uid", str(uid))
                        .field("loss", float(data["loss"] or 0))
                    )
                    points.append(point)

            # Write points to InfluxDB
            self.influx_write_api.write(
                bucket=self.config.neuron.influxdb_bucket,
                org=self.config.neuron.influxdb_org,
                record=points,
            )
        except Exception as e:
            bt.logging.error(f"Error reporting scoring metrics: {e}")

    def _init_metrics_collection(self):
        # Initialize InfluxDB client
        self.influx_client = None
        self.influx_write_api = None
        try:
            bt.logging.info(
                "Attempting to initialize InfluxDB client for metrics collection..."
            )
            self.influx_client = InfluxDBClient(
                url=self.config.neuron.influxdb_url,
                token=self.config.neuron.influxdb_token,
                org=self.config.neuron.influxdb_org,
            )
            self.influx_write_api = self.influx_client.write_api(
                write_options=SYNCHRONOUS
            )
            bt.logging.info("InfluxDB client and write_api initialized successfully.")

        except Exception as e:
            bt.logging.error(
                f"Failed to initialize InfluxDB client: {e}. Metrics collection will be disabled."
            )
            if self.influx_client:
                try:
                    self.influx_client.close()
                except Exception as close_e:
                    bt.logging.error(
                        f"Error closing InfluxDB client during cleanup: {close_e}"
                    )
            self.influx_client = None
            self.influx_write_api = None

    def _init_basic_components(self):
        """Initialize basic validator components"""
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
                self, self.config, self.wallet, "validator", str(self.dht.peer_id)
            )

        # Tracking metrics
        self._init_metrics_collection()

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
        self.local_progress.epoch = self.global_progress.epoch

        if self.global_progress.epoch is None:
            bt.logging.error(
                "Model Tag Is None. Make Sure You Are Using The Correct Model Name"
            )

    def _init_model_components(self):
        """Initialize model-related components including tokenizer and optimizer settings."""
        self._setup_model_params()
        self._init_tokenizer()
        self._setup_model_state()
        self._setup_training_params()
        self._setup_uid_api()

    def _setup_model_params(self):
        # Timeouts
        self.load_state_timeout = 180

        # Core parameters
        self.learning_rate_maximum = 4e-4
        self.weight_decay = 0.1
        self.num_inner_steps = 500
        self.offload_optimizer = True
        self.model_upload_retry_limit = 3
        self.model_upload_retry_delay = 10

        # Validator-specific training parameters
        self.maximum_steps = 306 * 4  # 10_000_000_000/(32000*1024)
        self.warmup_steps = 62  # 306 / 5
        self.failed_is_alive_counter_threshold = 10

    def _init_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.neuron.global_model_name, use_fast=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def _setup_model_state(self):
        self.learning_rate = self.get_learning_rate()
        self.average_loss = None
        self.loader = FastModelLoader(self.config.neuron.hf_repo_id)

        load_model_optimizer_gradient_averager(
            self, self.config.neuron.global_model_name, self.global_progress.epoch
        )
        cleanup_old_cache(self)

        if self.local_progress.epoch < self.global_progress.epoch:
            load_state_from_peer(self, epoch=self.global_progress.epoch)

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

    def _init_uid_components(self):
        self._setup_uids()
        self._init_peer_mapping()
        self._setup_allreduce_block()

    def _setup_uids(self):
        self.master_uid = self.metagraph.hotkeys.index(
            self.config.neuron.master_ss58_address,
        )
        self.failed_is_alive_counter = {uid: 0 for uid in self.metagraph.uids.tolist()}
        self.miner_uids = []

    def _randomly_reset_uid_tracker(self):
        if self.uid == self.master_uid:
            self.uid_tracker[randrange(256)]["train_similarity_score_last_updated"] = 0

    def _init_peer_mapping(self):
        self.stop_event = threading.Event()
        map_uid_to_peerid(self)
        update_run_peerid_list(self)

    def _setup_allreduce_block(self):
        if (self.uid == self.master_uid) or (
            "last_allreduce_block" not in self.model.config.__dict__
        ):
            self.last_allreduce_block = self.block
        else:
            self.last_allreduce_block = self.model.config.last_allreduce_block

    def _setup_uid_api(self):
        self.uid_api_url = self.config.neuron.uid_api_url
        self.uid_api_get_token = self.config.neuron.uid_api_get_token
        self.uid_api_post_token = self.config.neuron.uid_api_post_token

    def update_local_tracker_state(self, rewards, responses):
        for reward, response in zip(rewards, responses[0]):
            if (reward != 0) and (response.dataset_indices is not None):
                self.local_progress.samples_accumulated += len(response.dataset_indices)
            else:
                continue

    def get_learning_rate(self):
        learning_rate_minimum = self.learning_rate_maximum * 0.1
        # 1) linear warmup for warmup_steps
        if self.global_progress.epoch < self.warmup_steps:
            return (
                self.learning_rate_maximum
                * (self.global_progress.epoch + 1)
                / self.warmup_steps
            )
        # 2) if epoch > lr_decay_iters, return learning_rate_minimum
        if self.global_progress.epoch > self.maximum_steps:
            return learning_rate_minimum
        # 3) if in between, use cosine decay down to min learning rate
        decay_ratio = (self.global_progress.epoch - self.warmup_steps) / (
            self.maximum_steps - self.warmup_steps
        )
        assert 0 <= decay_ratio <= 1
        # coeff starts at 1 and goes to 0
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return (learning_rate_minimum + coeff) * (
            self.learning_rate_maximum - learning_rate_minimum
        )

    def get_validator_info(self):
        return {
            "block": self.metagraph.block.item(),
            "stake": self.metagraph.stake[self.uid],
            "rank": self.metagraph.ranks[self.uid],
            "vtrust": self.metagraph.validator_trust[self.uid],
            "dividends": self.metagraph.dividends[self.uid],
            "emissions": self.metagraph.emission[self.uid],
        }

    async def forward(self):
        return await forward(self)


# The main function parses the configuration and runs the validator.
if __name__ == "__main__":
    Validator().run()
