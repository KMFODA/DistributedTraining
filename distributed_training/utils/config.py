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

import argparse
import os

import bittensor as bt
import torch
from distributed_training import __run__, __version__


def check_config(cls, config: "bt.Config"):
    r"""Checks/validates the config namespace object."""
    bt.logging.check_config(config)

    full_path = os.path.expanduser(
        "{}/{}/{}/netuid{}/{}".format(
            config.logging.logging_dir,
            config.wallet.name,
            config.wallet.hotkey,
            config.netuid,
            config.neuron.name,
        )
    )
    print("full path:", full_path)
    config.neuron.full_path = os.path.expanduser(full_path)
    if not os.path.exists(config.neuron.full_path):
        os.makedirs(config.neuron.full_path, exist_ok=True)


def add_args(cls, parser, prefix=None):
    """
    Adds relevant arguments to the parser for operation.
    """
    # Netuid Arg: The netuid of the subnet to connect to.
    parser.add_argument("--netuid", type=int, help="Subnet netuid", default=1)

    neuron_type = "validator" if "miner" not in cls.__name__.lower() else "miner"

    prefix_str = "" if prefix == None else prefix + "."
    try:
        default_name = os.getenv("BT_WALLET_NAME") or "default"
        default_hotkey = os.getenv("BT_WALLET_NAME") or "default"
        default_path = os.getenv("BT_WALLET_PATH") or "~/.bittensor/wallets/"
        parser.add_argument(
            "--no_prompt",
            dest="no_prompt",
            action="store_true",
            help="""Set true to avoid prompting the user.""",
            default=False,
        )
        parser.add_argument(
            "--" + prefix_str + "wallet.name",
            required=False,
            default=default_name,
            help="The name of the wallet to unlock for running bittensor "
            "(name mock is reserved for mocking this wallet)",
        )
        parser.add_argument(
            "--" + prefix_str + "wallet.hotkey",
            required=False,
            default=default_hotkey,
            help="The name of the wallet's hotkey.",
        )
        parser.add_argument(
            "--" + prefix_str + "wallet.path",
            required=False,
            default=default_path,
            help="The path to your bittensor wallets",
        )
    except argparse.ArgumentError as e:
        pass

    parser.add_argument(
        "--dht.port",
        type=int,
        help="Trials for this neuron go in neuron.root / (wallet_cold - wallet_hot) / neuron.name. ",
        default=8009,
    )

    parser.add_argument(
        "--dht.ip",
        type=str,
        help="The IP address to use in announce_maddrs",
    )

    parser.add_argument(
        "--neuron.events_retention_size",
        type=str,
        help="Events retention size.",
        default=2 * 1024 * 1024 * 1024,  # 2 GB
    )

    parser.add_argument(
        "--neuron.name",
        type=str,
        help="Trials for this neuron go in neuron.root / (wallet_cold - wallet_hot) / neuron.name. ",
        default=neuron_type,
    )

    parser.add_argument(
        "--neuron.device",
        type=str,
        help="Device to run on.",
        default=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    parser.add_argument(
        "--neuron.epoch_length",
        type=int,
        help="The default epoch length (how often we set weights, measured in 12 second blocks).",
        default=100,
    )

    parser.add_argument(
        "--neuron.dont_save_events",
        action="store_true",
        help="If set, we dont save events to a log file.",
        default=False,
    )

    parser.add_argument(
        "--neuron.initial_peers",
        type=str,
        nargs="+",
        help="The addresses for the DHT",
        default=[
            "/ip4/161.97.156.125/tcp/8000/p2p/12D3KooWECM2JXC8qYyP74vxaEjhwCto4jarQDDw5hKTvCVNVAms",
        ],
    )
    parser.add_argument(
        "--neuron.blocks_per_allreduce",
        type=int,
        help="Amount of blocks between each all reduce",
        default=30,
    )

    parser.add_argument(
        "--neuron.grad_accum_steps",
        type=int,
        help="Amount of micro batches for gradient accumulation",
        default=512,
    )

    parser.add_argument(
        "--neuron.model_name",
        type=str,
        help="The model to be trained",
        default="distributed/optimized-gpt2-1b-vtestnet",
    )

    parser.add_argument(
        "--neuron.hf_repo_id",
        type=str,
        help="The model to be trained",
        default=None,
    )

    parser.add_argument(
        "--neuron.master_ss58_address",
        type=str,
        help="The ss58 address for the master validator UID",
        default="5EnC86fRRRoaXUZvkrDFYpAihuyEAp3wGkY5r3Gak1kPTDVP",
    )

    parser.add_argument(
        "--neuron.min_group_size",
        type=int,
        help="The minimum group size for an all reduce",
        default=2,
    )

    parser.add_argument(
        "--neuron.local_batch_size_train",
        type=int,
        help="The default batch size",
        default=4,
    )

    parser.add_argument(
        "--neuron.global_batch_size_train",
        type=int,
        help="The hivemind global target_batch_size",
        default=35200,
    )

    parser.add_argument(
        "--neuron.run_id",
        type=str,
        help="The DHT run_id",
        default=f"v{__version__.replace('.','_')}_r{__run__}",
    )

    parser.add_argument(
        "--neuron.dont_wandb_log",
        action="store_true",
        help="Toggles wandb logging for the project",
        default=False,
    )

    parser.add_argument(
        "--neuron.wandb_project",
        type=str,
        help="The wandb project to log to",
        default="distributed_training",
    )

    parser.add_argument(
        "--neuron.wandb_entity",
        type=str,
        help="The wandb project to log to",
        default="kmfoda",
    )

    if neuron_type == "validator":
        parser.add_argument(
            "--neuron.uid_isalive_limit",
            type=int,
            help="The maximum number of uids to call concurrently",
            default=25,
        )

        parser.add_argument(
            "--neuron.weight_update_interval",
            type=int,
            help="The number of steps before updating the model's weights",
            default=900,
        )

        parser.add_argument(
            "--neuron.training_examples_per_miner",
            type=int,
            help="The number of rows to train on per miner",
            default=1024,
        )

        parser.add_argument(
            "--neuron.upload_interval",
            type=int,
            help="The number of steps before uploading the model",
            default=900,
        )

        parser.add_argument(
            "--neuron.num_concurrent_forwards",
            type=int,
            help="The number of concurrent forwards running at any time.",
            default=1,
        )

        parser.add_argument(
            "--neuron.sample_size",
            type=int,
            help="The number of miners to query in a single step.",
            default=25,
        )

        parser.add_argument(
            "--neuron.disable_set_weights",
            action="store_true",
            help="Disables setting weights.",
            default=False,
        )

        parser.add_argument(
            "--neuron.moving_average_alpha",
            type=float,
            help="Moving average alpha parameter, how much to add of the new observation.",
            default=0.05,
        )

        parser.add_argument(
            "--neuron.axon_off",
            "--axon_off",
            action="store_true",
            # Note: the validator needs to serve an Axon with their IP or they may
            #   be blacklisted by the firewall of serving peers on the network.
            help="Set this flag to not attempt to serve an Axon.",
            default=False,
        )

        parser.add_argument(
            "--neuron.vpermit_tao_limit",
            type=int,
            help="The maximum number of TAO allowed to query a validator with a vpermit.",
            default=4096,
        )

    else:
        parser.add_argument(
            "--blacklist.force_validator_permit",
            action="store_true",
            help="If set, we will force incoming requests to have a permit.",
            default=False,
        )

        parser.add_argument(
            "--blacklist.allow_non_registered",
            action="store_true",
            help="If set, miners will accept queries from non registered entities. (Dangerous!)",
            default=False,
        )


def config(cls):
    """
    Returns the configuration object specific to this miner or validator after adding relevant arguments.
    """
    parser = argparse.ArgumentParser()
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)
    bt.axon.add_args(parser)
    cls.add_args(parser)
    return bt.config(parser)
