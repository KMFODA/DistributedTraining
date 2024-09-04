import asyncio
import io
import random
import time
import traceback
from collections import defaultdict
from typing import Dict, List, Optional

import bittensor as bt
# import matplotlib.pyplot as plt
import numpy as np
import torch
from hivemind.p2p import PeerID
from hivemind.utils.timed_storage import ValueWithExpiration

import template


async def check_uid(dendrite, axon, uid, epoch=None):
    try:
        response = await dendrite(
            axon, template.protocol.IsAlive(), deserialize=False, timeout=2.3
        )
        if response.is_success:
            if (epoch is not None) and (response.epoch == epoch):
                bt.logging.trace(f"UID {uid} is active and on epoch {epoch}")
                return True
            elif (epoch is not None) and (response.epoch != epoch):
                bt.logging.trace(f"UID {uid} is active but not on epoch {epoch}")
                return False
            else:
                bt.logging.trace(f"UID {uid} is active.")
                return True
        else:
            bt.logging.trace(f"UID {uid} is not active.")
            return False
    except Exception as e:
        bt.logging.error(f"Error checking UID {uid}: {e}\n{traceback.format_exc()}")
        # loop.close()
        return False


async def check_uid_availability(
    dendrite,
    metagraph: "bt.metagraph.Metagraph",
    uid: int,
    vpermit_tao_limit: int,
    epoch: int = None,
) -> bool:
    """Check if uid is available. The UID should be available if it is serving and has less than vpermit_tao_limit stake
    Args:
        metagraph (:obj: bt.metagraph.Metagraph): Metagraph object
        uid (int): uid to be checked
        vpermit_tao_limit (int): Validator permit tao limit
    Returns:
        bool: True if uid is available, False otherwise
    """
    # Filter non serving axons.
    if not metagraph.axons[uid].is_serving:
        return False
    # Filter validator permit > 1024 stake.
    if metagraph.validator_permit[uid]:
        if metagraph.S[uid] > vpermit_tao_limit:
            return False
    # Filter for miners that are processing other responses
    if not await check_uid(dendrite, metagraph.axons[uid], uid, epoch):
        return False
    # Available otherwise.
    return True


async def get_random_uids(
    self, dendrite, k: int, exclude: List[int] = None, epoch: int = None
) -> torch.LongTensor:
    """Returns k available random uids from the metagraph.
    Args:
        k (int): Number of uids to return.
        exclude (List[int]): List of uids to exclude from the random sampling.
    Returns:
        uids (torch.LongTensor): Randomly sampled available uids.
    Notes:
        If `k` is larger than the number of available `uids`, set `k` to the number of available `uids`.
    """

    candidate_uids = []
    avail_uids = []

    tasks = []
    for uid in range(self.metagraph.n.item()):
        # The dendrite client queries the network.
        tasks.append(
            check_uid_availability(
                dendrite,
                self.metagraph,
                uid,
                self.config.neuron.vpermit_tao_limit,
                epoch,
            )
        )

    responses = await asyncio.gather(*tasks)

    for uid, uid_is_available in zip(range(self.metagraph.n.item()), (responses)):
        uid_is_not_excluded = exclude is None or uid not in exclude
        if uid_is_available:
            avail_uids.append(uid)
            if uid_is_not_excluded:
                candidate_uids.append(uid)

    # Check if candidate_uids contain enough for querying, if not grab all avaliable uids
    available_uids = candidate_uids
    if len(candidate_uids) < k:
        uids = torch.tensor(available_uids)
    else:
        uids = torch.tensor(random.sample(available_uids, k))

    return uids


import json


async def map_uid_to_peerid(
    self, uids: List[int], max_retries: int = 3, retry_delay: float = 1.0
) -> Dict[int, Optional[str]]:
    bt.logging.info(f"Starting map_uid_to_peerid for UIDs: {uids}")
    uids_to_peerids = {uid: None for uid in uids}

    for attempt in range(max_retries):
        bt.logging.info(f"Attempt {attempt + 1} of {max_retries}")

        peer_list_dht = await self._p2p.list_peers()
        peer_list_dht_addrs = [
            str(peer.addrs[0]).split("/ip4/")[1].split("/")[0] for peer in peer_list_dht
        ]

        prefix = self.grad_averager.matchmaking_kwargs["prefix"]
        metadata, _ = self.dht.get(f"{prefix}.all_averagers", latest=True) or ({}, None)

        if metadata is None:
            bt.logging.warning(f"No metadata found in DHT for prefix {prefix}")
            await asyncio.sleep(retry_delay)
            continue

        peer_list_run = [
            str(PeerID(peer_id))
            for peer_id, info in metadata.items()
            if isinstance(info, ValueWithExpiration)
            and isinstance(info.value, (float, int))
        ]

        for uid in uids:
            if uids_to_peerids[uid] is not None:
                continue

            miner_ip = self.metagraph.axons[uid].ip

            if miner_ip not in peer_list_dht_addrs:
                bt.logging.warning(
                    f"Miner IP {miner_ip} for UID {uid} not in peer_list_dht_addrs"
                )
                update_peer_status(self, uid, "disconnected")
                continue

            peer_id = peer_list_dht[peer_list_dht_addrs.index(miner_ip)].peer_id

            if str(peer_id) not in peer_list_run:
                bt.logging.warning(
                    f"peer_id {peer_id} for UID {uid} not in peer_list_run"
                )
                update_peer_status(self, uid, "disconnected")
                continue

            uids_to_peerids[uid] = peer_id
            update_peer_status(self, uid, "connected")
            bt.logging.info(f"Successfully mapped UID {uid} to peer_id {peer_id}")

        if all(peer_id is not None for peer_id in uids_to_peerids.values()):
            bt.logging.info(f"Self UID: {self.uid} with peer_id: {self.dht.peer_id}")
            bt.logging.info(f"Is connected to uids: {uids}")
            bt.logging.info(f"Is connected to peer_list: {peer_list_dht_addrs}")
            bt.logging.info(f"Is connected to peer_run: {peer_list_run}")
            await asyncio.sleep(5)
            break

        await asyncio.sleep(retry_delay)

    # save_mapping(uids_to_peerids)
    # save_status_history(self)
    # generate_uptime_graph(self)

    bt.logging.info(f"Final mapping of UIDs to peer IDs: {uids_to_peerids}")
    return uids_to_peerids


def update_peer_status(self, uid: int, status: str):
    current_time = time.time() - self.start_time
    self.peer_status[uid] = status
    if uid not in self.status_history:
        self.status_history[uid] = []
    self.status_history[uid].append((current_time, status))


def save_mapping(uids_to_peerids):
    serializable_mapping = {}
    for uid, peer_id in uids_to_peerids.items():
        if isinstance(peer_id, PeerID):
            serializable_mapping[uid] = peer_id.to_string()
        else:
            serializable_mapping[uid] = peer_id

    with open("uid_to_peerid_mapping.json", "w") as f:
        json.dump(serializable_mapping, f, indent=4)


def save_status_history(self):
    with open("peer_status_history.json", "w") as f:
        json.dump(self.status_history, f, indent=4)


def generate_random_color():
    return tuple(random.random() for _ in range(3))


def get_uid_color(self, uid):
    if uid not in self.uid_colors:
        self.uid_colors[uid] = generate_random_color()
    return self.uid_colors[uid]


# def update_peer_status(self, uid, status, timestamp):
#     self.status_history[uid].append((timestamp, status))


def generate_uptime_graph(self):
    fig, ax = plt.subplots(figsize=(20, 15))  # Increased figure height

    # Prepare data
    all_times = sorted(
        set(time for history in self.status_history.values() for time, _ in history)
    )
    status_matrix = defaultdict(lambda: [None] * len(all_times))

    # Fill in status matrix
    for uid, history in self.status_history.items():
        current_status = "disconnected"  # Assume disconnected at start
        for i, current_time in enumerate(all_times):
            # Find the latest status update for this UID up to the current time
            latest_status = next(
                (status for time, status in reversed(history) if time <= current_time),
                current_status,
            )
            current_status = latest_status
            status_matrix[uid][i] = 1 if current_status == "connected" else 0

    # Calculate positions for each UID
    uid_positions = {uid: idx for idx, uid in enumerate(sorted(status_matrix.keys()))}
    num_uids = len(uid_positions)

    # Plot lines
    for uid, statuses in status_matrix.items():
        y_values = []
        for status in statuses:
            if status == 1:
                # Connected: Space UIDs between 0.6 and 0.9
                y_values.append(0.6 + (uid_positions[uid] / (num_uids - 1)) * 0.3)
            else:
                y_values.append(0.25)  # Disconnected

        color = get_uid_color(self, uid)
        ax.plot(
            all_times, y_values, color=color, alpha=0.7, linewidth=2, label=f"UID {uid}"
        )

        # Add label at the end of the line
        last_valid_index = len(y_values) - 1
        if last_valid_index >= 0:
            ax.annotate(
                f"UID {uid}",
                xy=(all_times[last_valid_index], y_values[last_valid_index]),
                xytext=(5, 0),
                textcoords="offset points",
                va="center",
                fontsize=8,
                color=color,
            )

    # Plot AllReduce operations
    for i, operation in enumerate(self.allreduce_history):
        color = "green" if operation["success"] else "red"
        ax.axvline(x=operation["time"], color=color, linestyle="--", alpha=0.5)

        # Add label for participating UIDs
        uids_str = ", ".join(map(str, operation["uids"]))
        status_str = "Success" if operation["success"] else "Failure"

        # Alternate between top and bottom placement to avoid overlapping
        y_pos = 0.95 if i % 2 == 0 else 0.05
        va = "bottom" if i % 2 == 0 else "top"

        ax.annotate(
            f"{status_str}\nUIDs: {uids_str}",
            xy=(operation["time"], y_pos),
            xytext=(0, 5 if i % 2 == 0 else -5),
            textcoords="offset points",
            rotation=45,
            va=va,
            ha="center",
            fontsize=6,
            color=color,
        )

    # Customize the plot
    ax.set_title(
        "Peer Connection Status and AllReduce Operations Over Time", fontsize=16
    )
    ax.set_xlabel("Time (seconds)", fontsize=12)
    ax.set_yticks([0.25, 0.75])
    ax.set_yticklabels(["Disconnected", "Connected"])
    ax.set_ylim(0, 1)  # Set y-axis limits

    # Add shaded areas for connected and disconnected lanes
    ax.axhspan(0, 0.5, facecolor="red", alpha=0.1)
    ax.axhspan(0.5, 1, facecolor="green", alpha=0.1)

    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()

    # Save static PNG
    plt.savefig("peer_uptime_graph.png", dpi=300, bbox_inches="tight")

    # Log to wandb
    if not self.config.neuron.dont_wandb_log:
        # Convert plot to image
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format="png")
        img_buf.seek(0)

        # Log the plot as an image
        self.wandb.log({"Peer Uptime Graph": self.wandb.Image(img_buf)})

    plt.close()


# async def run_uptime_tracking(self, uids: List[int], interval: int):
#     while True:
#         await map_uid_to_peerid(self, uids)
#         save_status_history(self)
#         generate_uptime_graph(self)
#         await asyncio.sleep(interval)


def initialize_uid_mapping(self):
    self.uid_colors = {}
    max_retries = 3  # TODO Make config
    for attempt in range(max_retries):
        uids_to_peerids = self.loop.run_until_complete(
            map_uid_to_peerid(self, range(self.metagraph.n))
        )
        if any(value is not None for value in uids_to_peerids.values()):
            with open("init_uid_mapping.txt", "w") as f:
                json.dump(str(uids_to_peerids), f, indent=4)
            return uids_to_peerids
        time.sleep(1)  # Sleep for 1 second between retries

    bt.logging.warning("Failed to map any UIDs to peer IDs after maximum retries")
    return {uid: None for uid in range(self.metagraph.n)}
