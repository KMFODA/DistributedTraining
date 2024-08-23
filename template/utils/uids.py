import time
import asyncio
import random
import traceback
from typing import Dict, Optional, List

import matplotlib.pyplot as plt
import bittensor as bt
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

async def map_uid_to_peerid(self, uids: List[int], max_retries: int = 3, retry_delay: float = 1.0) -> Dict[int, Optional[str]]:
    bt.logging.info(f"Starting map_uid_to_peerid for UIDs: {uids}")
    uids_to_peerids = {uid: None for uid in uids}
    
    for attempt in range(max_retries):
        bt.logging.info(f"Attempt {attempt + 1} of {max_retries}")
        
        peer_list_dht = await self._p2p.list_peers()
        peer_list_dht_addrs = [str(peer.addrs[0]).split("/ip4/")[1].split("/")[0] 
                                for peer in peer_list_dht]
        
        prefix = self.grad_averager.matchmaking_kwargs["prefix"]
        metadata, _ = self.dht.get(f"{prefix}.all_averagers", latest=True) or ({}, None)
        
        if metadata is None:
            bt.logging.warning(f"No metadata found in DHT for prefix {prefix}")
            await asyncio.sleep(retry_delay)
            continue
        
        peer_list_run = [
            str(PeerID(peer_id))
            for peer_id, info in metadata.items()
            if isinstance(info, ValueWithExpiration) and isinstance(info.value, (float, int))
        ]
        
        for uid in uids:
            if uids_to_peerids[uid] is not None:
                continue
            
            miner_ip = self.metagraph.axons[uid].ip
            
            if miner_ip not in peer_list_dht_addrs:
                bt.logging.warning(f"Miner IP {miner_ip} for UID {uid} not in peer_list_dht_addrs")
                update_peer_status(self, uid, 'disconnected')
                continue
            
            peer_id = peer_list_dht[peer_list_dht_addrs.index(miner_ip)].peer_id
            
            if str(peer_id) not in peer_list_run:
                bt.logging.warning(f"peer_id {peer_id} for UID {uid} not in peer_list_run")
                update_peer_status(self, uid, 'disconnected')
                continue
            
            uids_to_peerids[uid] = peer_id
            update_peer_status(self, uid, 'connected')
            bt.logging.info(f"Successfully mapped UID {uid} to peer_id {peer_id}")
        
        if all(peer_id is not None for peer_id in uids_to_peerids.values()):
            bt.logging.info(f"Self UID: {self.uid} with peer_id: {self.dht.peer_id}")
            bt.logging.info(f"Is connected to uids: {uids}")
            bt.logging.info(f"Is connected to peer_list: {peer_list_dht_addrs}")
            bt.logging.info(f"Is connected to peer_run: {peer_list_run}")
            await asyncio.sleep(5)
            break
        
        await asyncio.sleep(retry_delay)
    
    save_mapping(uids_to_peerids)
    save_status_history(self)
    generate_uptime_graph(self)
    
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

        with open('uid_to_peerid_mapping.json', 'w') as f:
            json.dump(serializable_mapping, f, indent=4)

def save_status_history(self):
    with open('peer_status_history.json', 'w') as f:
        json.dump(self.status_history, f, indent=4)

def generate_uptime_graph(self):
    plt.figure(figsize=(12, 8))
    for uid, history in self.status_history.items():
        times, statuses = zip(*history)
        plt.plot(times, [1 if status == 'connected' else 0 for status in statuses], label=f'UID {uid}')
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('Connection Status')
    plt.title('Peer Connection Status Over Time')
    plt.legend()
    plt.savefig("peer_uptime_graph.png")
    plt.close()

# async def run_uptime_tracking(self, uids: List[int], interval: int):
#     while True:
#         await map_uid_to_peerid(self, uids)
#         save_status_history(self)
#         generate_uptime_graph(self)
#         await asyncio.sleep(interval)

def initialize_uid_mapping(self):
    max_retries = 3 # TODO Make config
    for attempt in range(max_retries):
        uids_to_peerids = self.loop.run_until_complete(
            map_uid_to_peerid(self, range(self.metagraph.n))
        )
        if any(value is not None for value in uids_to_peerids.values()):
            with open('init_uid_mapping.txt', 'w') as f:
                json.dump(str(uids_to_peerids), f, indent=4)
            return uids_to_peerids
        time.sleep(1)  # Sleep for 1 second between retries

    bt.logging.warning("Failed to map any UIDs to peer IDs after maximum retries")
    return {uid: None for uid in range(self.metagraph.n)}