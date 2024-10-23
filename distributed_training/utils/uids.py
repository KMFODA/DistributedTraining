import random
import bittensor as bt
from typing import List
import traceback
import asyncio
import distributed_training
from hivemind.utils.timed_storage import ValueWithExpiration
from hivemind.p2p import PeerID
import time
import datetime as dt

import numpy as np

async def check_uid(dendrite, axon, uid, epoch=None):
    try:
        response = await dendrite(
            axon,
            distributed_training.protocol.IsAlive(),
            deserialize=False,
            timeout=2.3,
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
        print("NOW deing deaong....")
        bt.logging.error(f"Error checking UID {uid}: {e}\n{traceback.format_exc()}")
        # loop.close()
        return False

def check_uid_availability(
    metagraph: "bt.metagraph.Metagraph", uid: int, vpermit_tao_limit: int
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
    # if not await check_uid(dendrite, metagraph.axons[uid], uid):#, epoch):
#     #     return False
    # Available otherwise.
    return True

def get_random_uids(self, k: int, exclude: List[int] = None) -> np.ndarray:
    """Returns k available random uids from the metagraph.
    Args:
        k (int): Number of uids to return.
        exclude (List[int]): List of uids to exclude from the random sampling.
    Returns:
        uids (np.ndarray): Randomly sampled available uids.
    Notes:
        If `k` is larger than the number of available `uids`, set `k` to the number of available `uids`.
    """
    candidate_uids = []
    avail_uids = []

    for uid in range(self.metagraph.n.item()):
        uid_is_available = check_uid_availability(
            self.metagraph, uid, vpermit_tao_limit=1
        )
        uid_is_not_excluded = exclude is None or uid not in exclude

        if uid_is_available:
            avail_uids.append(uid)
            if uid_is_not_excluded:
                candidate_uids.append(uid)
    # If k is larger than the number of available uids, set k to the number of available uids.
    k = min(k, len(avail_uids))
    # Check if candidate_uids contain enough for querying, if not grab all avaliable uids
    available_uids = candidate_uids
    if len(candidate_uids) < k:
        available_uids += random.sample(
            [uid for uid in avail_uids if uid not in candidate_uids],
            k - len(candidate_uids),
        )
    uids = np.array(random.sample(available_uids, k))
    return uids


async def map_uid_to_peerid_archive(self, uids):
    # Get all peers connected to our DHT, their ips and their ports
    peer_list_dht = await self._p2p.list_peers()
    peer_list_dht_addrs = [
        str(peer.addrs[0]).split("/ip4/")[1].split("/")[0] for peer in peer_list_dht
    ]
    peer_list_dht_ports = [str(peer.addrs[0]).split("/")[-1] for peer in peer_list_dht]

    # Get only peers connected to the current run id
    prefix = self.grad_averager.matchmaking_kwargs["prefix"]
    metadata, _ = self.dht.get(f"{prefix}.all_averagers", latest=True) or (
        {},
        None,
    )

    uids_to_peerids = {}
    for uid in uids:
        miner_ip = self.metagraph.axons[uid].ip
        miner_port = self.metagraph.axons[uid].port

        if metadata is None:
            # return None
            uids_to_peerids[uid] = None
            continue

        run_peer_id_list = [
            str(PeerID(peer_id))
            for peer_id, info in metadata.items()
            if isinstance(info, ValueWithExpiration)
            and isinstance(info.value, (float, int))
        ]

        # If the UIDs ip address is not in the list of peer addrs then it is not connected to our DHT
        if miner_ip not in peer_list_dht_addrs:
            # return None
            uids_to_peerids[uid] = None
            continue
        else:
            # if peer_list_dht_addrs.count(miner_ip) > 1:
            #     indices = [
            #         i
            #         for i in range(len(peer_list_dht_addrs))
            #         if peer_list_dht_addrs[i] == miner_ip
            #     ]
            #     peer_id = None
            #     for index in indices:
            #         if abs(miner_port - int(peer_list_dht_ports[index])) < 10:
            #             peer_id = peer_list_dht[index].peer_id
            #             break
            #         elif index == indices[-1]:
            #             break
            #         else:
            #             continue

            #     if peer_id is None:
            #         uids_to_peerids[uid] = None
            #         continue
            # else:
            #     peer_id = peer_list_dht[peer_list_dht_addrs.index(miner_ip)].peer_id
            peer_id = peer_list_dht[peer_list_dht_addrs.index(miner_ip)].peer_id

        # If peer_id is not in the list of peer ids for our run then it is not connected to our run ID
        if str(peer_id) not in run_peer_id_list:
            # return None
            uids_to_peerids[uid] = None
            continue
        else:
            # return peer_id
            uids_to_peerids[uid] = peer_id
            continue

    return uids_to_peerids


def update_run_peerid_list(self):
    prefix = self.grad_averager.matchmaking_kwargs["prefix"]
    metadata, _ = self.dht.get(f"{prefix}.all_averagers", latest=True) or (
        {},
        None,
    )
    self.run_peer_id_list = [
        str(PeerID(peer_id))
        for peer_id, info in metadata.items()
        if isinstance(info, ValueWithExpiration)
        and isinstance(info.value, (float, int))
    ]


def map_uid_to_peerid(self):
    # Track how recently we updated each uid
    uid_last_checked = dict()
    # The below loop iterates across all miner uids and checks to see
    # if they should be updated.
    while not self.stop_event.is_set():
        try:
            # Get the next uid to check
            next_uid = next(self.uid_iterator)
            # Confirm that we haven't checked it in the last 5 minutes.
            time_diff = (
                dt.datetime.now() - uid_last_checked[next_uid]
                if next_uid in uid_last_checked
                else None
            )
            if time_diff and time_diff < dt.timedelta(minutes=5):
                # If we have seen it within 5 minutes then sleep until it has been at least 5 minutes.
                time_to_sleep = (dt.timedelta(minutes=5) - time_diff).total_seconds()
                bt.logging.trace(
                    f"Update loop has already processed all UIDs in the last 5 minutes. Sleeping {time_to_sleep} seconds."
                )
                time.sleep(time_to_sleep)

            uid_last_checked[next_uid] = dt.datetime.now()
            # Compare metadata and tracker, syncing new model from remote store to local if necessary.
            metadata = bt.core.extrinsics.serving.get_metadata(
                self.subtensor, self.config.netuid, self.metagraph.hotkeys[next_uid]
            )
            if metadata is not None:
                commitment = metadata["info"]["fields"][0]
                hex_data = commitment[list(commitment.keys())[0]][2:]
                chain_str = bytes.fromhex(hex_data).decode()
                updated = (chain_str, metadata["block"])
            else:
                updated = (None, None)

            if (self.uids_to_peerids[next_uid][0] != updated[0]) and (
                updated[0]
                not in [peerid_info[0] for peerid_info in self.uids_to_peerids.values()]
            ):
                bt.logging.info(
                    f"Updated peerID for UID={next_uid}. Previous = {self.uids_to_peerids[next_uid][0]}. Current = {updated[0]}"
                )
                self.uids_to_peerids[next_uid] = updated
            elif (self.uids_to_peerids[next_uid][0] != updated[0]) and (
                updated[0]
                in [peerid_info[0] for peerid_info in self.uids_to_peerids.values()]
            ):
                indices = [
                    index
                    for index, peerid_info in enumerate(self.uids_to_peerids.values())
                    if peerid_info[0] == updated[0]
                ]
                for index in indices:
                    if self.uids_to_peerids[index][1] > updated[1]:
                        self.uids_to_peerids[index] = (None, None)
                        bt.logging.info(
                            f"The same peerID was found for UID={index} with a later commit message. Setting the peerID for that UID={index} to None. Previous = {self.uids_to_peerids[next_uid][0]}. Current = {updated[0]}"
                        )
                        self.uids_to_peerids[next_uid] = updated
                        bt.logging.info(
                            f"Updated peerID for UID={next_uid}. Previous = {self.uids_to_peerids[next_uid][0]}. Current = {updated[0]}"
                        )
                        break
                    else:
                        updated = (None, None)
                        bt.logging.info(
                            f"The same peerID was found for UID={index} with an earlier commit message. Setting the peerID for UID={next_uid} to None. Previous = {self.uids_to_peerids[next_uid][0]}. Current = {updated[0]}"
                        )
                        self.uids_to_peerids[next_uid] = updated

        except Exception as e:
            bt.logging.error(f"Error in update loop: {e} \n {traceback.format_exc()}")
    bt.logging.info("Exiting update models loop.")
