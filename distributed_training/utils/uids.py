import torch
import random
import bittensor as bt
from typing import List
import traceback
import asyncio
import distributed_training
from hivemind.utils.timed_storage import ValueWithExpiration
from hivemind.p2p import PeerID


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


async def map_uid_to_peerid(self, uids):
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
