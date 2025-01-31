import bisect
import copy
import random
import threading
from typing import List

import bittensor as bt


class UIDIterator:
    """A thread safe infinite iterator to cyclically enumerate the current set of miner UIDs.
    Why? To perform miner evaluations, the validator will enumerate through the miners in order to help ensure
    each miner is evaluated at least once per epoch.
    """

    def __init__(self, uids: List[int]):
        self.uids = sorted(copy.deepcopy(uids))
        # Start the index at a random position. This helps ensure that miners with high UIDs aren't penalized if
        # the validator restarts frequently.
        self.index = random.randint(0, len(self.uids) - 1)
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self) -> int:
        with self.lock:
            if len(self.uids) == 0:
                # This iterator should be infinite. If there are no miner UIDs, raise an error.
                raise IndexError("No miner UIDs.")
            uid = self.uids[self.index]
            self.index += 1
            if self.index >= len(self.uids):
                self.index = 0
            return uid

    def peek(self) -> int:
        """Returns the next miner UID without advancing the iterator."""
        with self.lock:
            if len(self.uids) == 0:
                # This iterator should be infinite. If there are no miner UIDs, raise an error.
                raise IndexError("No miner UIDs.")
            return self.uids[self.index]

    def set_uids(self, uids: List[int]):
        """Updates the miner UIDs to iterate.
        The iterator will be updated to the first miner uid that is greater than or equal to UID that would be next
        returned by the iterator. This helps ensure that frequent updates to the uids does not cause too much
        churn in the sequence of UIDs returned by the iterator.
        """
        sorted_uids = sorted(copy.deepcopy(uids))
        with self.lock:
            next_uid = self.uids[self.index]
            new_index = bisect.bisect_left(sorted_uids, next_uid)
            if new_index >= len(sorted_uids):
                new_index = 0
            self.index = new_index
            self.uids = sorted_uids


async def get_chain_metadata(self, uid):
    metadata = bt.extrinsics.serving.get_metadata(
        self.subtensor, self.config.netuid, self.metagraph.hotkeys[uid]
    )
    if metadata is not None:
        commitment = metadata["info"]["fields"][0]
        hex_data = commitment[list(commitment.keys())[0]][2:]
        chain_str = bytes.fromhex(hex_data).decode()
        bt.logging(f"{uid}:{chain_str}")
        return chain_str
    else:
        bt.logging(f"{uid}:None")
        return None


def log_peerid_to_chain(self):
    try:
        metadata = {
            "peer_id": self.dht.peer_id.to_base58(),
            "model_huggingface_id": self.config.neuron.hf_repo_id,
        }
        self.subtensor.commit(self.wallet, self.config.netuid, str(metadata))
        self.peer_id_logged_to_chain = True
        bt.logging.info(f"Metadata dict {metadata} succesfully logged to chain.")
    except Exception:
        self.peer_id_logged_to_chain = False
        bt.logging.error(
            "Unable to log DHT PeerID to chain. Retrying on the next step."
        )
