import bisect
import copy
import functools
import multiprocessing
import random
import threading
from typing import Any, List, Optional

import bittensor as bt


class MinerIterator:
    """A thread safe infinite iterator to cyclically enumerate the current set of miner UIDs.

    Why? To perform miner evaluations, the validator will enumerate through the miners in order to help ensure
    each miner is evaluated at least once per epoch.
    """

    def __init__(self, miner_uids: List[int]):
        self.miner_uids = sorted(copy.deepcopy(miner_uids))
        # Start the index at a random position. This helps ensure that miners with high UIDs aren't penalized if
        # the validator restarts frequently.
        self.index = random.randint(0, len(self.miner_uids) - 1)
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self) -> int:
        with self.lock:
            if len(self.miner_uids) == 0:
                # This iterator should be infinite. If there are no miner UIDs, raise an error.
                raise IndexError("No miner UIDs.")

            uid = self.miner_uids[self.index]
            self.index += 1
            if self.index >= len(self.miner_uids):
                self.index = 0
            return uid

    def peek(self) -> int:
        """Returns the next miner UID without advancing the iterator."""
        with self.lock:
            if len(self.miner_uids) == 0:
                # This iterator should be infinite. If there are no miner UIDs, raise an error.
                raise IndexError("No miner UIDs.")

            return self.miner_uids[self.index]

    def set_miner_uids(self, miner_uids: List[int]):
        """Updates the miner UIDs to iterate.

        The iterator will be updated to the first miner uid that is greater than or equal to UID that would be next
        returned by the iterator. This helps ensure that frequent updates to the miner_uids does not cause too much
        churn in the sequence of UIDs returned by the iterator.
        """
        sorted_uids = sorted(copy.deepcopy(miner_uids))
        with self.lock:
            next_uid = self.miner_uids[self.index]
            new_index = bisect.bisect_left(sorted_uids, next_uid)
            if new_index >= len(sorted_uids):
                new_index = 0
            self.index = new_index
            self.miner_uids = sorted_uids


def _wrapped_func(func: functools.partial, queue: multiprocessing.Queue):
    try:
        result = func()
        queue.put(result)
    except (Exception, BaseException) as e:
        # Catch exceptions here to add them to the queue.
        queue.put(e)


def run_in_subprocess(func: functools.partial, ttl: int, mode="fork") -> Any:
    """Runs the provided function on a subprocess with 'ttl' seconds to complete.

    Args:
        func (functools.partial): Function to be run.
        ttl (int): How long to try for in seconds.

    Returns:
        Any: The value returned by 'func'
    """
    ctx = multiprocessing.get_context(mode)
    queue = ctx.Queue()
    process = ctx.Process(target=_wrapped_func, args=[func, queue])

    process.start()

    process.join(timeout=ttl)

    if process.is_alive():
        process.terminate()
        process.join()
        raise TimeoutError(f"Failed to {func.func.__name__} after {ttl} seconds")

    # Raises an error if the queue is empty. This is fine. It means our subprocess timed out.
    result = queue.get(block=False)

    # If we put an exception on the queue then raise instead of returning.
    if isinstance(result, Exception):
        raise result
    if isinstance(result, BaseException):
        raise Exception(f"BaseException raised in subprocess: {str(result)}")

    return result
