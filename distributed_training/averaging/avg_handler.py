import asyncio
import time
from typing import Any, Dict, List, Tuple

import bittensor as bt
import distributed_training
import numpy as np
import torch
from distributed_training.protocol import AllReduce
from distributed_training.utils.progress_tracker import get_global_epoch
from distributed_training.utils.state_loader import load_state_from_peer


class AllReduceError(Exception):
    """Base exception for AllReduce-related errors."""

    pass


class GradientAveragingTimeoutError(AllReduceError):
    """Raised when gradient averaging step times out."""

    pass


class GradientAveragingError(AllReduceError):
    """Raised when gradient averaging fails for non-timeout reasons."""

    pass


class StateAveragingError(AllReduceError):
    """Raised when state averaging fails."""

    pass


class ModelStateError(AllReduceError):
    """Raised when model weights are corrupted after an all reduce."""

    pass


# TODO cleanup code after moving to diloco
class AveragingHandler:
    """Handles averaging round and outer step for both validators and miners."""

    def __init__(
        self,
        model,
        grad_averager,
        state_averager,
        model_loading_manager=None,
    ):
        self.model = model
        self.grad_averager = grad_averager
        self.state_averager = state_averager

    def _get_weights_sample(self) -> List[float]:
        """Get a sample of model weights for validation."""
        return [layer for layer in self.model.parameters()][-2][-10:].tolist()

    def _validate_weight_update(self, initial_weights: List[float]) -> bool:
        """Validate model weight updates."""
        final_weights = self._get_weights_sample()
        bt.logging.info(f"Final Weights Sample: {final_weights}")

        if final_weights == initial_weights:
            raise ModelStateError("Weights unchanged after update")

        if sum(np.isnan(final_weights)) > 1:
            raise ModelStateError("NaN values detected in weights after update")

    async def run_validator_allreduce(
        self,
        timeout: int,
        dendrite_pool,
        peerids_to_uids,
        miner_uids,
        bandwidth=None,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Process allreduce specifically for validator.

        Returns:
            Tuple[bool, Dict[str, Any]]: (success, results)
            - success: True if allreduce completed successfully, False otherwise
            - results: Dictionary containing peers and bandwidth info if successful, empty dict if failed
        """
        query_tasks = []
        all_reduce_success_status = True
        results = {}

        try:
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            # Used for load balancing and scoring
            if bandwidth is not None:
                self.grad_averager.bandwidth = bandwidth["download"]

            bt.logging.info("Starting Pseudo Gradient Averaging..")
            # Start gradient averaging without waiting
            gradient_averaging_step = self.grad_averager.step(
                gather=0,
                wait=False,
                peerids_to_uids=peerids_to_uids,
            )

            # Send AllReduce query to pause miner training and perform global sync
            query_tasks.append(
                dendrite_pool.async_forward(
                    miner_uids,
                    [AllReduce(completion=False) for _ in miner_uids],
                    timeout=timeout,
                )
            )
            bt.logging.info(
                ":wait: AllReduce Query Sent Out. Waiting for AllReduce to finish.."
            )
            start_time = time.perf_counter()
            responses = await asyncio.gather(*query_tasks)
            bt.logging.info("AllReduce Query Responses Received..")

            while (gradient_averaging_step.done() is False) and (
                (time.perf_counter() - start_time) <= (timeout)
            ):
                time.sleep(1)

            if gradient_averaging_step.done():
                bt.logging.success(
                    ":white_heavy_check_mark: Finished Averaging Pseudo Gradients"
                )
                (
                    gathered,
                    failed_peers,
                    participating_peers,
                    modes,
                    bandwidths,
                ) = gradient_averaging_step.result()

                initial_weights = self._get_weights_sample()
                bt.logging.info(f"Initial Weights Sample: {initial_weights}")

                # Perform offloaded outer optimization steps
                bt.logging.info(":wait: Performing Outer Optimizer Step")
                self.state_averager.step(
                    increment_epoch=True, optimizer_step=True, zero_grad=False
                )
                self.state_averager.update_main_param_after_outer_step()
                self.state_averager.optimizer.zero_grad()
                bt.logging.success(
                    ":white_heavy_check_mark: Finished Outer Optimizer Step."
                )

                # Validate weight updates
                self._validate_weight_update(initial_weights)

                all_reduce_success_status = True
                results = {
                    "gathered": gathered,
                    "failed_peers": failed_peers,
                    "participating_peers": participating_peers,
                    "modes": modes,
                    "bandwidths": bandwidths,
                }
            else:
                all_reduce_success_status = False

        except Exception as e:
            bt.logging.error(f"Error during AllReduce setup: {str(e)}")
            all_reduce_success_status = False

        finally:
            if gradient_averaging_step:
                gradient_averaging_step.cancel()
                bt.logging.info(":white_heavy_check_mark: Gradient Step Cancelled")
            self.state_averager.optimizer.zero_grad()
            return all_reduce_success_status, results

    def calculate_allreduce_scores(
        self,
        participating_peers: list,
        failed_peers: list,
        peerids_to_uids: dict,
        event: dict,
        metagraph,
        modes: list = None,
        bandwidths: list = None,
    ) -> dict:
        """
        Calculate scores based on AllReduce participation status, modes, and bandwidths.

        Args:
            participating_peers (list): List of peers that participated in AllReduce
            failed_peers (list): List of peers that failed during AllReduce
            peerids_to_uids (dict): Mapping of peer IDs to UIDs
            modes (list, optional): List of modes for each participating peer
            bandwidths (list, optional): List of bandwidths for each participating peer

        Returns:
            dict: Scores for each UID based on participation and optional mode/bandwidth
        """
        # Convert peer IDs to UIDs
        participating_uids = []
        uid_modes = {}
        uid_bandwidths = {}

        for idx, peer in enumerate(participating_peers):
            uid = peerids_to_uids.get(str(peer), "'''")
            participating_uids.append(uid)
            if modes is not None:
                uid_modes[uid] = modes[idx]
            if bandwidths is not None:
                uid_bandwidths[uid] = bandwidths[idx]

        failed_uids = [
            peerids_to_uids.get(str(failed_peer), "'''") for failed_peer in failed_peers
        ]

        # Calculate participation metrics
        successful_peers_count = len(participating_peers) - len(failed_peers)

        # Update event metrics
        event.update(
            {
                "failed_peers_count": len(failed_peers),
                "participating_peers_count": len(participating_peers),
                "successful_peers_count": successful_peers_count,
            }
        )

        # Find max bandwidth for normalization if bandwidths are provided
        if (
            bandwidths
            and [bandwidth for bandwidth in bandwidths if bandwidth is not None] != []
            and max([bandwidth for bandwidth in bandwidths if bandwidth is not None])
            != []
        ):
            max_bandwidth = max(
                [bandwidth for bandwidth in bandwidths if bandwidth is not None]
            )

        # Initialize scores dictionary
        scores = {}
        status_dict = {}
        for uid in range(metagraph.n):  # Assuming 256 UIDs in metagraph
            str_uid = str(uid)
            if uid in participating_uids and uid not in failed_uids:
                # Base score for successful participation
                base_score = 1.0
                final_score = base_score
                status = "SUCCESS"

                # Apply mode penalty if modes are provided
                if modes is not None and uid in uid_modes:
                    if uid_modes[uid] == "AveragingMode.CLIENT":
                        final_score = 0.0
                        status = "WRONG_MODE"

                # Apply bandwidth bonus if bandwidths are provided
                if (
                    bandwidths is not None
                    and uid in uid_bandwidths
                    and status != "WRONG_MODE"
                ):
                    if uid_bandwidths[uid] is None:
                        final_score = 0.0
                    else:
                        bandwidth_bonus = 0.5 * (uid_bandwidths[uid] / max_bandwidth)
                        final_score += bandwidth_bonus
                        bt.logging.info(
                            f"UID {uid} score breakdown - Base: {base_score:.2f}, Bandwidth bonus: {bandwidth_bonus:.2f}"
                        )

                scores[str_uid] = 1.0
                status_dict[str_uid] = status

            elif uid in failed_uids:
                scores[str_uid] = 0.0
                status_dict[str_uid] = "FAIL"
            else:
                scores[str_uid] = 0.0
                status_dict[str_uid] = "NON_PARTICIPATING"

        # Log participation and scoring details
        bt.logging.info(f"Failed UIDs: {failed_uids}")
        bt.logging.info(f"Participating UIDs: {participating_uids}")
        if modes is not None:
            bt.logging.info(f"Modes by UID: {uid_modes}")
        if bandwidths is not None:
            bt.logging.info(f"Bandwidths by UID: {uid_bandwidths}")
        bt.logging.info(f"AllReduce UID Scores: {scores}")

        return scores, event

    async def run_miner_allreduce(
        self,
        synapse,
        local_progress,
        bandwidth=None,
    ) -> distributed_training.protocol.AllReduce:
        """Process allreduce specifically for miner."""
        try:
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            # Used for load balancing and scoring
            if bandwidth is not None:
                self.grad_averager.bandwidth = bandwidth["download"]

            bt.logging.info(":wait: Starting Pseudo Gradient Averaging..")
            gradient_averaging_step = self.grad_averager.step(
                timeout=(synapse.timeout - 20),
                wait=False,
                gather=local_progress.samples_accumulated,
            )
            start_time = time.perf_counter()

            while (gradient_averaging_step.done() is False) and (
                (time.perf_counter() - start_time) <= (synapse.timeout)
            ):
                time.sleep(1)

            if gradient_averaging_step.done():
                bt.logging.success(
                    ":white_heavy_check_mark: Finished Averaging Pseudo Gradients"
                )
                initial_weights = self._get_weights_sample()
                bt.logging.info(f"Initial Weights Sample: {initial_weights}")

                # Perform offloaded outer optimization steps
                bt.logging.info(":wait: Performing Outer Optimizer Step")
                self.state_averager.step(
                    increment_epoch=True, optimizer_step=True, zero_grad=False
                )
                self.state_averager.update_main_param_after_outer_step()
                self.state_averager.optimizer.zero_grad()
                bt.logging.success(
                    ":white_heavy_check_mark: Finished Outer Optimizer Step."
                )

                # Validate weight updates
                self._validate_weight_update(initial_weights)
                synapse.completion = True
            else:
                synapse.completion = False

        except Exception as e:
            synapse.completion = False
            raise AllReduceError(f"Unexpected error during AllReduce: {str(e)}") from e

        finally:
            if gradient_averaging_step:
                gradient_averaging_step.cancel()
                bt.logging.info(":white_heavy_check_mark: Gradient Step Cancelled")
            self.state_averager.optimizer.zero_grad()
            return synapse
