import asyncio
import time
from typing import Any, Dict, List, Tuple

import bittensor as bt
import numpy as np
import torch

import distributed_training
from distributed_training.exceptions import (
    ModelStateError,
    StateAveragingError,
)
from distributed_training.protocol import AllReduce
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
        self.model_loading_manager = model_loading_manager

    def _get_weights_sample(self) -> List[float]:
        """Get a sample of model weights for validation."""
        return [layer for layer in self.model.parameters()][-2][-10:].tolist()

    async def _validate_weight_update(self, initial_weights: List[float]) -> bool:
        """Validate model weight updates."""
        final_weights = self._get_weights_sample()

        if final_weights == initial_weights:
            raise ModelStateError("Weights unchanged after update")

        if sum(np.isnan(final_weights)) > 1:
            raise ModelStateError("NaN values detected in weights after update")

        return True

    async def run_validator_allreduce(
        self, timeout: int, dendrite_pool, miner_uids, bandwidth
    ) -> Tuple[bool, Dict[str, Any]]:
        """Process allreduce specifically for validator."""
        # TODO Weight/gradient validation
        gradient_averaging_step = None
        query_tasks = []

        try:
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            # Used for load balancing and scoring
            self.grad_averager.bandwidth = bandwidth[
                "download"
            ]  # TODO Either use average bandwidth or set each time here

            bt.logging.info("Starting Pseudo Gradient Averaging..")
            gradient_averaging_step = self.grad_averager.step(
                wait=False,
                timeout=timeout,
            )

            async with asyncio.timeout(timeout):
                # Send AllReduce query to pause miner training and perform global sync
                query_tasks.append(
                    dendrite_pool.async_forward(
                        miner_uids,
                        [AllReduce() for _ in miner_uids],
                        timeout=timeout,
                    )
                )
                bt.logging.info(
                    ":wait: AllReduce Query Sent Out. Waiting for AllReduce to finish.."
                )

                await asyncio.gather(*query_tasks)
                bt.logging.info("AllReduce Query Responses Received..")

            self.grad_averager.notify_used_averaged_gradients()
            bt.logging.success("Finished Averaging Pseudo Gradients!")

            if gradient_averaging_step.done():
                (
                    failed_peers,
                    participating_peers,
                    modes,
                    bandwidths,
                ) = gradient_averaging_step.result()

                initial_weights = self._get_weights_sample()
                bt.logging.debug(f"Initial weights sample: {initial_weights}")

                # Perform offloaded outer optimization steps
                bt.logging.info("Performing Outer Optimizer Step..")
                self.state_averager.step(
                    increment_epoch=True, optimizer_step=True, zero_grad=False
                )
                self.state_averager.update_main_param_after_outer_step()
                self.state_averager.optimizer.zero_grad()
                bt.logging.success(
                    ":white_heavy_check_mark: Finished Outer Optimizer Step!"
                )

                # Validate weight updates
                await self._validate_weight_update(initial_weights)

                return {
                    "failed_peers": failed_peers,
                    "participating_peers": participating_peers,
                    "modes": modes,
                    "bandwidths": bandwidths,
                }

            else:  # TODO Ensure failing avaerging is handled
                raise ModelStateError(
                    "Averaging Failed. Loading Updated Model State From HF."
                )

        except asyncio.TimeoutError:
            raise GradientAveragingTimeoutError(
                "AllReduce operation timed out waiting for miners to complete"
            )
        except Exception as e:
            if isinstance(e, AllReduceError):
                raise
            raise AllReduceError(f"Unexpected error during AllReduce: {str(e)}")
        finally:
            if gradient_averaging_step and not gradient_averaging_step.done():
                gradient_averaging_step.cancel()
                await asyncio.sleep(10)
                load_state_from_peer(
                    self, epoch=self.global_progress.epoch
                )  # TODO self is not available here

    def calculate_allreduce_scores(
        self,
        participating_peers: list,
        failed_peers: list,
        modes: list,
        bandwidths: list,
        peerids_to_uids: dict,
    ) -> dict:
        """
        Calculate scores based on AllReduce participation status, modes, and bandwidths.

        Args:
            participating_peers (list): List of peers that participated in AllReduce
            failed_peers (list): List of peers that failed during AllReduce
            modes (list): List of modes for each participating peer
            bandwidths (list): List of bandwidths for each participating peer
            peerids_to_uids (dict): Mapping of peer IDs to UIDs

        Returns:
            dict: Scores for each UID based on participation, mode, and bandwidth
        """
        # Convert peer IDs to UIDs and create mode/bandwidth mappings
        participating_uids = []
        uid_modes = {}
        uid_bandwidths = {}

        for idx, peer in enumerate(participating_peers):
            uid = peerids_to_uids.get(str(peer), "'''")
            participating_uids.append(uid)
            uid_modes[uid] = modes[idx]
            uid_bandwidths[uid] = bandwidths[idx]

        failed_uids = [
            peerids_to_uids.get(str(failed_peer), "'''") for failed_peer in failed_peers
        ]

        # Calculate participation metrics
        successful_peers_count = len(participating_peers) - len(failed_peers)

        # Update event metrics
        self.event.update(
            {
                "failed_peers_count": len(failed_peers),
                "participating_peers_count": len(participating_peers),
                "successful_peers_count": successful_peers_count,
            }
        )

        # Find max bandwidth for normalization
        max_bandwidth = max(bandwidths) if bandwidths else 1.0

        # Initialize scores dictionary
        scores = {}
        status_dict = {}

        for uid in range(256):  # Assuming 256 UIDs in metagraph
            str_uid = str(uid)
            if uid in participating_uids and uid not in failed_uids:
                # Check if mode is not CLIENT
                if uid_modes[uid] == "AveragingMode.CLIENT":
                    scores[str_uid] = 0.0
                    status_dict[str_uid] = "WRONG_MODE"
                else:
                    # Base score for successful participation
                    base_score = 1.0
                    # Add normalized bandwidth bonus (up to 0.5 additional score)
                    bandwidth_bonus = 0.5 * (uid_bandwidths[uid] / max_bandwidth)
                    scores[str_uid] = base_score + bandwidth_bonus
                    status_dict[str_uid] = "SUCCESS"

                    bt.logging.debug(
                        f"UID {uid} score breakdown - Base: {base_score:.2f}, Bandwidth bonus: {bandwidth_bonus:.2f}"
                    )

            elif uid in failed_uids:
                scores[str_uid] = 0.0
                status_dict[str_uid] = "FAIL"
            else:
                scores[str_uid] = 0.0
                status_dict[str_uid] = "NON_PARTICIPATING"

        # Log participation and scoring details
        bt.logging.info(f"Failed UIDs: {failed_uids}")
        bt.logging.info(f"Participating UIDs: {participating_uids}")
        bt.logging.debug(f"Modes by UID: {uid_modes}")
        bt.logging.debug(f"Bandwidths by UID: {uid_bandwidths}")
        bt.logging.info(f"AllReduce UID Scores: {scores}")

        # Store status in model config
        self.all_reduce_scores = status_dict

        return scores

    @staticmethod
    async def _wait_for_model_loading(model_loading_manager):
        """Wait for any ongoing model loading to complete."""
        if model_loading_manager:
            while model_loading_manager.is_loading:
                await asyncio.sleep(1)

    async def run_miner_allreduce(
        self, synapse, bandwidth
    ) -> distributed_training.protocol.AllReduce:
        """Process allreduce specifically for miner."""
        await self._wait_for_model_loading(self.model_loading_manager)

        if self.model_loading_manager:
            self.model_loading_manager.set_loading_state(True)
        # TODO Weight/gradient validation
        gradient_averaging_step = None
        try:
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            # Used for load balancing and scoring
            self.grad_averager.bandwidth = bandwidth[
                "download"
            ]  # TODO Either use average bandwidth or set each time here

            bt.logging.info(":wait: Starting Pseudo Gradient Averaging..")
            gradient_averaging_step = self.grad_averager.step(
                wait=True, timeout=synapse.timeout
            )

            try:
                # Wait for gradient averaging with timeout
                await asyncio.wait_for(
                    asyncio.wrap_future(gradient_averaging_step),
                    timeout=synapse.timeout,
                )
            except asyncio.TimeoutError:
                raise GradientAveragingTimeoutError("Gradient averaging step timed out")
            except Exception as e:
                raise GradientAveragingError(f"Gradient averaging failed: {str(e)}")

            self.grad_averager.notify_used_averaged_gradients()
            bt.logging.success("Finished Averaging Pseudo Gradients!")

            initial_weights = self._get_weights_sample()
            bt.logging.debug(f"Initial weights sample: {initial_weights}")

            # Perform offloaded outer optimization steps
            bt.logging.info("Performing Outer Optimizer Step..")
            self.state_averager.step(
                increment_epoch=True, optimizer_step=True, zero_grad=False
            )
            self.state_averager.update_main_param_after_outer_step()
            self.state_averager.optimizer.zero_grad()
            bt.logging.success("Finished Outer Optimizer Step!")
            bt.logging.success(
                ":white_heavy_check_mark: Finished Outer Optimizer Step!"
            )

            # Validate weight updates
            await self._validate_weight_update(initial_weights)

            synapse.completion = "True"
            return synapse

        except Exception as e:
            if isinstance(e, AllReduceError):
                raise
            raise AllReduceError(f"Unexpected error during AllReduce: {str(e)}")

        finally:
            if gradient_averaging_step and not gradient_averaging_step.done():
                gradient_averaging_step.cancel()
            if self.model_loading_manager:
                self.model_loading_manager.set_loading_state(False)
