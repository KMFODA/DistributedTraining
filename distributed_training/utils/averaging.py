import asyncio
import time
from typing import Any, Dict, List, Tuple

import bittensor as bt
import numpy as np
import torch

import distributed_training
from distributed_training.exceptions import (
    GradientAveragingError,
    ModelStateError,
    TrainingError,
    handle_training_error,
    log_and_handle_error,
)

# TODO Remove redundant code after moving to diloco
# TODO Check exception handling and logging.debug instead of logging.info
class AveragingHandler:
    """Handles averaging round and outer step for both validators and miners."""

    def __init__(
        self,
        model,
        optimizer,
        grad_averager,
        state_averager,
        model_loading_manager=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.grad_averager = grad_averager
        self.state_averager = state_averager
        self.model_loading_manager = model_loading_manager

    def _get_weights_sample(self) -> List[float]:
        """Get a sample of model weights for validation."""
        return [layer for layer in self.model.parameters()][-2][-10:].tolist()

    async def _cleanup_failed_averaging(self, gradient_averaging_step):
        """Clean up after failed gradient averaging."""
        try:
            gradient_averaging_step.cancel()
            with self.grad_averager.use_averaged_gradients():
                self.optimizer.zero_grad()
            bt.logging.info("Gradient averaging cleanup completed")
        except Exception as e:
            log_and_handle_error(e, "_cleanup_failed_averaging")

    async def _validate_weight_update(self, initial_weights: List[float]) -> bool:
        """Validate model weight updates."""
        final_weights = self._get_weights_sample()

        if final_weights == initial_weights:
            raise ModelStateError("Weights unchanged after update")

        if sum(np.isnan(final_weights)) > 1:
            raise ModelStateError("NaN values detected in weights after update")

        return True

    @handle_training_error(
        error_types=(ModelStateError, GradientAveragingError, Exception),
        default_return=False,
    )
    async def _step_outer(self) -> bool:
        """Sync pseudo grads, step outer optimizer and validate the update."""
        with self.grad_averager.use_averaged_gradients():
            initial_weights = self._get_weights_sample()
            bt.logging.debug(f"Initial weights sample: {initial_weights}")

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            # Perform optimization steps
            bt.logging.info("Performing Outer Optimizer Step..")
            self.state_averager.step(increment_epoch=True, should_step_optimizer=True)
            self.state_averager.update_main_param_after_outer_step()
            bt.logging.succes("Finished Outer Optimizer Step!")

            # Reset gradient buffers
            self.grad_averager.reset_accumulated_grads_()

            # Validate weight updates
            await self._validate_weight_update(initial_weights)
            return True

    @handle_training_error(
        error_types=(ModelStateError, GradientAveragingError, TrainingError),
        default_return=(False, {}),
    )
    async def run_validator_allreduce(
        self, timeout: int, peerids_to_uids: Dict[str, str]
    ) -> Tuple[bool, Dict[str, Any]]:
        """Process allreduce specifically for validator."""
        # Start gradient averaging
        gradient_averaging_step = self.grad_averager.step(
            timeout=timeout, wait=False, peerids_to_uids=peerids_to_uids
        )

        try:
            start_time = time.perf_counter()

            # Wait for completion
            while (
                not gradient_averaging_step.done()
                and (time.perf_counter() - start_time) <= timeout
            ):
                await asyncio.sleep(1)

            if not gradient_averaging_step.done():
                raise GradientAveragingError("Gradient averaging timed out")

            # TODO Move to state_averager.step() ?
            # Get averaging results
            gathered, failed_peers, participating_peers = (
                gradient_averaging_step.result()
            )

            # Step state_averager/outer optimizer
            success = await self._step_outer()

            return success, {
                "gathered": gathered,
                "failed_peers": failed_peers,
                "participating_peers": participating_peers,
            }

        except Exception as e:
            await self._cleanup_failed_averaging(gradient_averaging_step)
            log_and_handle_error(e, "run_validator_allreduce")
            raise

    def calculate_allreduce_scores(
        self,
        participating_peers: list,
        failed_peers: list,
        peerids_to_uids: dict,
    ) -> dict:
        """
        Calculate scores based on AllReduce participation status.

        Args:
            participating_peers (list): List of peers that participated in AllReduce
            failed_peers (list): List of peers that failed during AllReduce
            peerids_to_uids (dict): Mapping of peer IDs to UIDs

        Returns:
            dict: Scores for each UID based on their participation status
        """
        # Convert peer IDs to UIDs
        participating_uids = [
            peerids_to_uids.get(str(participating_peer), "'''")
            for participating_peer in participating_peers
        ]
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

        # Initialize scores dictionary with float values for reward calculation
        scores = {}
        for uid in range(256):  # Assuming max 256 UIDs in metagraph
            if uid in participating_uids and uid not in failed_uids:
                scores[str(uid)] = 1.0  # Full score for successful participation
            elif uid in failed_uids:
                scores[str(uid)] = 0.0  # No score for failed participation
            else:
                scores[str(uid)] = 0.0  # No score for non-participation

        # Log participation details
        bt.logging.info(f"Failed UIDs: {failed_uids}")
        bt.logging.info(f"Participating UIDs: {participating_uids}")
        bt.logging.info(f"AllReduce UID Scores: {scores}")

        # Create status dictionary for model config (optional)
        status_dict = {}
        for uid in range(256):
            if uid in participating_uids and uid not in failed_uids:
                status_dict[str(uid)] = "SUCCESS"
            elif uid in failed_uids:
                status_dict[str(uid)] = "FAIL"
            else:
                status_dict[str(uid)] = "NON_PARTICIPATING"

        # Store status in model config if needed
        self.all_reduce_scores = status_dict

        return scores

    @staticmethod
    async def _wait_for_model_loading(model_loading_manager):
        """Wait for any ongoing model loading to complete."""
        if model_loading_manager:
            while model_loading_manager.is_loading:
                await asyncio.sleep(1)

    @handle_training_error(
        error_types=(ModelStateError, GradientAveragingError, TrainingError)
    )
    async def run_miner_allreduce(
        self, synapse, timeout: int
    ) -> distributed_training.protocol.AllReduce:
        """Process allreduce specifically for miner."""
        await self._wait_for_model_loading(self.model_loading_manager)

        if self.model_loading_manager:
            self.model_loading_manager.set_loading_state(True)

        gradient_averaging_step = None
        try:
            # Start gradient averaging
            gradient_averaging_step = self.grad_averager.step(
                timeout=timeout, wait=False
            )

            start_time = time.perf_counter()

            # Wait for completion
            while (
                not gradient_averaging_step.done()
                and (time.perf_counter() - start_time) <= timeout
            ):
                await asyncio.sleep(1)

            if not gradient_averaging_step.done():
                raise GradientAveragingError("Gradient averaging timed out")

            # Step state_averager/outer optimizer
            await self._step_outer()

            synapse.completion = "True"
            return synapse

        except Exception as e:
            synapse.completion = "False"
            if gradient_averaging_step:
                await self._cleanup_failed_averaging(gradient_averaging_step)
            log_and_handle_error(e, "run_miner_allreduce")
            raise
        finally:
            if self.model_loading_manager:
                self.model_loading_manager.set_loading_state(False)
