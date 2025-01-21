import asyncio
from typing import Any, Dict, List, Tuple

import bittensor as bt
import numpy as np
import torch

import distributed_training
from distributed_training.exceptions import (
    ModelStateError,
    StateAveragingError,
    handle_error,
    log_and_handle_error,
)


# TODO Remove/cleanup redundant code after moving to diloco
class AveragingHandler:
    """Handles averaging round and outer step for both validators and miners."""

    def __init__(
        self,
        model,
        outer_optimizer,
        grad_averager,
        state_averager,
        model_loading_manager=None,
    ):
        self.model = model
        self.outer_optimizer = outer_optimizer
        self.grad_averager = grad_averager
        self.state_averager = state_averager
        self.model_loading_manager = model_loading_manager

    def _get_weights_sample(self) -> List[float]:
        """Get a sample of model weights for validation."""
        return [layer for layer in self.model.parameters()][-2][-10:].tolist()

    async def _cleanup_failed_averaging(self, gradient_averaging_step):
        # TODO Not sure if we should zero_grads on the outer optimizer?
        """Clean up after failed gradient averaging."""
        try:
            gradient_averaging_step.cancel()
            with self.grad_averager.use_averaged_gradients():
                self.outer_optimizer.zero_grad()
            bt.logging.debug("Gradient averaging cleanup completed")
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

    @handle_error(
        error_types=(ModelStateError, StateAveragingError, Exception),
        default_return=(False, {}),
    )
    async def run_validator_allreduce(
        self,
        timeout: int,
    ) -> Tuple[bool, Dict[str, Any]]:
        """Process allreduce specifically for validator."""
        # TODO Weight/gradient validation 
        gradient_averaging_step = None
        try:
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            gradient_averaging_step = self.grad_averager.step(
                wait=True, timeout=timeout
            )
            self.grad_averager.notify_used_averaged_gradients()
            bt.logging.success("Finished Averaging Pseudo Gradients!")

            (
                gathered,
                failed_peers,
                participating_peers,
            ) = gradient_averaging_step.result()

            initial_weights = self._get_weights_sample()
            bt.logging.debug(f"Initial weights sample: {initial_weights}")

            # Perform offloaded outer optimization steps
            bt.logging.info("Performing Outer Optimizer Step..")
            self.state_averager.step(
                increment_epoch=True, optimizer_step=True, zero_grad=False
            )
            self.state_averager.update_main_param_after_outer_step()
            self.outer_optimizer.zero_grad()
            bt.logging.success(":white_heavy_check_mark: Finished Outer Optimizer Step!")
            
            # Reset gradient buffers #TODO Do we need this?
            self.grad_averager.reset_accumulated_grads_()

            # Validate weight updates
            await self._validate_weight_update(initial_weights)

            return {
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

    @handle_error(error_types=(ModelStateError, StateAveragingError, Exception))
    async def run_miner_allreduce(
        self, synapse,
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

            gradient_averaging_step = self.grad_averager.step(
                wait=True, timeout=synapse.timeout
            )
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
            self.outer_optimizer.zero_grad()
            bt.logging.success("Finished Outer Optimizer Step!")

            # Reset gradient buffers
            self.grad_averager.reset_accumulated_grads_()

            # Validate weight updates
            await self._validate_weight_update(initial_weights)

            synapse.completion = "True"
            return synapse

        except Exception as e:
            await self._cleanup_failed_averaging(gradient_averaging_step)
            log_and_handle_error(e, "run_validator_allreduce")
            raise

        finally:
            if self.model_loading_manager:
                self.model_loading_manager.set_loading_state(False)
