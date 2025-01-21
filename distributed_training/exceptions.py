import asyncio
import functools
import traceback
from typing import Optional

import bittensor as bt

from distributed_training.utils.state_loader import (
    load_state_from_peer,
)


class TrainingError(Exception):
    """Base exception class for training-related errors."""

    def __init__(self, message: str, original_error: Optional[Exception] = None):
        self.message = message
        self.original_error = original_error
        super().__init__(self.message)


class ModelStateError(TrainingError):
    """Raised when there are issues with model state loading/saving."""

    pass


class StateAveragingError(TrainingError):
    """Raised when gradient averaging fails."""

    pass


class NetworkError(TrainingError):
    """Raised for network-related issues (DHT, peer communication)."""

    pass


def handle_error(error_types: tuple = (Exception,), default_return=None):
    """
    Decorator for standardized error handling in training functions.
    Automatically triggers state loading for StateAveragingError and ModelStateError.

    Args:
        error_types: Tuple of exception types to catch
        default_return: Value to return on error
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except error_types as e:
                error_name = e.__class__.__name__
                error_msg = str(e)
                tb = traceback.format_exc()

                bt.logging.error(f"{error_name} in {func.__name__}: {error_msg}\n{tb}")

                # Automatically load state for averaging/model state errors
                if isinstance(e, (StateAveragingError, ModelStateError)):
                    bt.logging.info("Loading latest model state after failure")
                    # Use run_coroutine_threadsafe since we're in a thread
                    asyncio.run_coroutine_threadsafe(
                        load_state_from_peer(self), self.loop
                    )

                return default_return

        return wrapper

    return decorator


def log_and_handle_error(error: Exception, context: str = "") -> None:
    """
    Standardized error logging and handling.

    Args:
        error: The exception to handle
        context: Additional context about where the error occurred
    """
    error_type = error.__class__.__name__
    error_msg = str(error)
    tb = traceback.format_exc()

    bt.logging.error(f"Error in {context}: {error_type} - {error_msg}\n{tb}")
