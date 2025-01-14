import functools
import traceback
from typing import Optional

import bittensor as bt


class TrainingError(Exception):
    """Base exception class for training-related errors."""

    def __init__(self, message: str, original_error: Optional[Exception] = None):
        self.message = message
        self.original_error = original_error
        super().__init__(self.message)


class ModelStateError(TrainingError):
    """Raised when there are issues with model state loading/saving."""

    pass


class GradientAveragingError(TrainingError):
    """Raised when gradient averaging fails."""

    pass


class NetworkError(TrainingError):
    """Raised for network-related issues (DHT, peer communication)."""

    pass


def handle_training_error(error_types: tuple = (Exception,), default_return=None):
    """
    Decorator for standardized error handling in training functions.

    Args:
        error_types: Tuple of exception types to catch
        default_return: Value to return on error
    """

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except error_types as e:
                error_name = e.__class__.__name__
                error_msg = str(e)
                tb = traceback.format_exc()

                bt.logging.error(f"{error_name} in {func.__name__}: {error_msg}\n{tb}")

                if isinstance(e, ModelStateError):
                    # Trigger model reload
                    if len(args) > 0 and hasattr(args[0], "load_state_from_peer"):
                        await args[0].load_state_from_peer()

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
