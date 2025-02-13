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
