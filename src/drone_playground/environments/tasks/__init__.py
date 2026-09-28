"""Task interfaces shared by learning, control and independent evaluation."""

from .tracking import TrackingEnv

__all__ = ["TrackingEnv", "wrap_for_training"]
