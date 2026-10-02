"""Task interfaces shared by learning, control and independent evaluation."""

from drone_playground.environments.tasks.tracking.rigid_body import TrackingEnv

__all__ = ["TrackingEnv", "wrap_for_training"]
