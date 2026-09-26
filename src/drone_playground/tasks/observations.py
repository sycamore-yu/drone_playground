"""Observation encoders are shared by training and frozen evaluation."""

from dataclasses import dataclass

import jax.numpy as jnp


@dataclass(frozen=True)
class TrackingObservation:
    name: str = "state_reference"
    n_samples: int = 10
    interval: float = 0.1

    @property
    def size(self):
        return 13 + (3 * self.n_samples if self.name == "state_reference" else 0)

    def __call__(self, state, references):
        parts = [state.pos[0, 0], state.quat[0, 0], state.vel[0, 0], state.ang_vel[0, 0]]
        if self.name == "state_reference":
            parts.append((references - state.pos[0, 0]).reshape(-1))
        elif self.name != "state":
            raise ValueError(f"Unknown observation encoder: {self.name}")
        return jnp.concatenate(parts)
