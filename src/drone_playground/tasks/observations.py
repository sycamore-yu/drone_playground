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


@dataclass(frozen=True)
class NavigationObservation:
    """Policy input for the navigation task.

    The body state, the goal direction and the previous normalized action are
    enumerated by the observation protocol. A sensor block is appended by the
    perception variants, which reuse this class's state fields unchanged so the
    proprioceptive input is identical across D435 and MID360 units.
    """

    name: str = "navigation_state"
    include_goal: bool = True
    include_previous_action: bool = True
    action_size: int = 4

    @property
    def size(self) -> int:
        return 13 + (3 if self.include_goal else 0) + (
            self.action_size if self.include_previous_action else 0
        )

    def __call__(self, states, goal, previous_action, extra=None):
        parts = [states.pos[0, 0], states.quat[0, 0], states.vel[0, 0], states.ang_vel[0, 0]]
        if self.include_goal:
            parts.append(goal - states.pos[0, 0])
        if self.include_previous_action:
            parts.append(previous_action)
        if extra is not None:
            parts.append(extra)
        return jnp.concatenate(parts)
