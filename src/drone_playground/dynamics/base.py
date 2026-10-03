"""One physical transition contract, with backend-native state and integrators."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod


class DynamicsBackend(ABC):
    """Native transition with an explicitly selected derivative rule."""

    forward: str
    backward: str

    @abstractmethod
    def step(self, state, control, dt):
        """Advance native state by dt seconds using a physical typed control."""


def physics_steps(dt: float, frequency: int) -> int:
    """Resolve a static interval without rounding away a native physics tick."""
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be a positive multiple of the native physics timestep")
    count = round(dt * frequency)
    if count < 1 or not math.isclose(count / frequency, dt, rel_tol=0, abs_tol=1e-9):
        raise ValueError("dt must be a positive multiple of the native physics timestep")
    return count
