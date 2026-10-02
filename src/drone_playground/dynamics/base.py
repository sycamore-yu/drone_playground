"""Native state/command transitions used by LOTF and the acceleration point mass.

State types remain model-specific. Crazyflow instead advances a bound simulation
container with staged controls through ``advance(data, substeps)``; it does not
implement this native-state interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class DynamicsBackend(ABC):
    """Native transition with an explicitly selected derivative rule."""

    forward: str
    backward: str

    @abstractmethod
    def step(self, state, command, dt):
        """Advance the physical state by one transition."""
