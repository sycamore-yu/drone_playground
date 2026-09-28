"""Own one command application followed by its complete physical time interval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class ExecutionTransition:
    """Callbacks retain model-native state; this object owns their execution order.

    ``apply_command`` runs once per control interval. ``advance`` receives the
    number of physical substeps. Optional probes run at every actual substep and
    return clearance/collision evidence for the task to interpret.
    """

    apply_command: Callable
    advance: Callable
    substeps: int

    def __post_init__(self):
        if self.substeps < 1:
            raise ValueError("Execution requires at least one physical substep")

    def step(self, state, physical):
        controlled = self.apply_command(state, physical)
        return self.advance(controlled, self.substeps)

    def step_with_evidence(self, state, physical, probe):
        controlled = self.apply_command(state, physical)

        def substep(carry, index):
            current, collided = carry
            current = self.advance(current, 1)
            clearance, hit = probe(current, index)
            return (current, collided | hit), clearance

        (result, collided), clearances = jax.lax.scan(
            substep, (controlled, jnp.array(False)), jnp.arange(self.substeps)
        )
        return result, jnp.min(clearances), collided
