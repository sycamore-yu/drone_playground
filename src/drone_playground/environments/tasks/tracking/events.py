"""First terminal event at each physics tick, shared by rollout and evaluation."""

import jax
import jax.numpy as jnp


def _select(mask, new, old):
    return jax.tree.map(
        lambda n, o: jnp.where(mask.reshape(mask.shape + (1,) * (n.ndim - mask.ndim)), n, o),
        new,
        old,
    )


def tracking_events(task, previous, proposed, time, passed):
    """Return tracking/racing events for one observed physical state transition."""
    del time
    finite = jnp.all(jnp.isfinite(proposed.vector()), axis=-1)
    clearance = task.clearance(proposed.pos)
    outside = jnp.any(
        (proposed.pos < task.bounds_low) | (proposed.pos > task.bounds_high),
        axis=-1,
    )
    event = jnp.where(~finite, 4, jnp.where(clearance < 0, 2, jnp.where(outside, 3, 0))).astype(
        jnp.int32
    )
    if task.name == "racing":
        from drone_playground.environments.tasks.lsy_upstream.utils import gate_passed

        order = jnp.minimum(passed, len(task.gate_order) - 1)
        gate_id = task.gate_order[order]
        crossed = gate_passed(
            proposed.pos,
            previous.pos,
            task.gate_positions[gate_id],
            task.gate_quaternions[gate_id],
            task.gate_reverse[order],
            (0.45, 0.45),
        )
        passed = passed + ((event == 0) & crossed).astype(jnp.int32)
        event = jnp.where((event == 0) & (passed >= len(task.gate_order)), 1, event)
    return finite, event, clearance, passed
