"""First terminal event at each physics tick, shared by rollout and evaluation."""

import jax
import jax.numpy as jnp


def _select(mask, new, old):
    return jax.tree.map(
        lambda n, o: jnp.where(mask.reshape(mask.shape + (1,) * (n.ndim - mask.ndim)), n, o),
        new,
        old,
    )


def advance_checked(task, state, command, previous, delay, timestamp, outcome, gates):
    """Integrate a delayed command and stop at the first actual 2ms terminal event."""

    def step(carry, tick):
        physical, clock, result, passed, minimum = carry
        active = result == 0
        due = jnp.where((tick < delay)[:, None], previous, command)
        proposed = task.dynamics.step(
            physical, task.controller.apply(physical, due), task.physics_dt
        )
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
                physical.pos,
                task.gate_positions[gate_id],
                task.gate_quaternions[gate_id],
                task.gate_reverse[order],
                (0.45, 0.45),
            )
            passed = passed + (active & (event == 0) & crossed).astype(jnp.int32)
            event = jnp.where((event == 0) & (passed >= len(task.gate_order)), 1, event)
        physical = _select(active & finite, proposed, physical)
        clock = jnp.where(active, clock + task.physics_dt, clock)
        result = jnp.where(active, event, result)
        minimum = jnp.where(active & finite, jnp.minimum(minimum, clearance), minimum)
        return (physical, clock, result, passed, minimum), None

    initial = (
        state,
        timestamp,
        outcome,
        gates,
        jnp.full(timestamp.shape, jnp.inf),
    )
    result, _ = jax.lax.scan(step, initial, jnp.arange(task.substeps))
    return (*result[:4], jnp.where(jnp.isfinite(result[4]), result[4], 0.0))
