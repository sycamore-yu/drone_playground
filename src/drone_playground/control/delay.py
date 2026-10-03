"""Explicit command delay; all queue state belongs to its environment instance."""

import math

import jax
import jax.numpy as jnp
from brax.envs.base import Wrapper

from drone_playground.control.setpoints import StateSetpoint


class ActionDelay(Wrapper):
    reset_info_fields = ("command_queue",)

    def __init__(self, env, steps: int):
        super().__init__(env)
        if steps < 1:
            raise ValueError("ActionDelay requires a positive delay")
        self.delay_steps = int(steps)
        self.reset_info_fields = tuple(
            dict.fromkeys((*getattr(env, "reset_info_fields", ()), *self.reset_info_fields))
        )

    def reset(self, rng, *args, **kwargs):
        state = self.env.reset(rng, *args, **kwargs)
        queue = jnp.broadcast_to(self.env.hover_action, (self.delay_steps, self.env.action_size))
        return state.replace(
            info={
                **state.info,
                "command_queue": queue,
                "applied_action": self.env.hover_action,
            }
        )

    def step(self, state, action):
        queue = state.info["command_queue"]
        result = self.env.step(state, queue[0])
        updated = jnp.concatenate((queue[1:], action[None]), axis=0)
        return result.replace(
            info={
                **result.info,
                "command_queue": updated,
                "applied_action": queue[0],
            }
        )

    def step_physical(self, state, physical):
        action = 2 * (physical - self.env.low) / (self.env.high - self.env.low) - 1
        return self.step(state, action)


def delayed_commands(queue, action, delay_seconds, control_dt, substeps):
    """Zero-order hold on the actuator clock; keep gradients into queued actions.

    Commands are generated on control ticks. Each is delivered after the same
    episode delay; delivery is rounded upward to the next physical clock tick.
    This keeps temporal causality and never interpolates future control values.
    """
    history = jnp.concatenate((action[None], queue[:-1]), axis=0)
    offsets = jnp.arange(substeps, dtype=jnp.float32) * (control_dt / substeps)
    ages = jnp.ceil((delay_seconds - offsets) / control_dt - 1e-6).astype(jnp.int32)
    ages = jnp.clip(ages, 0, history.shape[0] - 1)
    return history[ages], history


class RandomActionDelay(Wrapper):
    """Per-episode uniform transport delay, applied on the physical substep clock.

    Requested and realized milliseconds are explicit episode state. A 500Hz
    physical clock realizes 25--50ms requests as 26--50ms (at most 2ms rounding).
    Task rewards/events are evaluated once after the complete control interval.
    """

    reset_info_fields = (
        "delay_command_queue",
        "delay_requested_ms",
        "delay_effective_ms",
    )

    def __init__(self, env, milliseconds):
        super().__init__(env)
        low, high = map(float, milliseconds)
        if not (math.isfinite(low) and math.isfinite(high) and 0 <= low <= high):
            raise ValueError("Delay range must contain finite nonnegative ordered milliseconds")
        if not hasattr(env, "step_schedule") or not hasattr(env, "substeps"):
            raise ValueError("Random delay requires a substep command execution interface")
        self.delay_range_ms = (low, high)
        self.queue_size = math.ceil(high / (1000 * env.dt)) + 1
        self.clock_ms = 1000 * env.dt / env.substeps
        self.reset_info_fields = tuple(
            dict.fromkeys(
                (
                    *getattr(env, "reset_info_fields", ()),
                    *self.reset_info_fields,
                )
            )
        )

    def reset(self, rng, *args, **kwargs):
        environment_key, delay_key = jax.random.split(rng)
        state = self.env.reset(environment_key, *args, **kwargs)
        low, high = self.delay_range_ms
        requested = jax.random.uniform(delay_key, (), minval=low, maxval=high)
        effective = jnp.ceil(requested / self.clock_ms - 1e-6) * self.clock_ms
        queue = jnp.broadcast_to(self.env.hover_action, (self.queue_size, self.env.action_size))
        info = {
            **state.info,
            "delay_command_queue": queue,
            "delay_requested_ms": requested,
            "delay_effective_ms": effective,
            "applied_action": self.env.hover_action,
            "requested_action": self.env.hover_action,
        }
        return state.replace(info=info)

    def step(self, state, action):
        commands, queue = delayed_commands(
            state.info["delay_command_queue"],
            action,
            state.info["delay_effective_ms"] * 0.001,
            self.env.dt,
            self.env.substeps,
        )
        physical = jax.vmap(self.env.physical_action)(commands)
        result = self.env.step_schedule(state, physical)
        return result.replace(
            info={
                **result.info,
                "delay_command_queue": queue,
                "applied_action": commands[-1],
                "requested_action": action,
            }
        )

    def step_physical(self, state, physical):
        action = 2 * (physical - self.env.low) / (self.env.high - self.env.low) - 1
        return self.step(state, action)


def delayed_step(model, state, command, previous, delay_ticks, physics_dt, substeps):
    """Apply the previous command until delivery, then the current command."""

    def one(current, tick):
        due = jnp.where((tick < delay_ticks)[..., None], previous, command)
        nxt = model.step(current, StateSetpoint(acceleration=due), physics_dt)
        return nxt, nxt.pos

    return jax.lax.scan(one, state, jnp.arange(substeps))
