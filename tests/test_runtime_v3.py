"""Execution order and delay tests use observable state rather than private calls."""

import jax
import jax.numpy as jnp
import numpy as np
from brax.envs.base import State

from drone_playground.execution.delay import ActionDelay
from drone_playground.execution.transition import ExecutionTransition
from drone_playground.runtime.host_runner import run_steps


def test_complete_interval_applies_control_once_and_probes_every_physics_step():
    def apply(state, command):
        return state + command

    def advance(state, steps):
        return state + steps

    execution = ExecutionTransition(apply, advance, 4)
    assert int(execution.step(jnp.int32(0), jnp.int32(10))) == 14
    final, clearance, collided = execution.step_with_evidence(
        jnp.int32(0), jnp.int32(10), lambda state, index: (jnp.float32(12 - state), state == 12)
    )
    assert int(final) == 14 and float(clearance) == -2 and bool(collided)
    assert (
        float(jax.grad(lambda u: ExecutionTransition(apply, advance, 4).step(0.0, u))(2.0)) == 1.0
    )


class Toy:
    action_size = 1
    hover_action = jnp.array([0.0])
    low, high = jnp.array([-1.0]), jnp.array([1.0])

    def reset(self, key):
        del key
        return State(
            pipeline_state=None,
            obs=jnp.zeros(1),
            reward=jnp.float32(0),
            done=jnp.float32(0),
            metrics={},
            info={},
        )

    def step(self, state, action):
        return state.replace(obs=state.obs + action)


def test_action_delay_is_causal_and_reset_clears_queue():
    env = ActionDelay(Toy(), 2)
    state = env.reset(jax.random.PRNGKey(0))
    for command, expected in ((1.0, 0.0), (2.0, 0.0), (3.0, 1.0), (4.0, 3.0)):
        state = jax.jit(env.step)(state, jnp.array([command]))
        np.testing.assert_array_equal(state.obs, [expected])
    reset = env.reset(jax.random.PRNGKey(1))
    np.testing.assert_array_equal(reset.info["command_queue"], np.zeros((2, 1)))


def test_host_runner_keeps_the_terminal_physical_transition():
    toy = Toy()

    def advance(state, command):
        result = toy.step(state, command)
        return result.replace(done=(result.obs[0] >= 2).astype(jnp.float32))

    rows = list(run_steps(toy.reset(None), 5, lambda s, i: (jnp.ones(1), {"tick": i}), advance))
    assert len(rows) == 2
    assert float(rows[-1][1].after.obs[0]) == 2.0
    assert rows[-1][1].diagnostics["tick"] == 1


def test_training_autoreset_clears_delay_memory_per_finished_world():
    from drone_playground.learning.env_adapter import wrap_for_training

    class EpisodicToy(Toy):
        def step(self, state, action):
            result = super().step(state, action)
            return result.replace(done=(result.obs[0] >= 1).astype(jnp.float32))

    env = wrap_for_training(ActionDelay(EpisodicToy(), 1), 10)
    state = env.reset(jax.random.split(jax.random.PRNGKey(4), 2))
    state = env.step(state, jnp.array([[1.0], [0.0]]))
    state = env.step(state, jnp.array([[8.0], [0.0]]))
    np.testing.assert_array_equal(state.done, [1.0, 0.0])
    np.testing.assert_array_equal(state.info["command_queue"], np.zeros((2, 1, 1)))
    state = env.step(state, jnp.zeros((2, 1)))
    np.testing.assert_array_equal(state.obs, np.zeros((2, 1)))
