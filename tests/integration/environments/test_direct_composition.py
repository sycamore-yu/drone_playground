"""Direct component composition uses Brax State and one physical transition."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from brax.envs.base import State


def test_environment_composes_components_without_method_or_algorithm_names():
    from drone_playground.environments.base import DroneEnvironment

    calls = []

    class Dynamics:
        forward, drone = "test", "test"

        def step(self, state, control, dt):
            calls.append((control, dt))
            return state + control * dt

    class Controller:
        low, high = jnp.array([-1.0]), jnp.array([1.0])

        def physical_action(self, action):
            return action

        def apply(self, state, physical):
            return physical

    class Task:
        name, duration, time_limit_kind = "test", 2.0, "truncation"
        observation = SimpleNamespace(size=1)

        def bind(self, env):
            assert env.physics_freq == 100
            assert env.substeps == 10
            assert env.transition.physics_dt == 0.01

        def reset(self, env, rng):
            del env, rng
            return State(jnp.zeros(1), jnp.zeros(1), jnp.float32(0), jnp.float32(0), {}, {})

        def physics(self, data):
            return data

        def with_physics(self, data, physical):
            return physical

        def probe(self, env, state):
            return None

        def finish(self, env, previous, data, action, physical, evidence):
            del env, previous, action, physical, evidence
            return State(data, data, data.sum(), jnp.float32(0), {}, {})

    components = dict(
        dynamics=Dynamics(),
        controller=Controller(),
        reference=None,
        scene=object(),
        sensor=None,
        task=Task(),
    )
    env = DroneEnvironment(**components, freq=10, physics_freq=100)
    state = env.reset(jax.random.key(1))
    result = env.step(state, jnp.ones(1))
    assert isinstance(result, State)
    for name, component in components.items():
        assert getattr(env, name) is component
    assert len(calls) == 1
    assert calls[0][1] == 0.1
    np.testing.assert_allclose(result.pipeline_state, [0.1])
