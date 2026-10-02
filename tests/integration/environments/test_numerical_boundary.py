"""Numerical failure must be reported before a divergent state enters PPO statistics."""

import unittest
from dataclasses import replace

import crazyflow  # noqa: F401
import jax
import jax.numpy as jnp
import numpy as np


class NumericalBoundaryTests(unittest.TestCase):
    def make_env(self):
        from drone_playground.environments.tasks.tracking.rigid_body import TrackingEnv

        env = TrackingEnv(
            task="tracking",
            reference="random",
            dynamics="so_rpy_rotor_drag",
            device="cpu",
            reference_count=2,
            numerical_guard=True,
        )
        self.addCleanup(env.close)
        return env

    def test_divergent_but_finite_angular_rate_is_a_numerical_failure(self):
        env = self.make_env()
        state = env.reset(jax.random.PRNGKey(0))

        # Actual captured failing batch had angular rates up to 8.119e26 rad/s,
        # while position/quaternion/linear velocity remained within task bounds.
        def divergent_step(data, steps):
            del steps
            return data.replace(
                states=data.states.replace(ang_vel=jnp.full_like(data.states.ang_vel, 8.119e26))
            )

        # Execution owns the captured callable after assembly; inject the same
        # physical failure at that public transition boundary.
        env.transition = replace(env.transition, advance=divergent_step)
        out = jax.jit(env.step)(state, env.hover_action)
        self.assertTrue(np.isfinite(out.obs).all())
        self.assertTrue(bool(out.done))
        self.assertEqual(float(out.metrics["numerical_failure"]), 1.0)
        self.assertEqual(float(out.reward), -1.0)

    def test_same_step_reset_keeps_learner_observations_and_second_moments_finite(self):
        from drone_playground.learning.wrappers import wrap_for_training

        env = self.make_env()
        env.transition = replace(
            env.transition,
            advance=lambda data, steps: data.replace(
                states=data.states.replace(ang_vel=jnp.full_like(data.states.ang_vel, 8.119e26))
            ),
        )
        wrapped = wrap_for_training(env, env.episode_length)
        state = wrapped.reset(jax.random.split(jax.random.PRNGKey(0), 2))
        out = jax.jit(wrapped.step)(state, jnp.tile(env.hover_action, (2, 1)))
        self.assertTrue(np.isfinite(np.asarray(out.obs) ** 2).all())
        np.testing.assert_array_equal(out.metrics["numerical_failure"], [1.0, 1.0])
        np.testing.assert_array_equal(out.done, [1.0, 1.0])
