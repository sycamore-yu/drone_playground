"""Numerical failure must be reported before a divergent state enters PPO statistics."""

import unittest

import crazyflow  # noqa: F401
import jax
import jax.numpy as jnp
import numpy as np


class NumericalBoundaryTests(unittest.TestCase):
    def make_env(self):
        from drone_playground.tasks.tracking import TrackingEnv

        env = TrackingEnv(
            task="random",
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

        # Failure injection now crosses the injected dynamics interface.
        env.model.advance = divergent_step
        out = jax.jit(env.step)(state, env.hover_action)
        self.assertTrue(np.isfinite(out.obs).all())
        self.assertTrue(bool(out.done))
        self.assertEqual(float(out.metrics["numerical_failure"]), 1.0)
        self.assertEqual(float(out.reward), -1.0)

    def test_same_step_reset_keeps_learner_observations_and_second_moments_finite(self):
        from drone_playground.tasks.tracking import wrap_for_training

        env = self.make_env()
        env.model.advance = lambda data, steps: data.replace(
            states=data.states.replace(ang_vel=jnp.full_like(data.states.ang_vel, 8.119e26))
        )
        wrapped = wrap_for_training(env, env.episode_length)
        state = wrapped.reset(jax.random.split(jax.random.PRNGKey(0), 2))
        out = jax.jit(wrapped.step)(state, jnp.tile(env.hover_action, (2, 1)))
        self.assertTrue(np.isfinite(np.asarray(out.obs) ** 2).all())
        np.testing.assert_array_equal(out.metrics["numerical_failure"], [1.0, 1.0])
        np.testing.assert_array_equal(out.done, [1.0, 1.0])
