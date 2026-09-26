"""Tests invoke actual optimizers; source identity and physical outputs matter."""

import importlib
import importlib.util
import tempfile
import unittest
from pathlib import Path

import crazyflow  # noqa: F401
import numpy as np


class OptimizationTests(unittest.TestCase):
    def module(self, name):
        name = "drone_playground.controllers." + name
        self.assertIsNotNone(importlib.util.find_spec(name), "Optimizer adapter must exist")
        return importlib.import_module(name)

    def test_actual_acados_solve_preserves_native_problem(self):
        m = self.module("lsy_mpc")
        from drone_playground.tasks.racing import load_config

        obs = dict(
            pos=np.array([-1.5, 0.75, 0.01]),
            quat=np.array([0.0, 0.0, 0.0, 1.0]),
            vel=np.zeros(3),
            ang_vel=np.zeros(3),
        )
        with tempfile.TemporaryDirectory() as directory:
            controller = m.LSYAttitudeMPC(obs, {}, load_config(), workdir=Path(directory))
            action = controller.compute_control(dict(obs), {})
            self.assertEqual(action.shape, (4,))
            self.assertTrue(np.isfinite(action).all())
            self.assertGreater(action[3], 0)
            self.assertEqual(controller.last_diagnostics["status"], 0)
            self.assertEqual(controller.native._N, 25)
            np.testing.assert_array_equal(np.diag(controller.native._ocp.cost.W)[:3], [50, 50, 400])

    def test_actual_sampling_decision_and_warm_start(self):
        m = self.module("sampling")
        import jax

        from drone_playground.tasks.racing import race_reference
        from drone_playground.tasks.tracking import TrackingEnv

        env = TrackingEnv(
            task="random", dynamics="first_principles", device="cpu", reference_count=2
        )
        self.addCleanup(env.close)
        state = env.reset(jax.random.PRNGKey(3))
        ref, _ = race_reference(np.array([-1.5, 1, 0.07]))
        controller = m.SamplingMPC(
            drone="cf21B_500", reference=ref, frequency=50, samples=32, horizon=8, device="cpu"
        )
        self.addCleanup(controller.close)
        action = controller.compute_from_data(state.pipeline_state.sim_data, 0)
        self.assertEqual(action.shape, (4,))
        self.assertTrue(np.isfinite(action).all())
        self.assertGreater(controller.last_diagnostics["samples"], 0)
        second = controller.compute_from_data(state.pipeline_state.sim_data, 1)
        self.assertTrue(np.isfinite(second).all())
