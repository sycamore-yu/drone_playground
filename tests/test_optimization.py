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
        name = "drone_playground.methods.optimal_control." + name
        self.assertIsNotNone(importlib.util.find_spec(name), "Optimizer adapter must exist")
        return importlib.import_module(name)

    def test_actual_acados_solve_preserves_native_problem(self):
        m = self.module("lsy_mpc")
        from drone_playground.environments.scenes import load_lsy_config

        obs = dict(
            pos=np.array([-1.5, 0.75, 0.01]),
            quat=np.array([0.0, 0.0, 0.0, 1.0]),
            vel=np.zeros(3),
            ang_vel=np.zeros(3),
        )
        with tempfile.TemporaryDirectory() as directory:
            controller = m.LSYAttitudeMPC(obs, {}, load_lsy_config(), workdir=Path(directory))
            action = controller.compute_control(dict(obs), {})
            self.assertEqual(action.shape, (4,))
            self.assertTrue(np.isfinite(action).all())
            self.assertGreater(action[3], 0)
            self.assertEqual(controller.last_diagnostics["status"], 0)
            self.assertEqual(controller.native._N, 25)
            np.testing.assert_array_equal(np.diag(controller.native._ocp.cost.W)[:3], [50, 50, 400])
            from drone_playground.native.contracts import Trajectory

            coefficients = np.zeros((1, 4, 2))
            coefficients[0, :3, 0] = obs["pos"]
            coefficients[0, 0, 1] = 0.5
            curve = Trajectory(2, [2], coefficients)
            action = controller.compute_trajectory(obs, curve, 2)
            self.assertTrue(np.isfinite(action).all())
            self.assertEqual(controller.last_diagnostics["status"], 0)
            # Every acados stage sees a moving future point, including terminal.
            np.testing.assert_allclose(
                controller.native._waypoints_pos[:, 0],
                obs["pos"][0] + 0.5 * np.linspace(0, 0.5, 26),
            )
            np.testing.assert_allclose(controller.native._waypoints_vel[:, 0], 0.5)

    def test_actual_sampling_decision_and_warm_start(self):
        m = self.module("sampling")
        import jax

        from drone_playground.environments.tasks.tracking import TrackingEnv
        from drone_playground.methods.planners.reference import race_reference

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
        from drone_playground.native.contracts import Trajectory

        coefficients = np.zeros((1, 4, 2))
        coefficients[0, :3, 0] = env.controller_observation(state)["pos"]
        coefficients[0, 0, 1] = 1.0
        coefficients[0, 3, 0] = 0.3
        command = controller.compute_control(
            env.controller_observation(state), 0, trajectory=Trajectory(0, [2], coefficients)
        )
        self.assertTrue(np.isfinite(command).all())
        self.assertAlmostEqual(command[2], 0.3, places=5)
