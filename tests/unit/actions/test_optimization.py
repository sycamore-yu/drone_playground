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
        name = "drone_playground.actions.controllers.mpc." + name
        self.assertIsNotNone(importlib.util.find_spec(name), "Optimizer adapter must exist")
        return importlib.import_module(name)

    def test_actual_acados_solve_preserves_native_problem(self):
        m = self.module("lsy_mpc")
        from drone_playground.environments.scenes.racing import load_lsy_config

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
            from drone_playground.actions.commands import Trajectory

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
            controller.episode_callback()
            original = controller.native._waypoints_pos.copy()
            controller.enable_delay_compensation(38., [0., 0., 0., .3])
            np.testing.assert_allclose(controller.native._waypoints_pos[0, 0],
                                       original[0, 0] + .5 * .038)
            action = controller.compute_control(obs)
            self.assertTrue(np.isfinite(action).all())
            self.assertEqual(controller.last_diagnostics['status'], 0)
            self.assertEqual(len(controller.delay_predictor.history), 1)
            controller.episode_callback()
            self.assertEqual(len(controller.delay_predictor.history), 0)
            with self.assertRaisesRegex(ValueError, 'fixed task reference'):
                controller.compute_trajectory(obs, curve, 2)

def test_mpc_tracking_horizon_covers_the_final_tick_without_shortening_the_task():
    from drone_playground.actions.controllers.mpc.factory import tracking_reference

    phase = np.arange(500) / 500 * 2 * np.pi
    points = np.stack([np.sin(phase), np.cos(phase), np.ones(500)], axis=-1)
    extended, velocity = tracking_reference(points, 50, 25, periodic=True)
    np.testing.assert_array_equal(extended[:500], points)
    np.testing.assert_array_equal(extended[500:], points[:25])
    assert len(extended[499:499 + 26]) == 26
    assert np.isfinite(velocity).all()
    held, _ = tracking_reference(points, 50, 25, periodic=False)
    np.testing.assert_array_equal(held[500:], np.broadcast_to(points[-1], (25, 3)))
