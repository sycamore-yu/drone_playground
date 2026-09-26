"""The public configuration surface must build genuine, compatible systems."""

import importlib
import importlib.util
import unittest

import crazyflow  # noqa: F401
import jax
import numpy as np


class CompositionTests(unittest.TestCase):
    def module(self):
        name = "drone_playground.composition"
        self.assertIsNotNone(
            importlib.util.find_spec(name), "Composed environment entry is missing"
        )
        return importlib.import_module(name)

    def test_presets_overrides_and_unknown_key(self):
        m = self.module()
        cfg = m.compose_config("figure8_ppo", ["dynamics.forward=so_rpy_rotor", "training.seed=7"])
        self.assertEqual(cfg["dynamics"]["forward"], "so_rpy_rotor")
        self.assertEqual(cfg["training"]["seed"], 7)
        self.assertEqual(cfg["algorithm"]["name"], "ppo")
        for group in (
            "policy",
            "controller",
            "dynamics",
            "scene",
            "observation",
            "task",
            "algorithm",
            "network",
            "objective",
            "training",
        ):
            self.assertIn(group, cfg)
        with self.assertRaises(Exception):
            m.compose_config("figure8_ppo", ["training.misspelled=1"])

    def test_incompatible_action_and_gradient_are_rejected(self):
        m = self.module()
        cfg = m.compose_config("lotf_hybrid_hover")
        cfg["controller"]["name"] = "crazyflow_attitude"
        with self.assertRaisesRegex(ValueError, "controller|控制|接口"):
            m.validate_config(cfg)
        cfg = m.compose_config("racing_attitude_mpc")
        cfg["mode"] = "train"
        cfg["algorithm"]["name"] = "lotf_bptt"
        with self.assertRaisesRegex(ValueError, "gradient|BPTT|梯度|外部"):
            m.validate_config(cfg)

    def test_actual_step_and_selected_objective(self):
        m = self.module()
        cfg = m.compose_config("figure8_ppo")
        env = m.build_environment(cfg, device="cpu", split="dev", count=2)
        self.addCleanup(env.close)
        state = env.reset(jax.random.PRNGKey(3))
        out = jax.jit(env.step)(state, env.hover_action)
        self.assertEqual(out.obs.shape, (43,))
        self.assertTrue(np.isfinite(out.obs).all())
        self.assertEqual(env.component_identity["dynamics"]["forward"], "so_rpy")
        self.assertEqual(env.component_identity["controller"]["name"], "crazyflow_attitude")
        cfg["objective"]["scale"] = 2.0
        other = m.build_environment(cfg, device="cpu", split="dev", count=2)
        self.addCleanup(other.close)
        s = other.reset(jax.random.PRNGKey(3))
        scaled = jax.jit(other.step)(s, other.hover_action)
        np.testing.assert_allclose(scaled.obs, out.obs, atol=1e-6)
        np.testing.assert_allclose(scaled.reward, 2 * out.reward, atol=1e-6)
