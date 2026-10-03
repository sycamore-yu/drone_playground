"""The public configuration surface must build genuine, compatible systems."""

import importlib
import importlib.util
import unittest

import crazyflow  # noqa: F401
import jax
import numpy as np
from hydra.errors import ConfigCompositionException, InstantiationException


class CompositionTests(unittest.TestCase):
    def module(self):
        name = "drone_playground.composition"
        self.assertIsNotNone(
            importlib.util.find_spec(name), "Composed environment entry is missing"
        )
        return importlib.import_module(name)

    def test_presets_overrides_and_unknown_key(self):
        m = self.module()
        cfg = m.compose_experiment(
            "control/ppo",
            "tracking",
            ["env.dynamics.forward=so_rpy_rotor", "training.seed=7"],
        )
        self.assertEqual(cfg["env"]["dynamics"]["forward"], "so_rpy_rotor")
        self.assertEqual(cfg["training"]["seed"], 7)
        self.assertEqual(cfg["algorithm"]["name"], "ppo")
        for group in ("method", "env", "algorithm", "network", "training", "runtime"):
            self.assertIn(group, cfg)
        with self.assertRaises(ConfigCompositionException):
            m.compose_experiment("control/ppo", "tracking", ["training.misspelled=1"])

    def test_incompatible_action_and_gradient_are_rejected(self):
        m = self.module()
        from tests.helpers.configs import bodyrates_config

        cfg = bodyrates_config()
        cfg["env"]["controller"] = {
            "_target_": "drone_playground.control.controllers.crazyflow.AttitudeControl"
        }
        with self.assertRaisesRegex(ValueError, "control"):
            m.build_environment(cfg, device="cpu")
        cfg = m.compose_experiment("control/attitude_mpc")
        cfg["mode"] = "train"
        with self.assertRaisesRegex(ValueError, "training"):
            m.validate_config(cfg)

    def test_domain_randomization_never_leaks_to_unsupported_dynamics(self):
        from drone_playground.environments.environment import build_dynamics

        m = self.module()
        # PointMassLag supports only its declared motor/lag parameters; the
        # default mass/inertia randomization must fail instead of being ignored.
        for method in ("papers/differentiable_pointcloud",):
            cfg = m.compose_experiment(method)
            cfg["training"]["domain_randomization"]["enabled"] = True
            with self.assertRaisesRegex(InstantiationException, "domain_randomization"):
                build_dynamics(cfg, cfg["training"])
        # CrazyflowModel samples declared physical parameters on training resets.
        cfg = m.compose_experiment("control/ppo", "tracking")
        cfg["training"]["domain_randomization"]["enabled"] = True
        cfg["training"]["domain_randomization"]["dynamics"] = {
            "mass": [2.0, 2.0],
            "inertia": None,
            "motor_strength": [1.0, 1.0],
            "drag": None,
        }
        env = m.build_environment(cfg, device="cpu", role="train", count=1)
        self.addCleanup(env.close)
        nominal_mass = np.asarray(env.default.params.mass)
        state = env.reset(jax.random.PRNGKey(11))
        np.testing.assert_allclose(
            np.asarray(state.pipeline_state.sim_data.params.mass), 2.0 * nominal_mass, rtol=1e-6
        )
        race_cfg = m.compose_experiment("control/ppo", "racing")
        race_cfg["training"]["domain_randomization"]["enabled"] = True
        race_cfg["training"]["domain_randomization"]["dynamics"] = {
            "mass": [2.0, 2.0],
            "inertia": None,
            "motor_strength": [1.0, 1.0],
            "drag": None,
        }
        race = m.build_environment(race_cfg, device="cpu", role="train", count=1)
        self.addCleanup(race.close)
        race_state = race.reset(jax.random.PRNGKey(11))
        np.testing.assert_allclose(
            np.asarray(race_state.pipeline_state.sim_data.params.mass),
            2.0 * np.asarray(race.default.sim_data.params.mass),
            rtol=1e-6,
        )
        # Checkpoint selection and final benchmark use the frozen nominal plant.
        evaluation = m.build_environment(cfg, device="cpu", role="eval", count=1)
        self.addCleanup(evaluation.close)
        evaluated = evaluation.reset(jax.random.PRNGKey(11))
        np.testing.assert_allclose(
            np.asarray(evaluated.pipeline_state.sim_data.params.mass),
            np.asarray(evaluation.default.params.mass),
            rtol=1e-6,
        )
        # Disabled randomization leaves every recipe on its fixed physical model.
        cfg = m.compose_experiment("papers/differentiable_pointcloud")
        self.assertEqual(type(build_dynamics(cfg)).__name__, "PointMassLag")

    def test_actual_step_and_selected_objective(self):
        m = self.module()
        cfg = m.compose_experiment("control/ppo", "tracking")
        env = m.build_environment(cfg, device="cpu", role="eval", count=2)
        self.addCleanup(env.close)
        state = env.reset(jax.random.PRNGKey(3))
        out = jax.jit(env.step)(state, env.hover_action)
        self.assertEqual(out.obs.shape, (43,))
        self.assertTrue(np.isfinite(out.obs).all())
        self.assertEqual(env.component_identity["dynamics"]["forward"], "so_rpy")
        self.assertEqual(env.component_identity["controller"]["input_kind"], "attitude")
        cfg["env"]["task"]["reward"]["scale"] = 2.0
        other = m.build_environment(cfg, device="cpu", role="eval", count=2)
        self.addCleanup(other.close)
        s = other.reset(jax.random.PRNGKey(3))
        scaled = jax.jit(other.step)(s, other.hover_action)
        np.testing.assert_allclose(scaled.obs, out.obs, atol=1e-6)
        np.testing.assert_allclose(scaled.reward, 2 * out.reward, atol=1e-6)
