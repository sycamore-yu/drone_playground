"""LOTF updates must match the author's loss, RNG sequence and optimizer."""

import importlib
import importlib.util
import tempfile
import unittest
from pathlib import Path

import crazyflow  # noqa: F401
import jax
import numpy as np


class LOTFTrainingTests(unittest.TestCase):
    def module(self):
        name = "drone_playground.learning.algorithms.lotf_bptt"
        self.assertIsNotNone(importlib.util.find_spec(name), "LOTF BPTT implementation must exist")
        return importlib.import_module(name)

    def small_config(self):
        from tests.reference_configs import compose_reference as compose_config

        cfg = compose_config(
            "lotf_hybrid_hover",
            [
                "training.num_envs=2",
                "training.policy_updates=2",
                "training.num_evals=3",
                "network.hidden_sizes=[16,16]",
                "task.duration=0.08",
            ],
        )
        cfg["runtime"]["device"] = "cpu"
        return cfg

    def test_single_update_matches_upstream_bptt(self):
        m = self.module()
        from lotf.algos import bptt

        from drone_playground.composition import build_environment

        cfg = self.small_config()
        task = build_environment(cfg, device="cpu")
        self.addCleanup(task.close)
        runner, network = m.initialize(task, cfg)
        upstream = bptt.train(
            task.training_env,
            runner.env_state,
            runner.last_obs,
            runner.train_state,
            num_epochs=1,
            num_steps_per_epoch=task.episode_length,
            num_envs=2,
            res_model_params=None,
            key=runner.key,
        )
        update = m.make_update(task, cfg)
        local, metrics = update(runner)
        for a, b in zip(
            jax.tree.leaves(local.train_state.params),
            jax.tree.leaves(upstream["runner_state"].train_state.params),
        ):
            np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-7)
        np.testing.assert_allclose(metrics["loss"], upstream["metrics"][0], rtol=1e-6)
        np.testing.assert_array_equal(
            jax.random.key_data(local.key), jax.random.key_data(upstream["runner_state"].key)
        )

    def test_full_state_resume_exact(self):
        m = self.module()
        from drone_playground.composition import build_environment

        cfg = self.small_config()
        task = build_environment(cfg, device="cpu")
        self.addCleanup(task.close)
        initial, _ = m.initialize(task, cfg)
        update = m.make_update(task, cfg)
        first, _ = update(initial)
        continuous, _ = update(first)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "state.pkl"
            m.save_training_state(path, first, cfg)
            resumed = m.load_training_state(path, initial, cfg)
            restored, _ = update(resumed)
        for a, b in zip(
            jax.tree.leaves(continuous.train_state.params),
            jax.tree.leaves(restored.train_state.params),
        ):
            np.testing.assert_array_equal(a, b)

    def test_resume_uses_behavioral_config_and_rejects_model_or_policy_change(self):
        import copy

        m = self.module()
        from drone_playground.composition import build_environment

        cfg = self.small_config()
        task = build_environment(cfg, device="cpu")
        self.addCleanup(task.close)
        initial, _ = m.initialize(task, cfg)
        archived = copy.deepcopy(cfg)
        archived["algorithm"].pop("rollout")
        archived["algorithm"]["horizon_length"] = cfg["env"]["task"]["duration"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.pkl"
            m.save_training_state(path, initial, archived)
            restored = m.load_training_state(path, initial, cfg)
            self.assertEqual(int(restored.epoch_idx), 0)
            for group, key, value in [
                ("algorithm", "learning_rate", 0.02),
                ("method", "output", "attitude_thrust"),
                ("scene", "randomization", "different"),
            ]:
                changed = copy.deepcopy(cfg)
                (changed["env"]["scene"] if group == "scene" else changed[group])[key] = value
                with self.assertRaisesRegex(ValueError, "configuration"):
                    m.load_training_state(path, initial, changed)

    def test_complete_small_run_has_independent_evaluation_and_native_replay(self):
        from drone_playground.runs.layout import find_experiment

        m = self.module()
        cfg = self.small_config()
        cfg["evaluation"]["episodes"] = 2
        with tempfile.TemporaryDirectory() as folder:
            result = m.train(cfg, Path(folder), "lotf-small-test")
            self.assertEqual(result["actual_steps"], 16)
            self.assertGreater(result["actor_parameter_delta_l2"], 0.0)
            root = find_experiment(folder, 'lotf-small-test')
            self.assertTrue((root / "checkpoints/best.json").is_file())
            self.assertGreater(len(list((root / "rollouts").rglob("*.mj_unroll"))), 1)
