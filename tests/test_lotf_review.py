"""Regression checks for durable LOTF experiments and truthful configuration."""

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import crazyflow  # noqa: F401
import jax.numpy as jnp
import numpy as np

from drone_playground.composition import build_environment, compose_config, validate_config


class LOTFReviewTests(unittest.TestCase):
    def small_config(self):
        return compose_config(
            "lotf_hybrid_hover",
            [
                "training.device=cpu",
                "training.num_envs=2",
                "training.policy_updates=2",
                "training.num_evals=3",
                "network.hidden_sizes=[16,16]",
                "task.duration=0.08",
            ],
        )

    def test_native_tracking_rejects_unresampled_frequency(self):
        for frequency in (25, 100):
            config = compose_config("lotf_hybrid_tracking", [f"task.freq={frequency}"])
            with self.subTest(frequency=frequency):
                with self.assertRaisesRegex(ValueError, "50|reference|trajectory"):
                    validate_config(config)

    def test_declared_timesteps_must_match_native_update_budget(self):
        config = self.small_config()
        config["training"]["num_timesteps"] = 8
        with self.assertRaisesRegex(ValueError, "budget|num_timesteps|16"):
            validate_config(config)
        config["training"]["num_timesteps"] = 16
        validate_config(config)

    def test_physical_nonfinite_failure_remains_serializable(self):
        from drone_playground.evaluation.lotf import LOTFEvaluator
        from drone_playground.evaluation.tracking import save_report

        for field in ("p", "v"):
            task = build_environment(self.small_config(), "cpu")
            self.addCleanup(task.close)
            raw_step = task.raw_step

            def corrupt(state, action, key, raw_step=raw_step, field=field):
                transition = raw_step(state, action, key)
                broken = transition.state.quadrotor_state.replace(
                    **{field: jnp.full((3,), jnp.nan)}
                )
                return transition._replace(
                    state=transition.state.replace(quadrotor_state=broken),
                    obs=jnp.full_like(transition.obs, jnp.nan) if field == "p" else transition.obs,
                    reward=jnp.float32(jnp.nan) if field == "p" else transition.reward,
                )

            task.raw_step = corrupt

            def maker(params, deterministic=True):
                return lambda obs, key: (
                    jnp.broadcast_to(task.hover_action, (*obs.shape[:-1], 4)),
                    {},
                )

            report, trace = LOTFEvaluator(task, maker, [30000, 30001]).run(())
            self.assertEqual(report["num_trials"], 2)
            self.assertEqual(report["completed"], 0)
            self.assertEqual(report["failed"], 2)
            self.assertEqual(report.get("numerical_failures"), 2)
            self.assertFalse(report["quality_passed"])
            self.assertTrue(np.isfinite(trace["pos"]).all())
            self.assertTrue(np.isfinite(trace["quat"]).all())
            self.assertTrue(np.isfinite(trace["obs"]).all())
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "report.json"
                save_report(path, report)
                self.assertEqual(json.loads(path.read_text())["failed"], 2)

    def test_resume_inherits_best_before_snapshot_and_excludes_future(self):
        from drone_playground.learning import lotf_bptt as module

        self.assertTrue(hasattr(module, "inherit_development_best"))
        config = self.small_config()
        task = build_environment(config, "cpu")
        self.addCleanup(task.close)
        runner, _ = module.initialize(task, config)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, destination = root / "source", root / "destination"
            # Build actual policy/state files and explicit historical score fixtures.
            reports = []
            for epoch, error in ((0, 0.1), (1, 0.2), (2, 0.01)):
                path = source / "checkpoints" / f"step-{epoch * 8:010d}.pkl"
                module.save_policy(path, (None, runner.train_state.params), task, config, epoch * 8)
                report = dict(
                    updates=epoch,
                    step=epoch * 8,
                    completed=32,
                    rmse_all_mean=error,
                    last_second_rmse_mean_m=error,
                    quality_passed=True,
                    checkpoint=str(path.relative_to(source)),
                    split="dev",
                )
                report_path = source / "eval" / f"step-{epoch * 8:010d}.json"
                report_path.parent.mkdir(parents=True, exist_ok=True)
                report_path.write_text(json.dumps(report))
                reports.append(report)
            state_file = source / "training-state" / "update-000001.pkl"
            module.save_training_state(state_file, runner._replace(epoch_idx=1), config)
            original = {
                p.relative_to(source): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in source.rglob("*")
                if p.is_file()
            }
            score, best = module.inherit_development_best(state_file, destination, task.task)
            self.assertEqual(score, (32, -0.1))
            self.assertEqual(best["step"], 0)
            # A fresh evaluation at the resumed step can replace a normal report;
            # continuation provenance must have its own durable record.
            lineage = json.loads((destination / "resume-selection.json").read_text())
            self.assertEqual(lineage["resumed_update"], 1)
            self.assertEqual(lineage["selected_step"], 0)
            self.assertEqual(lineage["eligible_steps"], [0, 8])
            self.assertEqual(
                json.loads((destination / "checkpoints/best.json").read_text())["step"], 0
            )
            copied = destination / "checkpoints/step-0000000000.pkl"
            self.assertEqual(
                copied.read_bytes(), (source / "checkpoints" / copied.name).read_bytes()
            )
            self.assertEqual(
                original,
                {
                    p.relative_to(source): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in source.rglob("*")
                    if p.is_file()
                },
            )

    def test_evaluation_ignores_archived_training_only_budget(self):
        from drone_playground.cli import main
        from drone_playground.learning import lotf_bptt as module

        config = self.small_config()
        task = build_environment(config, "cpu")
        self.addCleanup(task.close)
        runner, _ = module.initialize(task, config)
        archived = copy.deepcopy(config)
        archived["training"]["num_timesteps"] = 2097152
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            checkpoint = folder / "initial.pkl"
            module.save_policy(checkpoint, (None, runner.train_state.params), task, archived, 0)
            output = folder / "evaluation"
            main(
                [
                    "evaluate",
                    "--checkpoint",
                    str(checkpoint),
                    "--device",
                    "cpu",
                    "--episodes",
                    "1",
                    "--output",
                    str(output),
                ]
            )
            report = json.loads((output / "report.json").read_text())
            self.assertTrue(report["parameters_frozen"])
            self.assertEqual(report["num_trials"], 1)


if __name__ == "__main__":
    unittest.main()
