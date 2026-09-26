"""Delivery summaries must validate evidence, rather than trust success labels."""

import hashlib
import importlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


class ReportingTests(unittest.TestCase):
    def module(self):
        name = "drone_playground.evaluation.reporting"
        self.assertIsNotNone(importlib.util.find_spec(name), "Evidence validator must exist")
        return importlib.import_module(name)

    def fixture(self, root):
        run = root / "experiments/example"
        checkpoint = run / "checkpoints/step-0000000100.pkl"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"only digest checking: no unpickling")
        meta = {
            "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "parameter_sha256": "frozen-policy",
            "step": 100,
            "config": {
                "task": "figure8",
                "algorithm": "ppo",
                "dynamics": "so_rpy",
                "drone": "cf2x_L250",
                "num_timesteps": 100,
                "seed": 0,
            },
        }
        checkpoint.with_suffix(".json").write_text(json.dumps(meta))
        (checkpoint.parent / "best.json").write_text(
            json.dumps({"path": checkpoint.name, "selection_split": "dev"})
        )
        (run / "result.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    "actual_steps": 100,
                    "full_budget_completed": True,
                    "engineer_passed": True,
                    "elapsed_seconds": 5.0,
                }
            )
        )
        for split, base, count in (("dev", 20000, 32), ("heldout", 30000, 128)):
            folder = run / f"independent-{split}"
            folder.mkdir()
            rows = [
                {
                    "case": i,
                    "seed": base + i,
                    "completed": i % 2 == 0,
                    "failed": i % 2 == 1,
                    "rmse_m": 0.1 if i % 2 == 0 else 0.3,
                }
                for i in range(count)
            ]
            report = {
                "task": "figure8",
                "dynamics": "so_rpy",
                "drone": "cf2x_L250",
                "num_trials": count,
                "completed": count // 2,
                "failed": count // 2,
                "completion_rate": 0.5,
                "rmse_all_mean": 0.2,
                "parameters_frozen": True,
                "parameter_sha256": "frozen-policy",
                "quality_passed": False,
                "episodes": rows,
                "process_id": 100 + count,
                "checkpoint": str(checkpoint),
            }
            (folder / "report.json").write_text(json.dumps(report))
        return run

    def test_keeps_valid_low_scores_and_verifies_budget(self):
        m = self.module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            result = m.learning_result(root, "example")
            self.assertEqual(result["status"], "completed")
            self.assertFalse(result["quality_passed"])
            self.assertEqual(result["heldout"]["completed"], 64)
            self.assertEqual(result["actual_steps"], 100)

    def test_duplicate_eval_seed_is_rejected(self):
        m = self.module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = self.fixture(root)
            path = run / "independent-heldout/report.json"
            report = json.loads(path.read_text())
            report["episodes"][1]["seed"] = 30000
            path.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "seed"):
                m.learning_result(root, "example")

    def test_changed_checkpoint_and_claimed_success_are_rejected(self):
        m = self.module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = self.fixture(root)
            path = run / "independent-heldout/report.json"
            report = json.loads(path.read_text())
            report["completed"] = 128
            path.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "completed"):
                m.learning_result(root, "example")
            report["completed"] = 64
            path.write_text(json.dumps(report))
            (run / "checkpoints/step-0000000100.pkl").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "digest"):
                m.learning_result(root, "example")

    def test_missing_evaluation_stays_incomplete(self):
        m = self.module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = self.fixture(root)
            (run / "independent-heldout/report.json").unlink()
            result = m.learning_result(root, "example")
            self.assertEqual(result["status"], "awaiting-evaluation")
            self.assertFalse(result["experiment_completed"])
