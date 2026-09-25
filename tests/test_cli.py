"""Exercise the public command entry and the native-control demonstration."""

import tempfile
import unittest
from pathlib import Path


class CommandTests(unittest.TestCase):
    def test_parser_exposes_all_committed_entrypoints(self):
        from drone_playground.cli import build_parser

        parser = build_parser()
        for command in ("demo", "train", "evaluate", "replay", "metrics", "status"):
            self.assertIn(command, parser.format_help())

    def test_real_control_demo_records_measured_states(self):
        from drone_playground.controllers.demo import run_demo

        with tempfile.TemporaryDirectory() as tmp:
            report = run_demo(Path(tmp), "native-flight", duration=0.2)
            self.assertEqual(report["frames"], 10)
            self.assertTrue(report["finite_states"])
            self.assertTrue((Path(tmp) / "experiments/native-flight/result.json").is_file())
            self.assertEqual(
                len(list((Path(tmp) / "experiments/native-flight/rollouts").rglob("*.mj_unroll"))),
                1,
            )


if __name__ == "__main__":
    unittest.main()
