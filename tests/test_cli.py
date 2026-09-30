"""Exercise the public command entry and the native-control demonstration."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class CommandTests(unittest.TestCase):
    def test_all_scripts_accept_hydra_configuration_display(self):
        root = Path(__file__).resolve().parents[1]
        for mode in ("train", "eval", "play"):
            with self.subTest(mode=mode):
                output = subprocess.run(
                    [
                        sys.executable,
                        str(root / "scripts" / f"{mode}.py"),
                        "method=learning/ppo",
                        "env=racing",
                        "--cfg",
                        "job",
                    ],
                    cwd=root,
                    text=True,
                    capture_output=True,
                    timeout=20,
                )
                self.assertEqual(output.returncode, 0, output.stderr)
                self.assertIn(f"mode: {mode}", output.stdout)

    def test_parser_exposes_all_committed_entrypoints(self):
        from drone_playground.cli import build_parser

        parser = build_parser()
        for command in ("demo", "train", "eval", "play", "replay", "metrics", "status"):
            self.assertIn(command, parser.format_help())

    def test_real_control_demo_records_measured_states(self):
        from drone_playground.execution.controllers.demo import run_demo
        from drone_playground.runs.layout import find_experiment

        with tempfile.TemporaryDirectory() as tmp:
            report = run_demo(Path(tmp), "native-flight", duration=0.2)
            self.assertEqual(report["frames"], 10)
            self.assertTrue(report["finite_states"])
            run = find_experiment(Path(tmp), 'native-flight')
            self.assertTrue((run / 'result.json').is_file())
            self.assertEqual(
                len(list((run / 'rollouts').rglob('*.mj_unroll'))),
                1,
            )


if __name__ == "__main__":
    unittest.main()
