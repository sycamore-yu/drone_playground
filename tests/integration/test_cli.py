"""Exercise the public command entry and the native-control demonstration."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tests.helpers.paths import REPO_ROOT


class CommandTests(unittest.TestCase):
    def test_all_scripts_accept_hydra_configuration_display(self):
        root = REPO_ROOT
        for mode in ("train", "eval", "play"):
            with self.subTest(mode=mode):
                output = subprocess.run(
                    [
                        sys.executable,
                        str(root / "scripts" / f"{mode}.py"),
                        "experiment=control/ppo",
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
        from drone_playground.artifacts.layout import find_experiment
        from drone_playground.runtime.demo import run_demo

        with tempfile.TemporaryDirectory() as tmp:
            report = run_demo(Path(tmp), "native-flight", duration=0.2)
            self.assertEqual(report["frames"], 10)
            self.assertTrue(report["finite_states"])
            run = find_experiment(Path(tmp), "native-flight")
            self.assertTrue((run / "result.json").is_file())
            self.assertEqual(
                len(list((run / "rollouts").rglob("*.mj_unroll"))),
                1,
            )


def test_train_eval_and_play_respect_replay_storage(tmp_path):
    """Keep ordinary runs compact while making frozen-policy play viewable."""
    from drone_playground.app import run_experiment
    from drone_playground.artifacts.layout import find_experiment
    from drone_playground.configuration import compose_experiment
    from drone_playground.visualization.viewer import replay

    config = compose_experiment(
        "control/bptt",
        "hovering",
        [
            "runtime.device=cpu",
            "runtime.action_delay_ms=null",
            "env.task.duration=0.04",
            "network.hidden_sizes=[8]",
            "training.num_envs=2",
            "training.policy_updates=1",
            "algorithm.horizon_length=2",
            "training.num_evals=2",
            "training.checkpoint_eval_episodes=1",
            "evaluation.episodes=1",
            "visualization.publish=false",
        ],
    )
    trained = run_experiment(config, tmp_path, "tiny-train")
    assert trained["actual_steps"] == 4
    assert trained["actor_parameter_delta_l2"] > 0
    train_run = find_experiment(tmp_path, "tiny-train")
    config["checkpoint"] = str(train_run / "checkpoints/step-0000000004.pkl")
    config["mode"] = "eval"
    evaluated = run_experiment(config, tmp_path, "tiny-eval")
    assert evaluated["parameters_frozen"]
    config["mode"] = "play"
    played = run_experiment(config, tmp_path, "tiny-play")
    runs = [find_experiment(tmp_path, name) for name in ("tiny-train", "tiny-eval", "tiny-play")]
    assert [any((run / "rollouts").rglob("*.mj_unroll")) for run in runs] == [False, False, True]
    assert all((run / "result.json").is_file() for run in runs)
    assert replay(Path(played["replay_directory"]), publish=False)["available_cases"]
    assert config["evaluation"]["record_replays"] is False


if __name__ == "__main__":
    unittest.main()
