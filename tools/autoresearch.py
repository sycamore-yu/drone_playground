"""Run Autoresearch Gym candidates through the existing Drone Playground CLI."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "autoresearch-gym"
REVISION = "7fd6391aecc3bf0ab8e8813cf4da7a053baa001f"
TASKS = ROOT / "tools" / "autoresearch"


class DronePlaygroundBackend:
    """Translate one external candidate into native training and frozen evaluation."""

    def build_bundle(self, bundle):
        """Keep physical conditions and evaluation outside candidate overrides."""
        if bundle.target_name != "local":
            raise ValueError("Drone Playground uses the local Pixi environment")
        if bundle.train_seconds is not None or bundle.train_episodes != 0:
            raise ValueError("Use execution_backend.train_env_steps; train_episodes must be 0")
        steps = bundle.execution_backend["train_env_steps"]
        if type(steps) is not int or steps < 1:
            raise ValueError("train_env_steps must be a positive integer")
        settings = bundle.benchmark.env_kwargs
        scenes = settings["eval_scenes"]
        if not scenes or len(set(scenes)) != len(scenes):
            raise ValueError("eval_scenes must contain unique scenes")
        if bundle.eval_episodes < len(scenes) or bundle.eval_episodes % len(scenes):
            raise ValueError("eval_episodes must be a positive multiple of the scene count")
        recipe = bundle.candidate_metadata["recipe"]
        for key in recipe:
            if not (
                key.startswith(("learning.options.", "method.actor."))
                or key == "simulation.num_envs"
            ):
                raise ValueError(f"Forbidden candidate override: {key}")
        request = {
            **settings,
            "train_env_steps": steps,
            "eval_episodes": bundle.eval_episodes,
            "max_steps": bundle.max_steps,
            "train_seed": bundle.benchmark.train_seed,
            "eval_seed_start": bundle.benchmark.eval_seed_start,
            "recipe": recipe,
        }
        bundle.external_dir.mkdir(parents=True, exist_ok=True)
        (bundle.external_dir / "request.json").write_text(json.dumps(request, indent=2) + "\n")
        return bundle

    def overrides(self, bundle, phase, checkpoint=None):
        """Build Hydra arguments; the candidate cannot override fixed run fields."""
        request = json.loads((bundle.external_dir / "request.json").read_text())
        output = (bundle.external_dir / phase).resolve()
        args = [
            f"experiment={request['experiment']}",
            f"learning={request['algorithm']}",
            *[f"++{key}={json.dumps(value)}" for key, value in request["recipe"].items()],
            "results.format=orbax",
            "initial_checkpoint=null",
            "resume=null",
            f"output={output}",
        ]
        if phase == "train":
            return [
                *args,
                "mode=train",
                "checkpoint=null",
                f"seed={request['train_seed']}",
                f"learning.max_interactions={request['train_env_steps']}",
                "++learning.budget_only=true",
            ]
        return [
            *args,
            f"mode={request['eval_mode']}",
            # Native evaluation adds 10000*seed. Keep the cohort fixed across training seeds.
            "seed=0",
            f"checkpoint={checkpoint}",
            "benchmark.protocol=diagnostic",
            f"benchmark.scenes={json.dumps(request['eval_scenes'])}",
            f"benchmark.episodes={request['eval_episodes'] // len(request['eval_scenes'])}",
            f"benchmark.seed_base={request['eval_seed_start']}",
            "benchmark.record_replay=false",
            "benchmark.record_trajectories=false",
        ]

    def _command(self, bundle, phase, checkpoint=None):
        from autoresearch_gym.external.base import CommandSpec

        return CommandSpec(
            argv=[
                str(ROOT / ".pixi/envs/default/bin/python"),
                "-m",
                "drone_playground.cli",
                *self.overrides(bundle, phase, checkpoint),
            ],
            cwd=ROOT,
            env={
                "PYTHONPATH": str(ROOT / "src"),
                "SCIPY_ARRAY_API": "1",
                "JAX_PLATFORMS": "cuda",
                "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                "MUJOCO_GL": "egl",
                "JAX_COMPILATION_CACHE_DIR": str(ROOT / ".cache/jax"),
            },
            label=f"drone-{phase}",
        )

    def training_command(self, bundle):
        """Run the existing trainer with the benchmark interaction budget."""
        return self._command(bundle, "train")

    def eval_command(self, bundle, checkpoint_path):
        """Load the saved Orbax actor in the native frozen evaluator."""
        return self._command(bundle, "eval", checkpoint_path)

    def media_command(self, bundle, checkpoint_path):
        """Leave replay generation to the native optional replay command."""
        return None

    def normalize_media(self, artifacts):
        """Report that this headless adapter does not generate video."""
        return {"media_available": False}

    def normalize_train(self, artifacts):
        """Read native counters and checkpoint paths without copying model data."""
        request = json.loads((artifacts.root / "request.json").read_text())
        directory = artifacts.root / "train"
        result = json.loads((directory / "run.json").read_text())
        if (
            result["status"] != "budget_exhausted"
            or result["interactions"] != request["train_env_steps"]
        ):
            raise ValueError(f"Incomplete training budget: {result['status']}")
        return {
            "total_steps": result["interactions"],
            "env_steps": result["interactions"],
            "policy_updates": result["updates"],
            "last_metrics": result["last_metrics"],
            "elapsed_seconds": result["training_wall_seconds"],
            "checkpoint_path": str((directory / result["selected_checkpoint"]).resolve()),
            "checkpoint_selection": result["checkpoint_selection"],
            "budget_mode": "env_steps",
            "train_env_steps": request["train_env_steps"],
            "episode_records": [],
            "episodes_completed": None,
            "completed_episodes": None,
            "episode_batches": 0,
            "avg_return": None,
            "avg_length": None,
            "success_rate": None,
            "curve_status": "unsupported",
            "curve_status_reason": "Native training curves remain in TensorBoard.",
            "native_run": str(directory.resolve()),
        }

    def normalize_eval(self, artifacts):
        """Score every requested scene and retain every failed episode."""
        request = json.loads((artifacts.root / "request.json").read_text())
        directory = artifacts.root / "eval"
        report = json.loads((directory / "report.json").read_text())
        if not report["complete"]:
            raise ValueError("Incomplete frozen evaluation")
        with (directory / "episodes.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        scenes = request["eval_scenes"]
        count = request["eval_episodes"] // len(scenes)
        if Counter(row["scene"] for row in rows) != dict.fromkeys(scenes, count):
            raise ValueError("Incomplete evaluation scene coverage")
        rates = {
            scene: sum(row["event"] == "SUCCESS" for row in rows if row["scene"] == scene) / count
            for scene in scenes
        }
        return {
            "success_rate": sum(rates.values()) / len(rates),
            "worst_scene_success_rate": min(rates.values()),
            "collision_rate": sum(row["event"] == "COLLISION" for row in rows) / len(rows),
            "scene_success_rates": rates,
            "avg_flight_seconds": sum(float(row["flight_seconds"]) for row in rows) / len(rows),
            "avg_return": None,
            "episodes": len(rows),
            "episode_records": [dict(row, success=row["event"] == "SUCCESS") for row in rows],
            "native_report": str((directory / "report.json").resolve()),
            "acceptance_passed": report["passed"],
        }


def setup():
    """Clone the pinned upstream and apply the tracked, small interoperability patch."""
    if not UPSTREAM.exists():
        subprocess.run(
            ["git", "clone", "https://github.com/gabec-slop/autoresearch-gym.git", str(UPSTREAM)],
            check=True,
        )
        subprocess.run(["git", "checkout", REVISION], cwd=UPSTREAM, check=True)
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=UPSTREAM, text=True
    ).strip()
    if revision != REVISION:
        raise ValueError(f"Expected upstream {REVISION}; found {revision}")
    patch = TASKS / "upstream.patch"
    applied = (
        subprocess.run(
            ["git", "apply", "--reverse", "--check", str(patch)], cwd=UPSTREAM, capture_output=True
        ).returncode
        == 0
    )
    if not applied:
        subprocess.run(["git", "apply", "--check", str(patch)], cwd=UPSTREAM, check=True)
        subprocess.run(["git", "apply", str(patch)], cwd=UPSTREAM, check=True)
    subprocess.run(["uv", "sync", "--locked", "--extra", "dev"], cwd=UPSTREAM, check=True)


def main():
    """Forward commands to Autoresearch Gym in its own Python environment."""
    arguments = sys.argv[1:]
    if arguments == ["setup"]:
        setup()
        return
    if not arguments:
        arguments = ["--help"]
    if arguments[0] in {"run", "init-session"}:
        for flag, path in (
            ("--benchmark", TASKS / "navigation_depth.json"),
            ("--seed-candidate", TASKS / "seed_trainable.py"),
        ):
            if flag not in arguments:
                arguments += [flag, str(path)]
    if arguments[0] == "run" and "--session-dir" not in arguments:
        for flag, path in (
            ("--out-dir", ROOT / "results/autoresearch/runs"),
            ("--results-path", ROOT / "results/autoresearch/results.jsonl"),
        ):
            if flag not in arguments:
                arguments += [flag, str(path)]
    os.chdir(ROOT)
    os.environ["PYTHONPATH"] = str(ROOT / "tools")
    python = str(UPSTREAM / ".venv/bin/python")
    os.execv(python, [python, "-m", "autoresearch_gym.cli", *arguments])


if __name__ == "__main__":
    main()
