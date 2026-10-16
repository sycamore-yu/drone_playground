"""Run Autoresearch Gym candidates through the existing Drone Playground CLI."""

from __future__ import annotations

import csv
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT.parent / "autoresearch-gym"
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
            "session_dir": str(bundle.session_dir)
            if getattr(bundle, "session_dir", None)
            else None,
            "run_id": getattr(bundle, "run_id", None),
            "tag": getattr(bundle, "tag", None),
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
            *request.get("overrides", []),
            *[f"++{key}={json.dumps(value)}" for key, value in request["recipe"].items()],
            "initial_checkpoint=null",
            f"output={output}",
        ]
        if phase == "train":
            return [
                *args,
                "mode=train",
                f"resume={request.get('resume') or 'null'}",
                "checkpoint=null",
                f"seed={request['train_seed']}",
                f"learning.max_interactions={request['train_env_steps']}",
            ]
        return [
            *args,
            "mode=benchmark",
            "resume=null",
            # Evaluation uses its declared seed independently of the training seed.
            "seed=0",
            f"checkpoint={checkpoint}",
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
                str(ROOT / "tools/autoresearch.py"),
                "worker",
                str(bundle.external_dir / "request.json"),
                phase,
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
            **native_training_view(directory),
            "budget_mode": "env_steps",
            "train_env_steps": request["train_env_steps"],
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
            "evaluation_complete": report["complete"],
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
    ensure_dashboard_assets()


def ensure_dashboard_assets():
    """Serve upstream dashboard files alongside local experiment results."""
    source = UPSTREAM / "dashboard"
    link = ROOT / "dashboard"
    if link.is_symlink() and link.resolve() == source.resolve():
        return
    if link.exists() or link.is_symlink():
        raise FileExistsError(f"Dashboard asset path already exists: {link}")
    if not (source / "index.html").is_file():
        raise FileNotFoundError(f"Missing upstream dashboard: {source}")
    link.symlink_to(Path(os.path.relpath(source, ROOT)), target_is_directory=True)


def register_session(directory):
    """Expose the canonical results directory through upstream's session index."""
    alias = ROOT / "autoresearch_runs/sessions" / directory.name
    alias.parent.mkdir(parents=True, exist_ok=True)
    if alias.is_symlink() and alias.resolve() == directory.resolve():
        return
    if alias.exists() or alias.is_symlink():
        raise FileExistsError(f"Session alias already exists: {alias}")
    alias.symlink_to(os.path.relpath(directory, alias.parent), target_is_directory=True)


def native_training_view(directory):
    """Project native measurements once for both live status and final summaries."""
    from drone_playground.simulation.policy import checkpoint_summary

    run_path = directory / "run.json"
    run = json.loads(run_path.read_text()) if run_path.exists() else {}
    path = directory / "metrics.jsonl"
    content = path.read_text() if path.exists() else ""
    lines = content.splitlines()
    if content and not content.endswith("\n"):
        lines = lines[:-1]  # A concurrent writer has not committed its last event yet.
    events = [json.loads(line) for line in lines]
    if run.get("status") != "running" and "selection" in run:
        selection = run["selection"]
        events.append(
            {
                "event": "update",
                "completed_episodes": selection["episodes_completed"],
                "episode_return_total": selection["episode_return_sum"],
                "episode_length_total": selection["episode_length_sum"],
                "episode_success_total": selection["episode_successes"],
                "interactions": run["interactions"],
                "update": run["updates"],
                "wall_seconds": run["wall_seconds"],
                "training_wall_seconds": run["training_wall_seconds"],
            }
        )
    latest = {}
    records, previous = (
        [],
        dict.fromkeys(
            (
                "completed_episodes",
                "episode_return_total",
                "episode_length_total",
                "episode_success_total",
            ),
            0,
        ),
    )
    for event in events:
        if event["event"] == "update":
            latest.update(event)
            count = event["completed_episodes"] - previous["completed_episodes"]
            if count < 0:
                raise ValueError("Native episode counters moved backwards")
            if count:
                success_rate = (
                    event["episode_success_total"] - previous["episode_success_total"]
                ) / count
                records.append(
                    {
                        "record_type": "train_collection_window",
                        "episode": len(records) + 1,
                        "return": (event["episode_return_total"] - previous["episode_return_total"])
                        / count,
                        "length": (event["episode_length_total"] - previous["episode_length_total"])
                        / count,
                        "episodes_in_window": count,
                        "success": success_rate == 1.0,
                        "success_rate": success_rate,
                        "step": event["interactions"],
                        "elapsed_seconds": event["wall_seconds"],
                        "info_metrics": {},
                    }
                )
            previous = {key: event[key] for key in previous}
        elif event["event"] == "selection" and event["source"] == "evaluation":
            rows = event["episode_records"]
            success_rate = sum(row["success"] for row in rows) / len(rows)
            records.append(
                {
                    "record_type": "policy_probe",
                    "episode": len(records) + 1,
                    "return": event["mean_reward"],
                    "length": sum(row["length"] for row in rows) / len(rows),
                    "success": success_rate == 1.0,
                    "step": event["interactions"],
                    "elapsed_seconds": event["wall_seconds"],
                    "probe_episodes": len(rows),
                    "probe_seed_start": rows[0]["seed"],
                    "deterministic": True,
                    "info_metrics": {
                        "policy_probe_success_rate": success_rate,
                        "policy_probe_return": event["mean_reward"],
                    },
                }
            )
    completed = previous["completed_episodes"]
    published = directory / "checkpoints/latest"
    checkpoint = checkpoint_summary(published) if published.is_symlink() else {}
    selected = checkpoint.get("best_checkpoint") or checkpoint.get("latest_checkpoint")
    return {
        "total_steps": int(latest.get("interactions", 0)),
        "env_steps": int(latest.get("interactions", 0)),
        "policy_updates": int(latest.get("update", 0)),
        "num_envs": run.get("num_envs"),
        "rollout_horizon": run.get("rollout_horizon"),
        "algorithm": run.get("algorithm"),
        "task": run.get("task"),
        "started_at": run.get("started_unix"),
        "elapsed_seconds": latest.get("training_wall_seconds", 0.0),
        "last_metrics": {
            key: value for key, value in latest.items() if isinstance(value, (int, float))
        },
        "checkpoint_path": selected,
        "checkpoint_selection": "best" if checkpoint.get("best_checkpoint") else "latest",
        "best_source": checkpoint.get("best_source"),
        "best_mean_reward": checkpoint.get("best_mean_reward"),
        "episode_records": records,
        "episodes_completed": completed,
        "completed_episodes": completed,
        "episode_batches": sum(row["record_type"] == "train_collection_window" for row in records),
        "avg_return": previous["episode_return_total"] / completed if completed else None,
        "avg_length": previous["episode_length_total"] / completed if completed else None,
        "success_rate": previous["episode_success_total"] / completed if completed else None,
        "curve_status": "available" if records else "unsupported",
        "curve_status_reason": None if records else "No completed episodes have been recorded.",
    }


def run_worker(request_path, phase, arguments):
    """Stream native progress into the existing Autoresearch dashboard contract."""
    request_path = Path(request_path)
    request = json.loads(request_path.read_text())
    session = Path(request["session_dir"]) if request.get("session_dir") else None
    started = time.time()
    payload = {
        "run": {
            "run_id": request["run_id"],
            "tag": request["tag"],
            "started_at": started,
            "train_env_steps": request["train_env_steps"],
            "budget_mode": "env_steps",
        },
        "current": {},
        "episodes": [],
        "latest_losses": {},
    }

    def publish(status):
        if session is None:
            return
        view = native_training_view(request_path.parent / "train")
        payload["run"].update(
            status=status,
            updated_at=time.time(),
            started_at=view["started_at"] or started,
            num_envs=view["num_envs"],
            rollout_horizon=view["rollout_horizon"],
            algorithm=view["algorithm"],
            task=view["task"],
        )
        payload["current"] = {
            "status": status,
            "step": view["total_steps"],
            "env_steps": view["env_steps"],
            "policy_updates": view["policy_updates"],
            "episodes_complete": view["episodes_completed"],
            "completed_episodes": view["completed_episodes"],
            "episode_batches": view["episode_batches"],
            "avg_return": view["avg_return"],
            "success_rate": view["success_rate"],
            "info_metrics": view["last_metrics"],
        }
        payload["latest_losses"] = view["last_metrics"]
        payload["episodes"] = view["episode_records"]
        live = session / "live"
        live.mkdir(parents=True, exist_ok=True)
        temporary = live / ".current_run_metrics.tmp"
        temporary.write_text(json.dumps(payload, allow_nan=False))
        temporary.replace(live / "current_run_metrics.json")

    publish("training" if phase == "train" else "evaluating")
    with (request_path.parent / f"{phase}-console.log").open("w") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", "drone_playground.cli", *arguments],
            stdout=subprocess.PIPE,
            text=True,
        )
        handlers = {
            sig: signal.signal(sig, lambda signum, _: process.send_signal(signum))
            for sig in (signal.SIGINT, signal.SIGTERM)
        }
        try:
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(event, dict):
                    publish("training" if phase == "train" else "evaluating")
            code = process.wait()
        except BaseException:
            process.terminate()
            process.wait()
            publish("failed")
            raise
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
    publish("failed" if code else "trained" if phase == "train" else "evaluated")
    return code


def prepare_training(overrides):
    """Turn a public Hydra recipe into an ordinary Autoresearch Gym task."""
    from hydra import compose, initialize_config_dir

    reserved = {"mode", "output", "checkpoint", "initial_checkpoint"}
    if any(item.lstrip("+").split("=", 1)[0] in reserved for item in overrides):
        raise ValueError("Autoresearch owns mode, output, and fresh candidate initialization")
    with initialize_config_dir(
        version_base=None, config_dir=str(ROOT / "src/drone_playground/configs")
    ):
        config = compose(config_name="config", overrides=list(overrides))
    if config.task.name == "navigation" and config.scene.name != "train/procedural":
        raise ValueError("OA training uses procedural geometry; fixed maps belong to Benchmark")
    steps = config.learning.max_interactions
    batch = config.simulation.num_envs * config.learning.options.horizon
    if type(steps) is not int or steps < 1 or steps % batch:
        raise ValueError("learning.max_interactions must be a positive multiple of worlds*horizon")
    scenes = list(
        config.benchmark.scenes
        or (
            ["S01", "S02", "S03", "S06", "D01", "D02", "D03", "D06"]
            if config.task.name == "navigation"
            else [config.scene.name]
        )
    )
    episodes = config.benchmark.episodes
    if episodes is None:
        episodes = 25 if config.task.name == "navigation" else 100
    if type(episodes) is not int or episodes < 1:
        raise ValueError("benchmark.episodes must be a positive integer")
    experiment = next(
        (item.split("=", 1)[1] for item in overrides if item.startswith("experiment=")), None
    )
    if experiment is None:
        raise ValueError("Specify a training experiment, for example experiment=tracking")
    backward = config.learning.backward_model or "native"
    task = {
        "name": f"{experiment}_{config.learning.algorithm}_{backward}",
        "env_id": "external:drone-playground",
        "env_kwargs": {
            "experiment": experiment,
            "algorithm": config.learning.algorithm,
            "overrides": list(overrides),
            "eval_scenes": scenes,
            "resume": str(Path(config.resume).resolve()) if config.resume else None,
        },
        "train_episodes": 0,
        "eval_episodes": episodes * len(scenes),
        "max_steps": int(config.task.duration * config.simulation.method_hz),
        "primary_metric": "eval_success_rate",
        "primary_metric_mode": "maximize",
        "train_seed": config.seed,
        "eval_seed_start": (
            config.benchmark.seed_base
            if any(value.startswith("benchmark.seed_base=") for value in overrides)
            else 1_600_000
        ),
        "device": "external",
        "execution_backend": {
            "kind": "external",
            "name": "drone_playground",
            "adapter": "autoresearch:DronePlaygroundBackend",
            "execution_target": "local",
            "train_env_steps": steps,
            "supports_live_frame": False,
            "supports_sampled_trajectory": False,
        },
    }
    root = ROOT / "results/autoresearch/tasks"
    root.mkdir(parents=True, exist_ok=True)
    path = (
        Path(tempfile.mkdtemp(prefix=f"{task['name']}_s{config.seed}_", dir=root))
        / "benchmark.json"
    )
    path.write_text(json.dumps(task, indent=2) + "\n")
    session = ROOT / "results/autoresearch/sessions" / path.parent.name
    candidates = session / "candidates"
    candidates.mkdir(parents=True)
    candidate = candidates / "pass01_baseline.py"
    candidate.write_text((TASKS / "seed_trainable.py").read_text())
    register_session(session)
    return [
        "run",
        "--benchmark",
        str(path),
        "--tag",
        path.parent.name,
        "--no-dashboard",
        "--session-dir",
        str(session),
        "--candidate",
        str(candidate),
    ]


def main():
    """Forward commands to Autoresearch Gym in its own Python environment."""
    arguments = sys.argv[1:]
    if arguments and arguments[0] == "worker":
        raise SystemExit(run_worker(arguments[1], arguments[2], arguments[3:]))
    if arguments and arguments[0] == "train":
        arguments = prepare_training(arguments[1:])
    if arguments == ["setup"]:
        setup()
        return
    if not arguments:
        arguments = ["--help"]
    if arguments[0] in {"run", "init-session", "dashboard", "session-dashboard"}:
        ensure_dashboard_assets()
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
    os.environ["PYTHONPATH"] = os.pathsep.join((str(ROOT / "tools"), str(ROOT / "src")))
    python = str(UPSTREAM / ".venv/bin/python")
    os.execv(python, [python, "-m", "autoresearch_gym.cli", *arguments])


if __name__ == "__main__":
    main()
