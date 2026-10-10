"""Independently benchmark the FIRST valid old ZIP checkpoint, at most once.

This is a one-time migration of the *selection decision*, not an initialization
from old weights or a repeatable benchmark-based checkpoint search. Existing
run.json and all older checkpoint evidence remain untouched. Use --inspect to
check the choice without consuming GPU resources.

Example:
  python tools/freeze_legacy_candidate.py --run results/scratch_v02/depth_shac_s1 --inspect
  python tools/freeze_legacy_candidate.py --run results/scratch_v02/depth_shac_s1
"""

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("SCIPY_ARRAY_API", "1")

from omegaconf import OmegaConf

from drone_playground import cli
from drone_playground.simulation.records import atomic_json


def first_validated_model(run: Path) -> tuple[Path, dict]:
    """Select the earliest *complete* original checkpoint validation."""
    saved = OmegaConf.load(run / "config.yaml")
    if saved.initial_checkpoint is not None or saved.learning.selection != (
        "last_of_first_three_consecutive_passes"
    ):
        raise ValueError("Expected a scratch-initialized legacy three-pass run")
    reports = []
    for path in (run / "checkpoints").glob("step-*/report.json"):
        report = json.loads(path.read_text())
        if (
            report.get("mode") == "checkpoint_eval"
            and report.get("complete") is True
            and report.get("passed") is True
            and 1_000_000 <= report.get("seed_base", 0) < 2_000_000
        ):
            policy = path.parent / "policy.zip"
            if policy.is_file():
                reports.append((int(path.parent.name.removeprefix("step-")), policy, report))
    if not reports:
        raise ValueError("No complete, originally passing frozen-policy candidate exists")
    _, policy, report = min(reports, key=lambda item: item[0])
    return policy, report


def audit(run: Path, *, inspect: bool):
    """Evaluate the frozen candidate once, using only never-seen benchmark seeds."""
    run = run.resolve(strict=True)
    policy, validation = first_validated_model(run)
    destination = run.parent / ".scheduler" / "frozen_audits" / run.name
    if inspect:
        print(
            json.dumps(
                {
                    "run": str(run),
                    "policy": str(policy),
                    "checkpoint_validation_seed_base": validation["seed_base"],
                    "output": str(destination),
                    "already_audited": (destination / "audit.json").exists(),
                },
                indent=2,
            )
        )
        return
    if destination.exists():
        raise FileExistsError(
            f"Refusing to repeat a frozen benchmark or overwrite its evidence: {destination}"
        )
    destination.mkdir(parents=True)
    manifest = {
        "status": "running",
        "reason": "first_full_checkpoint_validation",
        "source_run": str(run),
        "checkpoint": str(policy.relative_to(run)),
        "source_validation": str((policy.parent / "report.json").relative_to(run)),
        "source_validation_seed_base": validation["seed_base"],
        "started_unix": time.time(),
        "benchmark_attempts": 1,
    }
    atomic_json(destination / "audit.json", manifest)
    config = OmegaConf.load(run / "config.yaml")
    config.mode = "benchmark"
    config.output = str(destination)
    config.checkpoint = str(policy)
    config.resume = None
    # The v2 resolved config predates the Results v3 section entirely.
    OmegaConf.update(config, "results", {"format": "orbax", "keep_checkpoints": 3}, force_add=True)
    config.benchmark.seed_base = None
    config.benchmark.episodes = None
    config.benchmark.scenes = None
    config.benchmark.record_replay = False
    config.benchmark.record_trajectories = False
    try:
        # This calls the same frozen-policy CLI validation and execution path
        # used for formally accepted seeds, with the original resolved recipe.
        cli.main.__wrapped__(config)
        report = json.loads((destination / "report.json").read_text())
        passed = (
            report.get("mode") == "benchmark"
            and report.get("complete") is True
            and report.get("passed") is True
            and report.get("seed_base", 0) >= 2_000_000
        )
        manifest.update(
            status="accepted" if passed else "failed",
            benchmark_seed_base=report["seed_base"],
            benchmark_passed=passed,
            completed_unix=time.time(),
        )
    except BaseException as error:
        manifest.update(
            status="error",
            exception=f"{type(error).__name__}: {error}",
            completed_unix=time.time(),
        )
        raise
    finally:
        atomic_json(destination / "audit.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)


def main():
    """Require an explicit run path and never silently re-evaluate test seeds."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--inspect", action="store_true")
    arguments = parser.parse_args()
    audit(arguments.run, inspect=arguments.inspect)


if __name__ == "__main__":
    main()
