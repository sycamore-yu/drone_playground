"""Bounded, scratch-only PPO Navigation reward ablations.

Reads immutable Trial-9 proposals and emits independent A/B/C/D experiments.
Does not alter Trial 9/10, their checkpoints, or the original matrix manager.
"""

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

CASES = (
    ("depth", "b"),
    ("lidar", "c"),
    ("depth", "a"),
    ("lidar", "a"),
    ("depth", "d"),
    ("lidar", "d"),
)
CHANGE = {
    "a": "failure_cost_by_event",
    "b": "boundary_proximity",
    "c": "approach_risk",
    "d": "gamma",
}
ACCEPTED = {"accepted"}
TERMINAL_FAILURE = {"failed", "early_stopped", "frozen_benchmark_failed", "interrupted"}
SLEEP_SECONDS = 20


def reference_options(reference_root, sensor):
    """Recover the exact measured Trial-9 training recipe, never its weights."""
    path = reference_root / ".scheduler/proposals" / f"{sensor}_ppo_s0_trial9.json"
    proposal = json.loads(path.read_text())
    if not proposal.get("from_scratch") or proposal.get("initial_checkpoint") is not None:
        raise ValueError(f"Expected scratch-only provenance in {path}")
    return (
        [
            item
            for item in proposal["command"][3:]
            if item != "mode=train" and not item.startswith(("experiment=", "seed=", "output="))
        ],
        hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def overrides(sensor, letter, seed, output, options, *, baseline=False):
    """Use one versioned Hydra experiment and preserve every Trial-9 override."""
    variant = "base" if baseline else letter
    experiment = f"navigation_procedural_{sensor}_ppo_ablation_{variant}"
    return ["mode=train", f"experiment={experiment}", *options, f"seed={seed}", f"output={output}"]


def compose_recipe(python, project, arguments):
    """Perform true Hydra composition before creating an experiment directory."""
    result = subprocess.run(
        [str(python), "-m", "drone_playground.cli", "--cfg", "job", *arguments],
        cwd=project,
        env={**os.environ, "SCIPY_ARRAY_API": "1", "JAX_PLATFORMS": "cpu", "PYTHONPATH": "src"},
        text=True,
        capture_output=True,
        timeout=60,
    )
    if result.returncode:
        raise ValueError(f"Hydra preflight failed: {result.stderr[-1800:]}")
    from omegaconf import OmegaConf

    return OmegaConf.to_container(OmegaConf.create(result.stdout), resolve=True)


def validate_pair(base, candidate, letter):
    """Reject a treatment that changes more than its declared logical field."""
    if candidate["benchmark"].get("fixed_checkpoint_seeds") is not True:
        raise ValueError("Every arm must use the original fixed validation cohort")
    if candidate["results"].get("keep_best_screening") is not True:
        raise ValueError("Every arm must retain the best training checkpoint")
    if candidate["initial_checkpoint"] is not None or candidate["resume"] is not None:
        raise ValueError("Ablations must independently initialize all network weights")
    original = json.loads(json.dumps(base))
    changed = json.loads(json.dumps(candidate))
    if letter == "d":
        if changed["learning"]["options"]["gamma"] != 0.995:
            raise ValueError("Discount treatment must be exactly gamma=0.995")
        changed["learning"]["options"]["gamma"] = original["learning"]["options"]["gamma"]
    else:
        name = CHANGE[letter]
        control_shaping = original["task"]["options"]["reward_shaping"]
        treatment_shaping = changed["task"]["options"]["reward_shaping"]
        if set(treatment_shaping) - set(control_shaping) != {name}:
            raise ValueError(f"Arm {letter} must add exactly one Task reward term")
        treatment_shaping.pop(name)
    if changed != original:
        raise ValueError(f"Arm {letter} changes parameters outside its causal treatment")


def cases_status(root):
    """Read authoritative CLI run status, not TensorBoard reward heuristics."""
    status = {}
    live_paths = training_processes()[1]
    for sensor, letter in CASES:
        case_id = f"{sensor}_{letter}"
        trials = []
        for seed in (0, 1, 2):
            path = root / f"{case_id}_s{seed}"
            record_file = path / "run.json"
            record = json.loads(record_file.read_text()) if record_file.is_file() else {}
            recorded_status = record.get("status")
            if recorded_status == "running" or (path.is_dir() and recorded_status is None):
                recorded_status = "running" if str(path) in live_paths else "stale"
            trials.append(
                {
                    "seed": seed,
                    "status": recorded_status or "missing",
                    "updates": record.get("last_update"),
                    "best_screen_score": record.get("best_screen_score"),
                }
            )
        status[case_id] = trials
    return status


def training_processes():
    """Track native Python train processes and exact output directories."""
    count = 0
    outputs = set()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            tokens = (entry / "cmdline").read_bytes().decode(errors="replace").split("\0")
        except (OSError, PermissionError):
            continue
        if len(tokens) > 3 and tokens[1:4] == ["-m", "drone_playground.cli", "mode=train"]:
            count += 1
            for argument in tokens[4:]:
                if argument.startswith("output="):
                    outputs.add(argument.partition("=")[2])
    return count, outputs


def gpu_memory_free_mib():
    """Use measured VRAM admission rather than guessing device availability."""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    return int(result.stdout.strip().splitlines()[0])


def chosen_candidate(status):
    """Finish an accepted candidate's independent seeds before exploring more pilots."""
    for sensor, letter in CASES:
        trials = status[f"{sensor}_{letter}"]
        if trials[0]["status"] != "accepted":
            continue
        for seed in (1, 2):
            if (
                all(item["status"] == "accepted" for item in trials[:seed])
                and trials[seed]["status"] == "missing"
            ):
                return sensor, letter, seed
    for seed in (0, 1, 2):
        for sensor, letter in CASES:
            trials = status[f"{sensor}_{letter}"]
            if seed and any(item["status"] not in ACCEPTED for item in trials[:seed]):
                continue
            if trials[seed]["status"] == "missing":
                return sensor, letter, seed
    return None


def plan(project, reference_root, output_root, python):
    """Preflight every variation against its own Task-owned reward baseline."""
    proposals = []
    for sensor, letter in CASES:
        source_overrides, source_sha256 = reference_options(reference_root, sensor)
        identifier = f"{sensor}_{letter}"
        baseline = compose_recipe(
            python,
            project,
            overrides(
                sensor, letter, 0, str(output_root / identifier), source_overrides, baseline=True
            ),
        )
        treatment = compose_recipe(
            python,
            project,
            overrides(sensor, letter, 0, str(output_root / identifier), source_overrides),
        )
        validate_pair(baseline, treatment, letter)
        proposals.append(
            {
                "case": identifier,
                "source_trial": f"{sensor}_ppo_s0_trial9",
                "source_proposal_sha256": source_sha256,
                "seed0_experiment": treatment["task"]["options"]["reward_shaping"],
                "fixed_validation_seeds": True,
                "independent_initialization": True,
            }
        )
    return proposals


def launch(project, reference_root, root, python, sensor, letter, seed):
    """Launch a separately named tmux job after verifying its exact Hydra recipe."""
    identifier = f"{sensor}_{letter}"
    destination = root / f"{identifier}_s{seed}"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite prior experiment: {destination}")
    source_overrides, sha = reference_options(reference_root, sensor)
    args = overrides(sensor, letter, seed, str(destination), source_overrides)
    compose_recipe(python, project, args)
    destination.mkdir(parents=True)
    manifest = {
        "reference_trial": f"{sensor}_ppo_s0_trial9",
        "reference_proposal_sha256": sha,
        "treatment": letter,
        "seed": seed,
        "from_scratch": True,
        "argv": [str(python), "-m", "drone_playground.cli", *args],
    }
    (destination / "ablation.json").write_text(json.dumps(manifest, indent=2) + "\n")
    log = root / ".scheduler/logs" / f"{destination.name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    session = f"dp-abl-{sensor[:2]}-{letter}-s{seed}"
    cmd = (
        f"cd {shlex.quote(str(project))} && "
        "env SCIPY_ARRAY_API=1 XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=src "
        f"JAX_COMPILATION_CACHE_DIR={shlex.quote(str(project / '.cache/jax'))} "
        f"{shlex.join(manifest['argv'])} > {shlex.quote(str(log))} 2>&1"
    )
    subprocess.run(["tmux", "new-session", "-d", "-s", session, cmd], check=True)
    return session


def supervise(project, reference_root, root, python, *, max_parallel):
    """Run bounded pilots, then require three independently accepted seeds."""
    root.mkdir(parents=True, exist_ok=True)
    while True:
        results = cases_status(root)
        total_active = training_processes()[0]
        snapshot = {
            "observed_at_utc": datetime.now(UTC).isoformat(),
            "source": "scratch",
            "cases": results,
            "global_active_train_processes": total_active,
            "selected_baselines": [
                "depth_ppo_s0_trial9",
                "depth_ppo_s0_trial10",
                "lidar_ppo_s0_trial9",
                "lidar_ppo_s0_trial10",
            ],
            "validation": "fixed S01-S03/D01-D03, 8-screen -> 25-full -> held-out benchmark",
        }
        temp = root / ".status.tmp"
        temp.write_text(json.dumps(snapshot, indent=2) + "\n")
        temp.replace(root / "status.json")
        task = chosen_candidate(results)
        running = any(item["status"] == "running" for trials in results.values() for item in trials)
        stale = [
            case
            for case, trials in results.items()
            if any(item["status"] == "stale" for item in trials)
        ]
        if stale:
            raise RuntimeError(f"Stale experiment directories need inspection: {stale}")
        if task is None and not running:
            print("Finished bounded ablation queue; consult status.json", flush=True)
            return
        if task and total_active < max_parallel and gpu_memory_free_mib() >= 6500:
            sensor, letter, seed = task
            session = launch(project, reference_root, root, python, sensor, letter, seed)
            print("launch", session, sensor, letter, seed, flush=True)
        time.sleep(SLEEP_SECONDS)


def main():
    """Build a validated experimental queue or run its bounded supervisor."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "run", "status"))
    parser.add_argument("--reference-root", type=Path)
    parser.add_argument("--results-root", type=Path)
    parser.add_argument("--python", type=Path)
    parser.add_argument("--max-parallel", type=int, default=2)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    simulation_root = project.parents[2]
    reference_root = (
        args.reference_root or simulation_root / "new_drone_playground/results/procedural_v01"
    )
    output_root = (
        args.results_root
        or simulation_root / "new_drone_playground/results/ppo_reward_ablation_v01"
    )
    python = args.python or simulation_root / "new_drone_playground/.pixi/envs/default/bin/python"
    if args.mode == "status":
        print(json.dumps(cases_status(output_root), indent=2))
        return
    if not python.is_file():
        raise ValueError(f"Missing Python interpreter: {python}")
    checked = plan(project, reference_root, output_root, python)
    print(json.dumps(checked, indent=2), flush=True)
    if args.mode == "run":
        if args.max_parallel < 1:
            raise ValueError("max-parallel must be positive")
        supervise(project, reference_root, output_root, python, max_parallel=args.max_parallel)


if __name__ == "__main__":
    main()
