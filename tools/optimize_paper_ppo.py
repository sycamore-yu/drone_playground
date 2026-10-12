"""Bounded measured search for paper-derived PPO, using real fixed-screen evidence.

The navigation preset compares the original CNN/PointNet and paper policies.
Existing v04 baselines are adopted, never restarted. Only an accepted frozen
benchmark promotes a recipe to seeds 1/2. No model agent is launched.
"""

import argparse
import fcntl
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from drone_playground.learning.selection import screening_score

SCENES = ["S01", "S02", "S03", "D01", "D02", "D03"]
CASES = {
    "fan_equations": ("fan", "navigation_paper_fan2025_equations", None),
    "visfly_source": ("visfly", "navigation_paper_visfly2025_source", None),
    "fan_margin2": ("fan", "navigation_paper_fan2025_equations_margin2", "fan_equations"),
    "visfly_shared": ("visfly", "navigation_paper_visfly2025_shared", "visfly_source"),
}
BASE_UNITS = {
    "fan_equations_s0": "dp-paper-fan-equations-s0-20261012",
    "visfly_source_s0": "dp-paper-visfly-source-s0-20261012",
}
FINISHED = {
    "early_stopped",
    "accepted",
    "frozen_benchmark_failed",
    "failed",
    "pruned",
    "budget_exhausted",
}
ADOPTED_RUNS = {}
CASE_OPTIONS = {}
NAVIGATION_SEARCH = False
MODALITY = {"depth": "depth", "visfly": "depth", "lidar": "lidar", "fan": "lidar"}


def configure_navigation_search(results_root):
    """Add the two original architectures and retain verified paper baselines."""
    global NAVIGATION_SEARCH, CASES, BASE_UNITS, ADOPTED_RUNS, CASE_OPTIONS
    NAVIGATION_SEARCH = True
    depth = "navigation_search_depth"
    lidar = "navigation_search_lidar"
    fan = "navigation_paper_fan2025_equations_margin2"
    visfly = "navigation_paper_visfly2025_shared"
    CASES = {
        "depth_base": ("depth", depth, None),
        "lidar_base": ("lidar", lidar, None),
        "fan_base": ("fan", fan, None),
        "visfly_base": ("visfly", visfly, None),
        "depth_lowlr": ("depth", depth, "depth_base"),
        "lidar_lowlr": ("lidar", lidar, "lidar_base"),
        "fan_lowlr": ("fan", fan, "fan_base"),
        "visfly_lowstd": ("visfly", visfly, "visfly_base"),
        "depth_goal": ("depth", depth, "depth_base"),
        "lidar_goal": ("lidar", lidar, "lidar_base"),
        "fan_exploration": ("fan", fan, "fan_base"),
        "visfly_hoverthrust": ("visfly", visfly, "visfly_base"),
    }
    CASE_OPTIONS = {
        "depth_lowlr": ["learning.options.lr=0.00005"],
        "lidar_lowlr": ["learning.options.lr=0.000025"],
        "fan_lowlr": ["learning.options.lr=0.000025"],
        "visfly_lowstd": ["learning.options.initial_log_std=-1.0"],
        "depth_goal": ["simulation.navigation_goal_observation=true"],
        "lidar_goal": ["simulation.navigation_goal_observation=true"],
        "fan_exploration": ["learning.options.initial_log_std=-1.5"],
        "visfly_hoverthrust": ["method.action.thrust_to_weight_bounds=[0.0,2.0]"],
    }
    ADOPTED_RUNS = {
        "fan_base_s0": results_root / "paper_ppo_v04/fan_margin2_s0",
        "visfly_base_s0": results_root / "paper_ppo_v04/visfly_shared_s0",
    }
    BASE_UNITS = {
        "fan_base_s0": "dp-paper-study-fan-margin2-s0",
        "visfly_base_s0": "dp-paper-study-visfly-shared-s0",
    }


def interaction_controls(family, *, num_envs, horizon):
    """Convert measured family budgets to PPO updates, not arbitrary equal epochs."""
    per_update = num_envs * horizon
    # Fan's measured first fixed-scene success required approximately 4M
    # transitions. Its minimum budget must not prune before that evidence.
    minimum, maximum = {
        "depth": (737_280, 2_211_840),
        "lidar": (737_280, 2_211_840),
        "fan": (4_423_680, 8_847_360),
        "visfly": (1_474_560, 4_423_680),
    }[family]
    if 122_880 % per_update or minimum % per_update or maximum % per_update:
        raise ValueError("Search budgets must be divisible by the rollout sample count")
    return {
        "evaluation_interval": 122_880 // per_update,
        "checkpoint_interval": 61_440 // per_update,
        "min_updates": minimum // per_update,
        "max_interactions": maximum,
    }


def stable_modalities(records):
    """Require all three independent seeds for one recipe, not mixed recipes."""
    return {
        MODALITY[family]
        for case, (family, _, _) in CASES.items()
        if all(records[f"{case}_s{i}"]["status"] == "accepted" for i in range(3))
    }


def navigation_candidate(records):
    """Cover each architecture first, then prefer measured improvements fairly."""
    stable = stable_modalities(records)
    for case, (family, _, parent) in CASES.items():
        key = f"{case}_s0"
        if (
            parent is None
            and key not in ADOPTED_RUNS
            and records[key]["status"] == "missing"
            and MODALITY[family] not in stable
        ):
            return case, 0
    candidates = []
    for index, (case, (family, _, parent)) in enumerate(CASES.items()):
        if parent is None or records[f"{case}_s0"]["status"] != "missing":
            continue
        if MODALITY[family] in stable or records[f"{parent}_s0"]["status"] not in FINISHED:
            continue
        peers = [other for other, spec in CASES.items() if spec[0] == family]
        if any(records[f"{other}_s0"]["status"] == "running" for other in peers):
            continue
        count = sum(records[f"{other}_s0"]["status"] in FINISHED for other in peers)
        evidence = records[f"{parent}_s0"].get("best")
        score = evidence["value"] if evidence is not None else -1.0
        candidates.append(((count, -score, index), case))
    return (min(candidates)[1], 0) if candidates else None


def trial_options(case, seed, output):
    """Resolve each recipe as a new run with one declared parameter treatment."""
    family, experiment, _ = CASES[case]
    options = [
        "mode=train",
        f"experiment={experiment}",
        f"output={output}",
        f"seed={seed}",
        "simulation.num_envs=16",
        "initial_checkpoint=null",
        "resume=null",
        *CASE_OPTIONS.get(case, []),
    ]
    if NAVIGATION_SEARCH:
        horizon = {"depth": 32, "lidar": 32, "fan": 128, "visfly": 256}[family]
        settings = interaction_controls(family, num_envs=16, horizon=horizon)
        options += [
            f"learning.options.horizon={horizon}",
            f"learning.evaluation_interval={settings['evaluation_interval']}",
            f"learning.checkpoint_interval={settings['checkpoint_interval']}",
            f"++learning.max_interactions={settings['max_interactions']}",
            f"early_stopping.min_updates={settings['min_updates']}",
            "early_stopping.patience=3",
            "early_stopping.min_delta=0.005",
            "++screening.probe_between_full=true",
            "++screening.full_every=3",
            "results.keep_best_screening=true",
        ]
    return options


def full_screen_evidence(directory):
    """Read only complete six-scene, eight-episode reports; unknown remains None."""
    evidence = []
    for path in directory.glob("checkpoints/model_*/screening/report.json"):
        try:
            report = json.loads(path.read_text())
        except FileNotFoundError:
            continue  # Atomic checkpoint pruning may race this read-only observer.
        if not report.get("complete") or set(report.get("reports", {})) != set(SCENES):
            continue
        value = screening_score(report, "navigation", SCENES)
        evidence.append(
            {
                "value": value,
                "successes": sum(row["successes"] for row in report["reports"].values()),
                "per_scene": {key: report["reports"][key]["successes"] for key in SCENES},
                "checkpoint_update": int(path.parents[1].name.split("_")[-1]),
                "report": str(path),
            }
        )
    return max(evidence, key=lambda row: row["value"], default=None)


def next_trial(records):
    """Prioritize independent confirmations, then one informed candidate per family."""
    for case in CASES:
        for seed in (1, 2):
            earlier = [records[f"{case}_s{i}"]["status"] for i in range(seed)]
            if all(status == "accepted" for status in earlier) and (
                records[f"{case}_s{seed}"]["status"] == "missing"
            ):
                return case, seed
    if NAVIGATION_SEARCH:
        return navigation_candidate(records)
    for case, (family, _, parent) in CASES.items():
        if parent is None or records[f"{case}_s0"]["status"] != "missing":
            continue
        family_has_acceptance = any(
            spec[0] == family
            and records[f"{other}_s0"]["status"] == "accepted"
            and all(
                records[f"{other}_s{seed}"]["status"] not in FINISHED - {"accepted"}
                for seed in (1, 2)
            )
            for other, spec in CASES.items()
        )
        if not family_has_acceptance and records[f"{parent}_s0"]["status"] in FINISHED:
            return case, 0
    return None


def unit_name(identifier):
    """Keep one stable systemd identity per candidate, including existing baselines."""
    prefix = "dp-nav-search" if NAVIGATION_SEARCH else "dp-paper-study"
    return BASE_UNITS.get(identifier, f"{prefix}-{identifier.replace('_', '-')}")


def active_units():
    """Return running user services without inspecting or signaling other workloads."""
    result = subprocess.run(
        [
            "systemctl",
            "--user",
            "list-units",
            "--type=service",
            "--state=running",
            "--no-legend",
            "--plain",
            "--no-pager",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    return {line.split()[0] for line in result.stdout.splitlines() if line.strip()}


def training_process_count():
    """Count real Python module trainers, accepting both python -m and python -u -m."""
    count = 0
    for proc in Path("/proc").iterdir():
        if not proc.name.isdecimal():
            continue
        try:
            argv = (proc / "cmdline").read_bytes().decode(errors="replace").split("\0")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if (
            argv
            and Path(argv[0]).name.startswith("python")
            and "-m" in argv
            and "drone_playground.cli" in argv
            and "mode=train" in argv
        ):
            count += 1
    return count


def collect(root, previous):
    """Persist the best actually observed score even after obsolete reports are pruned."""
    units = active_units()
    records = {}
    for case in CASES:
        for seed in range(3):
            key = f"{case}_s{seed}"
            directory = ADOPTED_RUNS.get(key, root / key)
            identity = directory / "run.json"
            running = f"{unit_name(key)}.service" in units
            record = json.loads(identity.read_text()) if identity.exists() else {}
            status = record.get("status", "running" if running else "missing")
            actual_status = status
            if status == "interrupted" and (directory / "study_pruned.json").is_file():
                status = "pruned"
            if status == "running" and not running:
                status = "stale"
            candidate = full_screen_evidence(directory)
            best = previous.get(key, {}).get("best")
            if candidate is not None and (best is None or candidate["value"] > best["value"]):
                best = candidate
            records[key] = {
                "status": status,
                "run_status": actual_status,
                "best": best,
                "updates": record.get("updates", record.get("last_update")),
                "configuration_sha256": record.get("config_sha256"),
                "source_sha256": record.get("source_sha256"),
                "directory": str(directory),
                "adopted": key in ADOPTED_RUNS,
                "error": record.get("error"),
            }
    return records


def launch(project, root, python, case, seed):
    """Validate Hydra, reserve a new run, and launch without CPU/memory/swap quotas."""
    key = f"{case}_s{seed}"
    out = root / key
    if out.exists():
        raise FileExistsError(f"Never overwrite a previous trial: {out}")
    options = trial_options(case, seed, out)
    env = {
        **os.environ,
        "SCIPY_ARRAY_API": "1",
        "PYTHONPATH": str(project / "src"),
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "OPENBLAS_NUM_THREADS": "1",
    }
    checked = subprocess.run(
        [str(python), "-m", "drone_playground.cli", "--cfg", "job", *options],
        cwd=project,
        env={**env, "JAX_PLATFORMS": "cpu"},
        check=True,
        capture_output=True,
        text=True,
        timeout=90,
    )
    if NAVIGATION_SEARCH:
        from omegaconf import OmegaConf

        resolved = OmegaConf.to_container(OmegaConf.create(checked.stdout), resolve=True)
        if (
            resolved["simulation"]["procedural"]["min_start_goal_distance_m"] != 80.0
            or not resolved["benchmark"]["fixed_checkpoint_seeds"]
            or resolved["initial_checkpoint"] is not None
            or resolved["resume"] is not None
        ):
            raise ValueError("Search changed long-course, fixed-cohort or scratch-init contract")
    out.mkdir()
    command = [str(python), "-u", "-m", "drone_playground.cli", *options]
    (out / "trial.json").write_text(
        json.dumps(
            {
                "case": case,
                "seed": seed,
                "family": CASES[case][0],
                "parent": CASES[case][2],
                "from_scratch": True,
                "argv": command,
                "composed_yaml_sha256": hashlib.sha256(checked.stdout.encode()).hexdigest(),
                "git_commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=project, text=True
                ).strip(),
            },
            indent=2,
        )
        + "\n"
    )
    log = root / ".scheduler" / f"{key}.log"
    exported = [
        "env",
        "SCIPY_ARRAY_API=1",
        "XLA_PYTHON_CLIENT_PREALLOCATE=false",
        "OPENBLAS_NUM_THREADS=1",
        "OMP_NUM_THREADS=4",
        "MKL_NUM_THREADS=1",
        f"PYTHONPATH={project / 'src'}",
        f"JAX_COMPILATION_CACHE_DIR={project / '.cache/jax'}",
    ]
    shell = shlex.join([*exported, *command]) + f" > {shlex.quote(str(log))} 2>&1"
    subprocess.run(
        [
            "systemd-run",
            "--user",
            "--collect",
            f"--unit={unit_name(key)}",
            "--property=RuntimeMaxSec=21600",
            f"--working-directory={project}",
            "/bin/bash",
            "-lc",
            shell,
        ],
        check=True,
        timeout=30,
    )


def main():
    """Run the measured finite study or print its current evidence snapshot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "status"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--suite", choices=("paper", "navigation"), default="paper")
    parser.add_argument("--max-parallel", type=int, default=2)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    if args.max_parallel < 1:
        raise ValueError("max-parallel must be positive")
    if args.suite == "navigation":
        configure_navigation_search(args.root.parent)
        missing = [str(path) for path in ADOPTED_RUNS.values() if not (path / "run.json").exists()]
        if missing:
            raise FileNotFoundError(f"The adopted paper baselines are missing: {missing}")
    snapshot = args.root / "study.json"
    if args.mode == "status":
        previous = json.loads(snapshot.read_text()).get("trials", {}) if snapshot.exists() else {}
        print(json.dumps({"trials": collect(args.root, previous)}, indent=2))
        return
    args.root.mkdir(parents=True, exist_ok=True)
    directory = args.root / ".scheduler"
    directory.mkdir(exist_ok=True)
    with (directory / "study.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = json.loads(snapshot.read_text()).get("trials", {}) if snapshot.exists() else {}
        while True:
            records = collect(args.root, previous)
            status = {
                "observed_at_utc": datetime.now(UTC).isoformat(),
                "direction": "maximize",
                "objective": "min(scene rates) + 0.25*mean(scene rates)",
                "pilot_budget": len(CASES),
                "suite": args.suite,
                "stable_modalities": sorted(stable_modalities(records))
                if NAVIGATION_SEARCH
                else [],
                "independent_training_seeds": [0, 1, 2],
                "trials": records,
            }
            candidate = next_trial(records)
            active = any(record["status"] == "running" for record in records.values())
            blocked = [
                key
                for key, record in records.items()
                if record["status"] in {"stale", "interrupted"}
            ]
            complete = candidate is None and not active
            status["status"] = "blocked" if blocked else "finished" if complete else "running"
            temporary = snapshot.with_suffix(".tmp")
            temporary.write_text(json.dumps(status, indent=2) + "\n")
            temporary.replace(snapshot)
            if blocked or complete:
                print(json.dumps(status, indent=2), flush=True)
                return
            if candidate is not None and training_process_count() < args.max_parallel:
                memory = {
                    line.split(":")[0]: int(line.split()[1])
                    for line in Path("/proc/meminfo").read_text().splitlines()
                    if line.startswith("MemAvailable:")
                }
                free = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                    text=True,
                    timeout=15,
                )
                pressure = next(
                    float(line.split()[1].split("=")[1])
                    for line in Path("/proc/pressure/io").read_text().splitlines()
                    if line.startswith("full ")
                )
                if (
                    memory["MemAvailable"] >= 8 * 1024 * 1024
                    and int(free.strip().splitlines()[0]) >= 7000
                    and pressure < 40.0
                ):
                    launch(project, args.root, args.python, *candidate)
                    print(datetime.now(UTC).isoformat(), "launched", candidate, flush=True)
            previous = records
            time.sleep(30)


if __name__ == "__main__":
    main()
