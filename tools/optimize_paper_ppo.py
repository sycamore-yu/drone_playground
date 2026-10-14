"""Bounded measured search for paper-derived PPO, using real fixed-screen evidence.

The navigation preset compares the original CNN/PointNet and paper policies.
Each navigation baseline uses the same controls as its treatments. An accepted frozen
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
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from drone_playground.learning.selection import PatienceConfig, screening_score, update_patience

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
CASE_OPTIONS = {}
NAVIGATION_SEARCH = False
MODALITY = {"depth": "depth", "visfly": "depth", "lidar": "lidar", "fan": "lidar"}
ALLOCATION_POLICY = "navigation-feedback-20261014"


def write_json(path, value):
    """Replace scheduler evidence atomically so restart never reads half a manifest."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def configure_navigation_search():
    """Compare all four architectures with fresh, matched family baselines."""
    global NAVIGATION_SEARCH, CASES, BASE_UNITS, CASE_OPTIONS
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
        "depth_mixed": ("depth", depth, "depth_base"),
        "depth_mixed_lowlr": ("depth", depth, "depth_mixed"),
        "lidar_mixed": ("lidar", lidar, "lidar_base"),
        "depth_world_rotation": ("depth", depth, "depth_base"),
        "lidar_world_rotation": ("lidar", lidar, "lidar_base"),
        "lidar_batch64": ("lidar", lidar, "lidar_base"),
        "lidar_privileged": ("lidar", lidar, "lidar_base"),
        "lidar_velocity1": ("lidar", lidar, "lidar_base"),
        "lidar_horizon128": ("lidar", lidar, "lidar_base"),
        "lidar_task_only": ("lidar", lidar, "lidar_base"),
        "depth_fixed_heading": ("depth", depth, "depth_base"),
        "lidar_fixed_heading": ("lidar", lidar, "lidar_base"),
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
        "depth_mixed": ["simulation.procedural.min_start_goal_distance_m=0.0"],
        "depth_mixed_lowlr": ["learning.options.lr=0.00005"],
        "lidar_mixed": ["simulation.procedural.min_start_goal_distance_m=0.0"],
        "depth_world_rotation": ["++observation.navigation_world_rotation=true"],
        "lidar_world_rotation": ["++observation.navigation_world_rotation=true"],
        "lidar_batch64": ["simulation.num_envs=64"],
        "lidar_privileged": [
            "learning.options.critic_uses_sensor=false",
            "learning.options.critic_uses_privileged=true",
        ],
        "lidar_velocity1": ["++learning.options.perception_loss.velocity_window=1"],
        "lidar_horizon128": ["learning.options.horizon=128"],
        "lidar_task_only": ["learning.options.perception_weight=0.0"],
        "depth_fixed_heading": ["method.action.heading=fixed"],
        "lidar_fixed_heading": ["method.action.heading=fixed"],
        "fan_exploration": ["learning.options.initial_log_std=-1.5"],
        "visfly_hoverthrust": ["method.action.thrust_to_weight_bounds=[0.0,2.0]"],
    }
    BASE_UNITS = {}


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
    """Cover idle modalities, then rank the declared candidates by real evidence."""
    stable = stable_modalities(records)
    active = Counter(
        MODALITY[CASES[key.rsplit("_s", 1)[0]][0]]
        for key, row in records.items()
        if row["status"] == "running"
    )
    candidates = []
    for index, (case, (family, _, parent)) in enumerate(CASES.items()):
        modality = MODALITY[family]
        if modality in stable or records[f"{case}_s0"]["status"] != "missing":
            continue
        if parent is not None and records[f"{parent}_s0"]["status"] not in FINISHED:
            continue
        control = records[f"{parent}_s0"] if parent is not None else {}
        best, latest = control.get("best"), control.get("latest")
        score = best["value"] if best else 0.0
        priority = 2
        outcomes = latest.get("outcomes", {}) if latest else {}
        if best and latest and latest["value"] < best["value"] and case.endswith("_lowlr"):
            priority = 0
        elif outcomes.get("OUT_OF_BOUNDS", 0) > outcomes.get("COLLISION", 0):
            priority = 0 if case.endswith(("_goal", "_world_rotation")) else 2
        elif outcomes.get("COLLISION", 0):
            priority = 0 if case.endswith(("_goal", "_task_only", "_horizon128")) else 2
        # Original sensor baselines are prerequisites for interpreting any treatment.
        baseline = parent is None and family in {"depth", "lidar"}
        candidates.append(
            ((active[modality] > 0, not baseline, -score, priority, active[modality], index), case)
        )
    return (min(candidates)[1], 0) if candidates else None


def trial_options(case, seed, output):
    """Resolve each recipe as a new run with one declared parameter treatment."""
    family, experiment, _ = CASES[case]
    treatments = []
    ancestor = case
    while ancestor is not None:
        treatments = CASE_OPTIONS.get(ancestor, []) + treatments
        ancestor = CASES[ancestor][2]
    options = [
        "mode=train",
        f"experiment={experiment}",
        f"output={output}",
        f"seed={seed}",
        "simulation.num_envs=16",
        "initial_checkpoint=null",
        "resume=null",
        *treatments,
    ]
    if NAVIGATION_SEARCH:
        overrides = dict(option.lstrip("+").split("=", 1) for option in options)
        horizon = int(
            overrides.get(
                "learning.options.horizon",
                {"depth": 32, "lidar": 32, "fan": 128, "visfly": 256}[family],
            )
        )
        settings = interaction_controls(
            family, num_envs=int(overrides["simulation.num_envs"]), horizon=horizon
        )
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


def full_screen_history(directory):
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
                "outcomes": dict(
                    sum(
                        (Counter(row.get("outcomes", {})) for row in report["reports"].values()),
                        Counter(),
                    )
                ),
            }
        )
    return sorted(evidence, key=lambda row: row["checkpoint_update"])


def full_screen_evidence(directory):
    """Return the best complete measured screen, leaving unknown scores unset."""
    return max(full_screen_history(directory), key=lambda row: row["value"], default=None)


def next_trial(records):
    """Prioritize independent confirmations, then one informed candidate per family."""
    stable = stable_modalities(records) if NAVIGATION_SEARCH else set()
    for case in CASES:
        if MODALITY[CASES[case][0]] in stable:
            continue
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
            directory = root / key
            identity = directory / "run.json"
            running = f"{unit_name(key)}.service" in units
            record = json.loads(identity.read_text()) if identity.exists() else {}
            status = record.get("status", "running" if running else "missing")
            actual_status = status
            if status == "interrupted" and (directory / "study_pruned.json").is_file():
                status = "pruned"
            if status == "running" and not running:
                status = "stale"
            screens = {
                row["checkpoint_update"]: row for row in previous.get(key, {}).get("screens", [])
            }
            screens.update(
                {row["checkpoint_update"]: row for row in full_screen_history(directory)}
            )
            screens = [screens[update] for update in sorted(screens)]
            candidate = max(screens, key=lambda row: row["value"], default=None)
            best = previous.get(key, {}).get("best")
            if candidate is not None and (best is None or candidate["value"] >= best["value"]):
                best = candidate
            records[key] = {
                "status": status,
                "run_status": actual_status,
                "best": best,
                "latest": screens[-1] if screens else None,
                "screens": screens,
                "updates": record.get("updates", record.get("last_update")),
                "interactions": record.get("interactions", record.get("last_interactions")),
                "configuration_sha256": record.get("config_sha256"),
                "source_sha256": record.get("source_sha256"),
                "directory": str(directory),
                "error": record.get("error"),
            }
    return records


def prune_lidar_candidate(case, record):
    """Shorten only stagnant native LiDAR pilots after 1,105,920 interactions."""
    if CASES[case][0] != "lidar" or record["status"] != "running":
        return False
    if not record.get("updates") or not record.get("interactions"):
        return False
    per_update = record["interactions"] / record["updates"]
    screens = record.get("screens", [])
    if not screens or screens[-1]["checkpoint_update"] * per_update < 1_105_920:
        return False
    best, stagnant, stop = None, 0, False
    settings = PatienceConfig(min_updates=737_280, patience=2, min_delta=0.005)
    for row in screens:
        best, stagnant, _, stop = update_patience(
            row["value"],
            int(row["checkpoint_update"] * per_update),
            best,
            stagnant,
            settings=settings,
        )
    return stop


def apply_search_pruning(records):
    """Request a graceful checkpointed stop; retain the original worker run record."""
    for key, record in records.items():
        case = key.rsplit("_s", 1)[0]
        marker = Path(record["directory"]) / "study_pruned.json"
        if marker.exists() or not prune_lidar_candidate(case, record):
            continue
        write_json(
            marker,
            {
                "policy": ALLOCATION_POLICY,
                "observed_at_utc": datetime.now(UTC).isoformat(),
                "reason": "two_stagnant_full_screens_after_lidar_minimum",
                "latest": record["latest"],
                "updates_at_request": record["updates"],
                "interactions_at_request": record["interactions"],
            },
        )
        try:
            subprocess.run(
                [
                    "systemctl",
                    "--user",
                    "kill",
                    "--signal=SIGINT",
                    "--kill-whom=all",
                    f"{unit_name(key)}.service",
                ],
                check=True,
                timeout=15,
            )
        except subprocess.SubprocessError:
            # A worker can finish between collection and signaling; allow a later retry.
            marker.rename(marker.with_name("study_prune_signal_failed.json"))


def sync_optim_study(study, directory, records):
    """Recover real completed outputs once, and persist running-trial evidence."""
    known = {trial.params["case_id"]: trial for trial in study.trials}
    for key, record in records.items():
        output = Path(record["directory"])
        source = output / "trial.json"
        if not source.exists():
            continue
        launch_record = json.loads(source.read_text())
        params = {
            "case_id": key,
            "family": launch_record["family"],
            "seed": launch_record["seed"],
            "overrides": launch_record["argv"][4:],
        }
        trial_dir = directory / key
        trial_dir.mkdir(exist_ok=True)
        path = trial_dir / "manifest.json"
        manifest = json.loads(path.read_text()) if path.exists() else {}
        manifest.update(
            {
                "parameters": params,
                "launch": launch_record,
                "output": str(output),
                "record": record,
                "status": record["status"],
                "best": record["best"],
                "observed_at_utc": datetime.now(UTC).isoformat(),
            }
        )
        write_json(path, manifest)
        trial = known.get(key)
        if record["status"] not in FINISHED or (trial is not None and trial.state != "running"):
            continue
        latest = record.get("latest")
        if record["status"] != "failed" and latest is None:
            continue  # An interrupted evaluation cannot provide a fabricated objective.
        trial = trial if trial is not None else study.ask(params)
        trial.intermediate = []
        for screen in record.get("screens", []):
            trial.report(screen["value"], screen["checkpoint_update"])
        if record["status"] == "failed":
            study.tell(trial, state="failed")
        elif record["status"] in {"pruned", "early_stopped"}:
            study.tell(trial, state="pruned")
        else:
            study.tell(trial, record["best"]["value"])


def launch(project, root, python, case, seed, *, study=None, optim_directory=None, selection=None):
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
        minimum_distance = (
            0.0 if case in {"depth_mixed", "depth_mixed_lowlr", "lidar_mixed"} else 80.0
        )
        if (
            resolved["simulation"]["procedural"]["min_start_goal_distance_m"] != minimum_distance
            or not resolved["benchmark"]["fixed_checkpoint_seeds"]
            or resolved["initial_checkpoint"] is not None
            or resolved["resume"] is not None
        ):
            raise ValueError(
                "Search changed declared starts, fixed-cohort or scratch-init contract"
            )
    out.mkdir()
    command = [str(python), "-u", "-m", "drone_playground.cli", *options]
    launch_record = {
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
        "allocation_policy": ALLOCATION_POLICY if NAVIGATION_SEARCH else None,
    }
    write_json(out / "trial.json", launch_record)
    trial = None
    if study is not None:
        params = {"case_id": key, "family": CASES[case][0], "seed": seed, "overrides": options}
        trial_dir = optim_directory / key
        trial_dir.mkdir(exist_ok=True)
        write_json(
            trial_dir / "manifest.json",
            {
                "parameters": params,
                "launch": launch_record,
                "cwd": str(project),
                "output": str(out),
                "selection_evidence": selection,
                "allocation_policy": ALLOCATION_POLICY,
                "status": "proposed",
                "observed_at_utc": datetime.now(UTC).isoformat(),
            },
        )
        trial = study.ask(params)
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
    try:
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
    except subprocess.SubprocessError as error:
        (out / "run.json").write_text(
            json.dumps(
                {
                    "status": "failed",
                    "error": f"Worker launch failed: {error}",
                }
            )
            + "\n"
        )
        if trial is not None:
            study.tell(trial, state="failed")
        raise


def main():
    """Run the measured finite study or print its current evidence snapshot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "status"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--suite", choices=("paper", "navigation"), default="paper")
    parser.add_argument("--max-parallel", type=int, default=2)
    parser.add_argument("--optim-directory", type=Path)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    if args.max_parallel < 1:
        raise ValueError("max-parallel must be positive")
    if args.suite == "navigation":
        configure_navigation_search()
    snapshot = args.root / "study.json"
    if args.mode == "status":
        previous = json.loads(snapshot.read_text()).get("trials", {}) if snapshot.exists() else {}
        print(json.dumps({"trials": collect(args.root, previous)}, indent=2))
        return
    study = None
    if NAVIGATION_SEARCH:
        if args.optim_directory is None:
            parser.error("Navigation run requires --optim-directory for ask/tell evidence")
        import optim_agent

        args.optim_directory.mkdir(parents=True, exist_ok=True)
        study = optim_agent.create_study(
            direction="maximize", storage=args.optim_directory / "study.json", seed=0
        )
    args.root.mkdir(parents=True, exist_ok=True)
    directory = args.root / ".scheduler"
    directory.mkdir(exist_ok=True)
    with (directory / "study.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = json.loads(snapshot.read_text()).get("trials", {}) if snapshot.exists() else {}
        if any(record.get("adopted") for record in previous.values()):
            raise ValueError(
                "Historical adopted-control studies are read-only; use a new result root"
            )
        while True:
            records = collect(args.root, previous)
            if study is not None:
                sync_optim_study(study, args.optim_directory, records)
                apply_search_pruning(records)
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
                "allocation_policy": ALLOCATION_POLICY if NAVIGATION_SEARCH else None,
                "max_parallel": args.max_parallel,
                "optim_directory": str(args.optim_directory) if study is not None else None,
            }
            candidate = next_trial(records)
            active = any(record["status"] == "running" for record in records.values())
            blocked = [
                key
                for key, record in records.items()
                if record["status"] in {"stale", "interrupted"}
            ]
            complete = candidate is None and not active
            status["status"] = (
                "blocked" if blocked and complete else "finished" if complete else "running"
            )
            status["interrupted_trials"] = blocked
            temporary = snapshot.with_suffix(".tmp")
            temporary.write_text(json.dumps(status, indent=2) + "\n")
            temporary.replace(snapshot)
            if complete:
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
                    launch(
                        project,
                        args.root,
                        args.python,
                        *candidate,
                        study=study,
                        optim_directory=args.optim_directory,
                        selection={
                            key: {field: row.get(field) for field in ("status", "best", "latest")}
                            for key, row in records.items()
                            if row["status"] != "missing"
                        },
                    )
                    print(datetime.now(UTC).isoformat(), "launched", candidate, flush=True)
            previous = records
            time.sleep(30)


if __name__ == "__main__":
    main()
