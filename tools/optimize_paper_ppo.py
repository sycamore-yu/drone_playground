"""Bounded measured search for paper-derived PPO, using real fixed-screen evidence.

Four seed-zero candidates maximum. Existing v04 baselines are adopted, never
restarted. Only an accepted frozen benchmark promotes a recipe to seeds 1/2.
This implements the optim-agent sequential protocol without requiring its package.
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
FINISHED = {"early_stopped", "accepted", "frozen_benchmark_failed", "failed", "pruned"}


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
    return BASE_UNITS.get(identifier, f"dp-paper-study-{identifier.replace('_', '-')}")


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
            }
    return records


def launch(project, root, python, case, seed):
    """Validate Hydra, reserve a new run, and launch without CPU/memory/swap quotas."""
    key = f"{case}_s{seed}"
    out = root / key
    if out.exists():
        raise FileExistsError(f"Never overwrite a previous trial: {out}")
    options = [
        "mode=train",
        f"experiment={CASES[case][1]}",
        f"output={out}",
        f"seed={seed}",
        "simulation.num_envs=16",
        "initial_checkpoint=null",
        "resume=null",
    ]
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
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
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
                "pilot_budget": 4,
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
            if candidate is not None and training_process_count() < 2:
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
                if (
                    memory["MemAvailable"] >= 8 * 1024 * 1024
                    and int(free.strip().splitlines()[0]) >= 7000
                ):
                    launch(project, args.root, args.python, *candidate)
                    print(datetime.now(UTC).isoformat(), "launched", candidate, flush=True)
            previous = records
            time.sleep(30)


if __name__ == "__main__":
    main()
