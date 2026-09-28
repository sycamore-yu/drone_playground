"""Continue the fixed paper budget, then evaluate the frozen policy on Navigation8.

This host-only coordinator starts independent training/evaluation processes. It
keeps a durable phase ledger, verifies existing results, and never reinitializes
a partially trained policy. The source tree is frozen for the lifetime of a run.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

TARGET_UPDATES = 50000
STEPS_PER_UPDATE = 32 * 160
SCENES = ("S01", "S02", "S03", "S06", "D01", "D02", "D03", "D06")
SPEEDS = (4.0, 6.0, 8.0)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def verify_checkpoint(path):
    path = Path(path)
    metadata = json.loads(path.with_suffix(".json").read_text())
    if metadata.get("family") != "paper_pointcloud_gru":
        raise ValueError("Unexpected checkpoint family")
    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["sha256"]:
        raise ValueError(f"Checkpoint digest mismatch: {path}")
    return metadata


def verify_training_result(run, expected_updates):
    run = Path(run)
    result = json.loads((run / "result.json").read_text())
    target = int(result["target_updates"])
    complete = expected_updates == target
    expected_status = "completed" if complete else "paused"
    if result.get("status") != expected_status:
        raise ValueError(f"Unexpected training status: {result.get('status')}")
    if (
        target != TARGET_UPDATES
        or result["actual_updates"] != expected_updates
        or result["actual_steps"] != expected_updates * STEPS_PER_UPDATE
        or result["target_steps"] != TARGET_UPDATES * STEPS_PER_UPDATE
        or bool(result["full_budget_completed"]) != complete
    ):
        raise ValueError("Training budget does not match the fixed paper recipe")
    last = verify_checkpoint(result["checkpoint"])
    selected = verify_checkpoint(result["selected"]["checkpoint"])
    if last["updates"] != expected_updates or selected["updates"] > expected_updates:
        raise ValueError("Checkpoint iteration is inconsistent with the training budget")
    return result


def build_commands(python, stage_result, full_run, early_run, final_run):
    prefix = [python, "-m", "drone_playground.app"]
    return {
        "early_evaluation": prefix
        + [
            "method=paper/pointcloud_flight",
            "env=paper/pointcloud_navigation",
            "mode=eval",
            f"checkpoint={stage_result['selected']['checkpoint']}",
            f"run_id={early_run}",
        ],
        "training": prefix
        + [
            "method=paper/pointcloud_flight",
            "mode=train",
            f"run_id={full_run}",
            f"training.resume={stage_result['checkpoint']}",
            "training.stop_after_updates=null",
        ],
        "final_evaluation_prefix": prefix
        + [
            "method=paper/pointcloud_flight",
            "env=paper/pointcloud_navigation",
            "mode=eval",
            f"run_id={final_run}",
        ],
    }


def verify_evaluation_result(run, *, expected_selected, expected_training_run):
    run = Path(run)
    result = json.loads((run / "result.json").read_text())
    report = json.loads((run / "eval/report.json").read_text())
    if result.get("status") != "completed" or not report.get("parameters_frozen"):
        raise ValueError("Evaluation is incomplete or altered policy parameters")
    if (
        Path(report["checkpoint"]).resolve() != Path(expected_selected["checkpoint"]).resolve()
        or report["parameter_sha256"] != expected_selected["parameter_sha256"]
    ):
        raise ValueError("Evaluation selected policy identity does not match this training phase")
    training_run = Path(expected_training_run).resolve()
    provenance = report.get("training_run_evidence") or {}
    if Path(provenance.get("run", "")).resolve() != training_run:
        raise ValueError("Evaluation training run identity does not match this pipeline")
    training_digest = hashlib.sha256((training_run / "result.json").read_bytes()).hexdigest()
    if provenance.get("result_sha256") != training_digest:
        raise ValueError("Evaluation training result identity does not match its current source")
    cells = {(row["scene_id"], float(row["command_speed_m_s"])) for row in report["episodes"]}
    expected = {(scene, speed) for scene in SCENES for speed in SPEEDS}
    if report["num_trials"] != 24 or len(report["episodes"]) != 24 or cells != expected:
        raise ValueError("Navigation8 denominator or cells are incomplete")
    if (
        sum(
            report[key]
            for key in ("arrived", "collision", "out_of_bounds", "numerical_failure", "timeout")
        )
        != 24
    ):
        raise ValueError("Terminal outcomes do not account for every trial")
    metadata = verify_checkpoint(report["checkpoint"])
    if metadata.get("parameter_sha256") != expected_selected["parameter_sha256"]:
        raise ValueError(
            "Checkpoint selected parameter identity does not match the evaluated policy"
        )
    return report


def source_digest(root):
    digest = hashlib.sha256()
    paths = [
        *root.glob("src/**/*.py"),
        *root.glob("configs/**/*.yaml"),
        root / "assets/scenes/navigation/catalog.json",
        Path(__file__).resolve(),
    ]
    for path in sorted(set(paths)):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def validate_source_reconciliation(changed_paths):
    """Allow explicit supervisor repair while preserving live training source semantics."""
    allowed = {
        "scripts/tools/run_pointcloud_pipeline.py",
        "src/drone_playground/evaluation/pointcloud.py",
    }
    unexpected = sorted(set(changed_paths) - allowed)
    if unexpected:
        raise ValueError(
            f"Source reconciliation would change live training semantics: {unexpected}"
        )


def process_alive(record):
    try:
        pid = int(record["pid"])
        ticks = Path(f"/proc/{pid}/stat").read_text().split()[21]
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        return record.get("start_marker") == f"{boot}:{pid}:{ticks}"
    except (KeyError, OSError, ValueError, IndexError):
        return False


def read_process_command(pid):
    return [
        part.decode() for part in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0") if part
    ]


def verify_live_phase(run, phase_record, expected_command):
    """Authenticate an existing child so coordinator recovery never restarts training."""
    run = Path(run)
    state = json.loads((run / "state.json").read_text())
    identity = state.get("process", {})
    if phase_record.get("pid") != identity.get("pid") or phase_record.get("run") != run.name:
        raise RuntimeError("Running phase process identity differs from the durable ledger")
    if phase_record.get("command") != expected_command:
        raise RuntimeError("Recovered phase command differs from the durable ledger")
    if not process_alive(identity):
        raise RuntimeError(
            "The recorded phase process is absent; recover a saved training state explicitly"
        )
    if read_process_command(identity["pid"]) != expected_command:
        raise RuntimeError("Running process command differs from the declared training phase")
    return identity


def wait_for_training_stage(run, ledger, ledger_path):
    run = Path(run)
    while not (run / "result.json").exists():
        state_path = run / "state.json"
        if not state_path.exists():
            raise FileNotFoundError(f"The declared first-stage run has not started: {run}")
        state = json.loads(state_path.read_text())
        if not process_alive(state.get("process", {})):
            raise RuntimeError(
                f"Stage process is absent; recover its saved state explicitly: {run}"
            )
        ledger.update(
            phase="waiting_for_stage1",
            updated_at=now(),
            stage1_step=state.get("step"),
            stage1_process=state.get("process"),
        )
        atomic_json(ledger_path, ledger)
        time.sleep(10)
    return verify_training_result(run, 1000)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--stage1-run", default="paper-pointcloud-seed0-stage1-v1")
    parser.add_argument("--full-run", default="paper-pointcloud-seed0-full-v1")
    parser.add_argument("--early-eval-run", default="paper-pointcloud-navigation8-stage1-v1")
    parser.add_argument("--final-eval-run", default="paper-pointcloud-navigation8-full-v1")
    parser.add_argument("--pipeline-run", default="paper-pointcloud-seed0-pipeline-v1")
    args = parser.parse_args()
    root = args.root.resolve()
    directory = root / "experiments" / args.pipeline_run
    directory.mkdir(parents=True, exist_ok=True)
    ledger_path = directory / "state.json"
    with (directory / "coordinator.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        digest = source_digest(root)
        run_identity = {
            "stage1": args.stage1_run,
            "full": args.full_run,
            "early_evaluation": args.early_eval_run,
            "final_evaluation": args.final_eval_run,
        }
        if ledger_path.exists():
            ledger = json.loads(ledger_path.read_text())
            if ledger["source_digest"] != digest:
                raise ValueError(
                    "Pipeline source changed; reconcile the saved protocol before resuming"
                )
            if ledger.get("run_identity") != run_identity:
                raise ValueError(
                    "Pipeline training/evaluation run identities changed during recovery"
                )
        else:
            ledger = dict(
                status="running",
                phase="starting",
                started_at=now(),
                source_digest=digest,
                source_commit=subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=root, text=True
                ).strip(),
                target_updates=TARGET_UPDATES,
                target_steps=TARGET_UPDATES * STEPS_PER_UPDATE,
                run_identity=run_identity,
                phases={},
            )
        env = os.environ.copy()
        env["PYTHONPATH"] = str(root / "src")
        env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        env["SCIPY_ARRAY_API"] = "1"
        env.setdefault("OMP_NUM_THREADS", "4")

        def phase(name, command, run_name, verify):
            if source_digest(root) != digest:
                raise ValueError("Frozen source changed during the pipeline")
            run = root / "experiments" / run_name
            if (run / "result.json").exists():
                report = verify(run)
                ledger["phases"][name] = dict(status="completed", run=run_name, verified_at=now())
                atomic_json(ledger_path, ledger)
                return report
            if run.exists():
                identity = verify_live_phase(run, ledger["phases"].get(name, {}), command)
                ledger.update(status="running", phase=name, updated_at=now())
                ledger["phases"][name].update(coordinator_adopted_at=now(), process=identity)
                while not (run / "result.json").exists():
                    if not process_alive(identity):
                        if (run / "result.json").exists():
                            break
                        # A successful child writes its result before exiting; an absent result
                        # therefore indicates a failed or interrupted phase, never permission to restart.
                        raise RuntimeError(
                            f"Adopted process ended without a complete result: {run}"
                        )
                    current = json.loads((run / "state.json").read_text())
                    ledger.update(
                        updated_at=now(),
                        child_progress=dict(
                            step=current.get("step"),
                            phase=current.get("phase"),
                            details=current.get("details"),
                        ),
                    )
                    atomic_json(ledger_path, ledger)
                    time.sleep(10)
                report = verify(run)
                ledger["phases"][name].update(status="completed", finished_at=now())
                atomic_json(ledger_path, ledger)
                return report
            ledger.update(status="running", phase=name, updated_at=now())
            with (directory / f"{name}.log").open("w") as output:
                process = subprocess.Popen(
                    command, cwd=root, env=env, stdout=output, stderr=subprocess.STDOUT
                )
                ledger["phases"][name] = dict(
                    status="running",
                    run=run_name,
                    command=command,
                    pid=process.pid,
                    started_at=now(),
                )
                atomic_json(ledger_path, ledger)
                while process.poll() is None:
                    ledger["updated_at"] = now()
                    child_state = run / "state.json"
                    if child_state.exists():
                        current = json.loads(child_state.read_text())
                        ledger["child_progress"] = dict(
                            step=current.get("step"),
                            phase=current.get("phase"),
                            details=current.get("details"),
                        )
                    atomic_json(ledger_path, ledger)
                    time.sleep(10)
                if process.returncode != 0:
                    raise subprocess.CalledProcessError(process.returncode, command)
            report = verify(run)
            ledger["phases"][name].update(status="completed", finished_at=now())
            atomic_json(ledger_path, ledger)
            return report

        try:
            first = wait_for_training_stage(
                root / "experiments" / args.stage1_run, ledger, ledger_path
            )
            commands = build_commands(
                sys.executable, first, args.full_run, args.early_eval_run, args.final_eval_run
            )
            commands["early_evaluation"].append(
                f"evaluation.training_run={root / 'experiments' / args.stage1_run}"
            )
            early = phase(
                "stage1_evaluation",
                commands["early_evaluation"],
                args.early_eval_run,
                lambda run: verify_evaluation_result(
                    run,
                    expected_selected=first["selected"],
                    expected_training_run=root / "experiments" / args.stage1_run,
                ),
            )
            full = phase(
                "full_training",
                commands["training"],
                args.full_run,
                lambda run: verify_training_result(run, TARGET_UPDATES),
            )
            final_command = commands["final_evaluation_prefix"] + [
                f"checkpoint={full['selected']['checkpoint']}",
                f"evaluation.training_run={root / 'experiments' / args.full_run}",
            ]
            final = phase(
                "final_evaluation",
                final_command,
                args.final_eval_run,
                lambda run: verify_evaluation_result(
                    run,
                    expected_selected=full["selected"],
                    expected_training_run=root / "experiments" / args.full_run,
                ),
            )
            if not final["training_budget_completed"]:
                raise ValueError("Final evaluation did not authenticate the completed training run")
            ledger.update(
                status="completed",
                phase="completed",
                finished_at=now(),
                actual_training_updates=full["actual_updates"],
                early_arrived=early["arrived"],
                final_arrived=final["arrived"],
                final_trials=final["num_trials"],
                final_selected_updates=final["trained_updates"],
                full_training_run=args.full_run,
                final_evaluation_run=args.final_eval_run,
            )
            atomic_json(directory / "result.json", ledger)
        except BaseException as error:
            ledger.update(status="failed", updated_at=now(), error=repr(error))
            raise
        finally:
            atomic_json(ledger_path, ledger)


if __name__ == "__main__":
    main()
