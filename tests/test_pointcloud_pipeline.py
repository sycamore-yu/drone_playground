"""Durable paper pipeline phase selection and complete-budget accounting."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


def pipeline_module():
    path = Path(__file__).resolve().parents[1] / "scripts/run_pointcloud_pipeline.py"
    assert path.exists(), "The full-budget training/evaluation pipeline must exist"
    spec = importlib.util.spec_from_file_location("paper_pipeline", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture_stage(tmp_path, updates=1000):
    run = tmp_path / "experiments/stage1"
    run.mkdir(parents=True)
    checkpoint = run / "state.pkl"
    checkpoint.write_bytes(b"fixed-checkpoint")
    metadata = {
        "family": "paper_pointcloud_gru",
        "updates": updates,
        "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    }
    checkpoint.with_suffix(".json").write_text(json.dumps(metadata))
    result = {
        "status": "paused",
        "actual_updates": updates,
        "target_updates": 50000,
        "actual_steps": updates * 32 * 160,
        "target_steps": 50000 * 32 * 160,
        "full_budget_completed": False,
        "checkpoint": str(checkpoint),
        "selected": {"checkpoint": str(checkpoint), "updates": updates},
    }
    (run / "result.json").write_text(json.dumps(result))
    return run, result


def test_stage_result_requires_real_budget_and_authenticated_checkpoint(tmp_path):
    module = pipeline_module()
    run, result = fixture_stage(tmp_path)
    assert module.verify_training_result(run, expected_updates=1000)["actual_steps"] == 5120000
    result["actual_steps"] = 256000000
    (run / "result.json").write_text(json.dumps(result))
    with pytest.raises(ValueError, match="budget"):
        module.verify_training_result(run, expected_updates=1000)


def test_failed_training_is_never_advanced_to_evaluation(tmp_path):
    module = pipeline_module()
    run, result = fixture_stage(tmp_path)
    result["status"] = "failed"
    (run / "result.json").write_text(json.dumps(result))
    with pytest.raises(ValueError, match="status"):
        module.verify_training_result(run, expected_updates=1000)


def test_checkpoint_tamper_is_detected_before_continuation(tmp_path):
    module = pipeline_module()
    run, result = fixture_stage(tmp_path)
    Path(result["checkpoint"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="digest"):
        module.verify_training_result(run, expected_updates=1000)


def test_phase_commands_resume_final_state_and_evaluate_selected_policy(tmp_path):
    module = pipeline_module()
    _, result = fixture_stage(tmp_path)
    result["selected"]["checkpoint"] = str(tmp_path / "best.pkl")
    commands = module.build_commands("python3", result, "full", "early-eval", "full-eval")
    assert f"training.resume={result['checkpoint']}" in commands["training"]
    assert f"checkpoint={result['selected']['checkpoint']}" in commands["early_evaluation"]
    assert "training.stop_after_updates=null" in commands["training"]
    assert "experiment=paper_pointcloud_navigation8" in commands["early_evaluation"]
    assert commands["final_evaluation_prefix"][-1] == "run_id=full-eval"


def fixture_evaluation(tmp_path):
    module = pipeline_module()
    training_run, training = fixture_stage(tmp_path)
    selected = {**training["selected"], "parameter_sha256": "parameter-digest"}
    training["selected"] = selected
    (training_run / "result.json").write_text(json.dumps(training))
    metadata_path = Path(selected["checkpoint"]).with_suffix(".json")
    metadata = json.loads(metadata_path.read_text())
    metadata["parameter_sha256"] = selected["parameter_sha256"]
    metadata_path.write_text(json.dumps(metadata))
    evaluation_run = tmp_path / "evaluation"
    (evaluation_run / "eval").mkdir(parents=True)
    (evaluation_run / "result.json").write_text(json.dumps({"status": "completed"}))
    report = {
        "parameters_frozen": True,
        "num_trials": 24,
        "episodes": [
            {"scene_id": scene, "command_speed_m_s": speed}
            for scene in module.SCENES
            for speed in module.SPEEDS
        ],
        "arrived": 0,
        "collision": 0,
        "out_of_bounds": 0,
        "numerical_failure": 0,
        "timeout": 24,
        "checkpoint": selected["checkpoint"],
        "parameter_sha256": selected["parameter_sha256"],
        "training_run_evidence": {
            "run": str(training_run.resolve()),
            "result_sha256": hashlib.sha256(
                (training_run / "result.json").read_bytes()
            ).hexdigest(),
        },
    }
    (evaluation_run / "eval/report.json").write_text(json.dumps(report))
    return module, evaluation_run, training_run, selected


def test_evaluation_reuse_authenticates_the_expected_training_and_selected_policy(tmp_path):
    module, evaluation_run, training_run, selected = fixture_evaluation(tmp_path)
    report = module.verify_evaluation_result(
        evaluation_run, expected_selected=selected, expected_training_run=training_run
    )
    assert report["num_trials"] == 24


@pytest.mark.parametrize("mismatch", ["checkpoint", "parameters", "training_run"])
def test_completed_evaluation_from_another_training_identity_is_rejected(tmp_path, mismatch):
    module, evaluation_run, training_run, selected = fixture_evaluation(tmp_path)
    if mismatch == "checkpoint":
        selected["checkpoint"] = str(tmp_path / "another.pkl")
    elif mismatch == "parameters":
        selected["parameter_sha256"] = "another-parameter-digest"
    else:
        training_run = tmp_path / "another_training"
    with pytest.raises(ValueError, match="(selected|training).*(identity|match)"):
        module.verify_evaluation_result(
            evaluation_run, expected_selected=selected, expected_training_run=training_run
        )


def test_source_reconciliation_is_limited_to_evaluation_and_coordinator():
    module = pipeline_module()
    assert hasattr(module, "validate_source_reconciliation")
    module.validate_source_reconciliation(
        ["src/drone_playground/evaluation/pointcloud.py", "scripts/run_pointcloud_pipeline.py"]
    )
    with pytest.raises(ValueError, match="training"):
        module.validate_source_reconciliation(["src/drone_playground/learning/pointcloud_bptt.py"])


def test_live_training_adoption_checks_saved_process_identity(tmp_path):
    import os

    module = pipeline_module()
    assert hasattr(module, "verify_live_phase")
    run = tmp_path / "existing-training"
    run.mkdir()
    pid = os.getpid()
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    ticks = Path(f"/proc/{pid}/stat").read_text().split()[21]
    process = {"pid": pid, "start_marker": f"{boot}:{pid}:{ticks}"}
    (run / "state.json").write_text(
        json.dumps({"run_id": run.name, "status": "running", "process": process})
    )
    command = [
        part.decode() for part in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0") if part
    ]
    phase = {"run": run.name, "pid": pid, "command": command}
    assert module.verify_live_phase(run, phase, command) == process
    with pytest.raises(RuntimeError, match="identity"):
        module.verify_live_phase(run, {**phase, "pid": pid + 1}, command)
    with pytest.raises(RuntimeError, match="command"):
        module.verify_live_phase(run, phase, command + ["training.seed=1"])


def test_coordinator_recovery_adopts_the_authenticated_existing_process(tmp_path, monkeypatch):
    module = pipeline_module()
    assert hasattr(module, "verify_live_phase"), (
        "Recovery must preserve the running training process"
    )
    run = tmp_path / "full-run"
    run.mkdir()
    identity = {"pid": 12345, "start_marker": "boot:12345:99"}
    (run / "state.json").write_text(json.dumps({"process": identity, "phase": "training"}))
    command = ["python", "-m", "drone_playground.app", "run_id=full-run"]
    phase = {"pid": 12345, "run": "full-run", "command": command}
    monkeypatch.setattr(module, "process_alive", lambda p: p == identity)
    monkeypatch.setattr(module, "read_process_command", lambda pid: command)
    assert module.verify_live_phase(run, phase, command) == identity


@pytest.mark.parametrize("change", ["pid", "command", "missing_process"])
def test_coordinator_never_adopts_a_different_or_dead_process(tmp_path, monkeypatch, change):
    module = pipeline_module()
    assert hasattr(module, "verify_live_phase"), "Recovery must authenticate its process"
    run = tmp_path / "full-run"
    run.mkdir()
    identity = {"pid": 12345, "start_marker": "boot:12345:99"}
    (run / "state.json").write_text(json.dumps({"process": identity, "phase": "training"}))
    command = ["python", "-m", "drone_playground.app", "run_id=full-run"]
    phase = {"pid": 999 if change == "pid" else 12345, "run": "full-run", "command": command}
    monkeypatch.setattr(module, "process_alive", lambda p: change != "missing_process")
    monkeypatch.setattr(
        module,
        "read_process_command",
        lambda pid: ["unrelated"] if change == "command" else command,
    )
    with pytest.raises(RuntimeError, match="(identity|command|absent)"):
        module.verify_live_phase(run, phase, command)
