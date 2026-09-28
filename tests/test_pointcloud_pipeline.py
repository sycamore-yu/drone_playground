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
