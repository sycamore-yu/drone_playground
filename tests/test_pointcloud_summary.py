"""The delivery summary verifies saved evidence without retraining the policy."""

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


def summary_module():
    path = Path(__file__).resolve().parents[1] / "scripts/tools/summarize_pointcloud.py"
    assert path.exists(), "A reusable final point-cloud delivery verifier is required"
    spec = importlib.util.spec_from_file_location("pointcloud_summary", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evidence_fixture(tmp_path, updates=50000):
    """Only exercises artifact integrity; no pickle execution or simulated training."""
    scenes = ["S01", "S02", "S03", "S06", "D01", "D02", "D03", "D06"]
    train = tmp_path / "experiments/training"
    evaluation = tmp_path / "experiments/evaluation"
    train.mkdir(parents=True)
    from drone_playground.composition import compose_method

    config = compose_method("paper/pointcloud_flight")
    save(train / "resolved-config.json", config)
    checkpoint = train / "selected.pkl"
    checkpoint.write_bytes(b"selected state: only checksum validation")
    save(
        checkpoint.with_suffix(".json"),
        {
            "family": "paper_pointcloud_gru",
            "sha256": digest(checkpoint),
            "parameter_sha256": "test-parameters",
            "updates": 1000,
            "config": config,
        },
    )
    final = train / "last.pkl"
    final.write_bytes(b"last training state: only checksum validation")
    save(
        final.with_suffix(".json"),
        {
            "family": "paper_pointcloud_gru",
            "sha256": digest(final),
            "parameter_sha256": "last-parameters",
            "updates": updates,
            "config": config,
        },
    )
    training_result = {
        "status": "completed" if updates == 50000 else "paused",
        "actual_updates": updates,
        "target_updates": 50000,
        "actual_steps": updates * 5120,
        "target_steps": 50000 * 5120,
        "full_budget_completed": updates == 50000,
        "checkpoint": str(final),
        "selected": {
            "checkpoint": str(checkpoint),
            "updates": 1000,
            "parameter_sha256": "test-parameters",
            "development_loss": 2.0,
        },
    }
    save(train / "result.json", training_result)
    episodes, cells, replays = [], {}, []
    for speed in (4, 6, 8):
        label = f"speed-{speed}"
        arrays = {
            "active": np.ones((2, 8), dtype=bool),
            "outcome": np.array([[0] * 8, [2] * 8], dtype=np.int32),
            "time": np.array([[0.1] * 8, [0.2] * 8]),
            "pos": np.zeros((2, 8, 3)),
        }
        archive = evaluation / "traces" / f"{label}.npz"
        archive.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(archive, **arrays)
        cell_rows = []
        for scene in scenes:
            row = {
                "scene_id": scene,
                "command_speed_m_s": float(speed),
                "outcome": "collision",
                "arrived": False,
                "collision": True,
                "out_of_bounds": False,
                "numerical_failure": False,
                "timeout": False,
                "steps": 2,
                "elapsed_s": 0.2,
            }
            cell_rows.append(row)
            replay = evaluation / "rollouts" / label / scene / "trace.mj_unroll"
            replay.parent.mkdir(parents=True)
            replay.write_bytes(f"replay for {scene} at {speed}".encode())
            (replay.parent / "scene.xml").write_text("<mujoco><worldbody/></mujoco>")
            (replay.parent / "rscope_meta.pkl").write_bytes(
                b"metadata fixture for file integrity only"
            )
            save(
                replay.parent / "readback-verification.json",
                {
                    "frames": 3,
                    "transitions": 2,
                    "positions_and_action_channels_match": True,
                    "model_xml_recompiled": True,
                    "replay_sha256": digest(replay),
                },
            )
            replays.append(
                {
                    "scene_id": scene,
                    "speed_m_s": float(speed),
                    "frames": 3,
                    "transitions": 2,
                    "path": str(replay.relative_to(evaluation)),
                    "sha256": digest(replay),
                    "readback_verified": True,
                    "initial_frame_included": True,
                }
            )
        episodes.extend(cell_rows)
        cells[label] = {"num_trials": 8, "episodes": cell_rows, "trace_sha256": digest(archive)}
    report = {
        "num_trials": 24,
        "arrived": 0,
        "collision": 24,
        "out_of_bounds": 0,
        "numerical_failure": 0,
        "timeout": 0,
        "success_rate": 0.0,
        "parameters_frozen": True,
        "parameter_sha256": "test-parameters",
        "checkpoint": str(checkpoint),
        "trained_updates": 1000,
        "training_budget_completed": updates == 50000,
        "training_run_evidence": {
            "run": str(train),
            "result_sha256": digest(train / "result.json"),
        },
        "scene_ids": scenes,
        "policy_hz": 10,
        "dynamics_transition_hz": 10,
        "collision_sampling_hz": 500,
        "duration_s": 40,
        "body_radius_m": 0.07,
        "goal_radius_m": 0.5,
        "dynamics": "point_mass_lag",
        "catalog_sha256": digest(
            Path(__file__).resolve().parents[1] / "assets/scenes/navigation/catalog.json"
        ),
        "episodes": episodes,
        "cells": cells,
    }
    save(evaluation / "result.json", {"status": "completed", "num_trials": 24})
    save(evaluation / "eval/report.json", report)
    save(evaluation / "rollouts/index.json", {"replays": replays})
    return train, evaluation


def test_final_summary_keeps_negative_results_and_earlier_selected_weight(tmp_path):
    module = summary_module()
    train, evaluation = evidence_fixture(tmp_path)
    result = module.verify_delivery(train, evaluation, require_complete=True)
    assert result["experiment_completed"]
    assert result["training"]["actual_updates"] == 50000
    assert result["selection"]["updates"] == 1000
    assert result["outcomes"]["collision"] == 24 and result["outcomes"]["arrived"] == 0
    assert result["verified_trace_count"] == 3 and result["verified_replay_count"] == 24


def test_stage_budget_is_rejected_as_final(tmp_path):
    module = summary_module()
    train, evaluation = evidence_fixture(tmp_path, updates=1000)
    with pytest.raises(ValueError, match="budget"):
        module.verify_delivery(train, evaluation, require_complete=True)
    result = module.verify_delivery(train, evaluation, require_complete=False)
    assert not result["experiment_completed"] and result["verified_replay_count"] == 24


@pytest.mark.parametrize(
    "damage", ["trace_bytes", "replay_bytes", "reported_outcome", "case_identity"]
)
def test_inconsistent_saved_evidence_is_rejected(tmp_path, damage):
    module = summary_module()
    train, evaluation = evidence_fixture(tmp_path)
    if damage == "trace_bytes":
        (evaluation / "traces/speed-4.npz").write_bytes(b"changed")
    elif damage == "replay_bytes":
        (evaluation / "rollouts/speed-4/S01/trace.mj_unroll").write_bytes(b"changed")
    else:
        path = evaluation / "eval/report.json"
        report = json.loads(path.read_text())
        if damage == "reported_outcome":
            report["episodes"][0]["outcome"] = "arrived"
        else:
            report["episodes"][0]["scene_id"] = "S02"
        save(path, report)
    with pytest.raises(ValueError):
        module.verify_delivery(train, evaluation, require_complete=True)


def test_summary_output_contains_actual_slots_and_refuses_source_overwrite(tmp_path):
    module = summary_module()
    train, evaluation = evidence_fixture(tmp_path)
    result = module.verify_delivery(train, evaluation, require_complete=True)
    destination = tmp_path / "summary"
    module.write_delivery(result, destination)
    assert b"\r" not in (destination / "episodes.csv").read_bytes()
    content = (destination / "report.md").read_text()
    assert "50000" in content and "1000" in content and "碰撞" in content
    slots = json.loads((destination / "module-slots.json").read_text())
    assert slots["algorithm"]["name"] == "pointcloud_bptt"
    assert slots["env"]["sensor"]["azimuth_count"] == 180
    with pytest.raises(FileExistsError):
        module.write_delivery(result, train)


def test_selected_checkpoint_update_count_is_checked_against_its_metadata(tmp_path):
    module = summary_module()
    train, evaluation = evidence_fixture(tmp_path)
    path = train / "selected.json"
    metadata = json.loads(path.read_text())
    metadata["updates"] = 900
    save(path, metadata)
    with pytest.raises(ValueError, match="checkpoint.*(age|update)"):
        module.verify_delivery(train, evaluation, require_complete=True)


def test_another_scene_catalog_cannot_be_reported_as_navigation8(tmp_path):
    module = summary_module()
    train, evaluation = evidence_fixture(tmp_path)
    path = evaluation / "eval/report.json"
    report = json.loads(path.read_text())
    report["catalog_sha256"] = "different-catalog"
    save(path, report)
    with pytest.raises(ValueError, match="catalog"):
        module.verify_delivery(train, evaluation, require_complete=True)


@pytest.mark.parametrize("filename", ["scene.xml", "rscope_meta.pkl"])
def test_missing_replay_companion_file_prevents_complete_delivery(tmp_path, filename):
    module = summary_module()
    train, evaluation = evidence_fixture(tmp_path)
    (evaluation / "rollouts/speed-4/S01" / filename).unlink()
    with pytest.raises(ValueError, match="companion"):
        module.verify_delivery(train, evaluation, require_complete=True)


def test_different_cases_cannot_share_one_replay_artifact(tmp_path):
    module = summary_module()
    train, evaluation = evidence_fixture(tmp_path)
    path = evaluation / "rollouts/index.json"
    index = json.loads(path.read_text())
    index["replays"][1]["path"] = index["replays"][0]["path"]
    index["replays"][1]["sha256"] = index["replays"][0]["sha256"]
    save(path, index)
    with pytest.raises(ValueError, match="(Replay path|replay path)"):
        module.verify_delivery(train, evaluation, require_complete=True)


def test_report_method_description_uses_actual_module_settings(tmp_path):
    module = summary_module()
    train, evaluation = evidence_fixture(tmp_path)
    path = train / "resolved-config.json"
    config = json.loads(path.read_text())
    config["algorithm"]["gradient"]["transition"] = "direct"
    config["env"]["sensor"]["azimuth_count"] = 90
    config["env"]["sensor"]["state_gradient"] = "detached"
    save(path, config)
    summary = module.verify_delivery(train, evaluation, require_complete=True)
    destination = tmp_path / "direct-summary"
    module.write_delivery(summary, destination)
    text = (destination / "report.md").read_text()
    assert "2700" in text and "直接状态导数" in text
    assert "5400" not in text and "指数衰减状态导数" not in text
