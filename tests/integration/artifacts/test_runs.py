"""Integration tests for run recording and rscope export."""

from __future__ import annotations

import json
import struct
import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
from tensorboardX.proto import event_pb2

from drone_playground.artifacts import RunRecorder
from drone_playground.artifacts.layout import find_experiment
from drone_playground.visualization.rscope_io import export_rollout
from drone_playground.visualization.rscope_publish import publish_run


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _init_dirty_repo(root: Path) -> str:
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "tests@example.invalid")
    _git(root, "config", "user.name", "Drone Playground Tests")
    source = root / "tracked.txt"
    source.write_text("before\n")
    _git(root, "add", "tracked.txt")
    _git(root, "commit", "-qm", "fixture")
    commit = _git(root, "rev-parse", "HEAD")
    source.write_text("after\n")
    (root / "untracked.py").write_text("VALUE = 3\n")
    return commit


def _read_event_scalars(event_file: Path) -> list[tuple[int, str, float]]:
    scalars: list[tuple[int, str, float]] = []
    with event_file.open("rb") as handle:
        while header := handle.read(8):
            assert len(header) == 8
            size = struct.unpack("<Q", header)[0]
            assert len(handle.read(4)) == 4
            payload = handle.read(size)
            assert len(payload) == size
            assert len(handle.read(4)) == 4
            event = event_pb2.Event.FromString(payload)
            if not event.HasField("summary"):
                continue
            for value in event.summary.value:
                scalars.append((event.step, value.tag, value.simple_value))
    return scalars


def _trace(sim, frames: int = 4) -> dict:
    pos0 = np.asarray(sim.data.states.pos[0, 0], dtype=np.float64)
    quat0 = np.asarray(sim.data.states.quat[0, 0], dtype=np.float64)
    pos = np.repeat(pos0[None, None, :], frames, axis=0)
    pos[:, 0, 0] += np.arange(frames) * 0.01
    quat = np.repeat(quat0[None, None, :], frames, axis=0)
    time_values = np.arange(1, frames + 1, dtype=np.float64)[:, None] / sim.freq
    return {
        "pos": pos,
        "quat": quat,
        "time": time_values,
        "obs": np.arange(frames * 5, dtype=np.float64).reshape(frames, 1, 5),
        "reward": np.linspace(0.0, 1.0, frames, dtype=np.float64)[:, None],
        "metrics": {"tracking_error": np.linspace(0.2, 0.1, frames)[:, None]},
        "actions": np.arange(frames * 4, dtype=np.float64).reshape(frames, 1, 4),
    }


def test_run_recorder_writes_manifest_metrics_state_and_result(tmp_path: Path) -> None:
    commit = _init_dirty_repo(tmp_path)
    config = {"seed": 17, "task": {"name": "figure_eight", "freq": 50}}

    recorder = RunRecorder(tmp_path, "run-a", config)
    assert recorder.path == tmp_path / "results/runs/figure_eight/run/run-a"
    recorder.log(7, {"tracking_error": 0.125, "record_overhead_s": 0.002})
    recorder.phase("evaluating", step=7, episodes=2)
    recorder.finish("completed", accepted=True, steps=7)

    manifest = json.loads((recorder.path / "manifest.json").read_text())
    assert manifest["run_id"] == "run-a"
    assert manifest["task_id"] == "01"
    assert manifest["config"] == config
    assert manifest["code"]["commit"] == commit
    assert manifest["code"]["dirty"] is True
    assert manifest["process"]["pid"] > 0
    assert manifest["process"]["start_marker"]
    assert manifest["dependencies"]
    code_patch = (recorder.path / manifest["code"]["patch_path"]).read_text()
    assert "tracked.txt" in code_patch
    assert "untracked.py" in code_patch
    assert "VALUE = 3" in code_patch
    assert (recorder.path / "resolved-config.json").exists()
    assert (recorder.path / "command.txt").read_text().strip()

    state = json.loads((recorder.path / "state.json").read_text())
    assert state["phase"] == "evaluating"
    assert state["step"] == 7
    assert state["details"] == {"episodes": 2}
    assert state["status"] == "completed"

    metric_rows = [
        json.loads(line)
        for line in (recorder.path / "metrics" / "metrics.jsonl").read_text().splitlines()
    ]
    assert metric_rows == [
        {
            "step": 7,
            "metrics": {"tracking_error": 0.125, "record_overhead_s": 0.002},
        }
    ]

    event_files = list((recorder.path / "metrics").glob("events.out.tfevents.*"))
    assert len(event_files) == 1
    scalars = _read_event_scalars(event_files[0])
    assert (7, "tracking_error", pytest.approx(0.125)) in scalars
    assert (7, "record_overhead_s", pytest.approx(0.002)) in scalars

    result = json.loads((recorder.path / "result.json").read_text())
    assert result["status"] == "completed"
    assert result["accepted"] is True
    assert result["steps"] == 7

    with pytest.raises(FileExistsError):
        RunRecorder(tmp_path, "run-a", config)


def test_run_recorder_heartbeat_and_exception_result(tmp_path: Path, monkeypatch) -> None:
    import drone_playground.artifacts.record as record_module

    monkeypatch.setattr(record_module, "_HEARTBEAT_INTERVAL_SECONDS", 0.02)
    recorder = RunRecorder(tmp_path, "heartbeat", {"seed": 1})
    initial = json.loads((recorder.path / "state.json").read_text())
    deadline = time.monotonic() + 1.0
    later = initial
    while later["heartbeat_count"] <= initial["heartbeat_count"] and time.monotonic() < deadline:
        time.sleep(0.01)
        later = json.loads((recorder.path / "state.json").read_text())
    assert later["heartbeat_count"] > initial["heartbeat_count"]
    recorder.finish("cancelled", reason="test")

    with pytest.raises(ValueError, match=r"boom"), RunRecorder(tmp_path, "failed", {"seed": 2}):
        raise ValueError("boom")
    failed = json.loads((find_experiment(tmp_path, "failed") / "result.json").read_text())
    assert failed["status"] == "failed"
    assert failed["exception"]["type"] == "ValueError"
    assert failed["exception"]["message"] == "boom"
    assert "ValueError: boom" in failed["exception"]["traceback"]


def test_rscope_export_native_roundtrip_and_safe_publish(tmp_path: Path) -> None:
    from crazyflow.envs import FigureEightEnv
    from rscope import config as rscope_config
    from rscope import model_loader, rollout

    env = FigureEightEnv(num_envs=1, freq=50, dynamics="so_rpy", device="cpu")
    env.sim.reset()
    sim = env.sim
    original_config = (
        rscope_config.BASE_PATH,
        rscope_config.TEMP_PATH,
        rscope_config.META_PATH,
    )
    first = tmp_path / "first"
    second = tmp_path / "second"
    active = tmp_path / "active"
    try:
        first_unroll = export_rollout(sim, first, _trace(sim))
        assert first_unroll.suffix == ".mj_unroll"
        assert first_unroll.parent == first
        assert first_unroll.exists()
        assert (first / "rscope_meta.pkl").exists()
        xml_path = first / "scene.xml"
        assert xml_path.exists()
        xml = ET.fromstring(xml_path.read_text())
        for mesh in xml.findall("./asset/mesh"):
            mesh_file = mesh.get("file")
            assert mesh_file
            assert (first / mesh_file).is_file()

        assert original_config == (
            rscope_config.BASE_PATH,
            rscope_config.TEMP_PATH,
            rscope_config.META_PATH,
        )

        rollout.rollouts.clear()
        rollout.num_evals = 0
        rollout.append_unroll(first_unroll)
        record = rollout.rollouts[-1]
        assert np.allclose(record.mocap_pos[:, 0, 0], _trace(sim)["pos"][:, 0])
        assert np.allclose(
            record.metrics["tracking_error"], _trace(sim)["metrics"]["tracking_error"]
        )
        assert np.allclose(record.metrics["action/0"], _trace(sim)["actions"][:, :, 0])

        rscope_config.BASE_PATH = first
        rscope_config.TEMP_PATH = first / ".native-temp"
        rscope_config.META_PATH = first / "rscope_meta.pkl"
        loaded_model, loaded_data, _ = model_loader.load_model_and_data(False)
        assert loaded_model.nq == sim.mj_model.nq
        assert loaded_model.nmocap == sim.mj_model.nmocap
        dummy_src = sim.mj_model.body("_dummy")
        dummy_loaded = loaded_model.body("_dummy")
        assert float(dummy_loaded.mass[0]) == pytest.approx(float(dummy_src.mass[0]))
        assert np.allclose(dummy_loaded.inertia, dummy_src.inertia)
        loaded_data.mocap_pos[:] = record.mocap_pos[-1, 0]

        second_unroll = export_rollout(sim, second, _trace(sim, frames=3))
        publish_run(first, active)
        assert (active / "rscope_meta.pkl").exists()
        assert list(active.glob("*.mj_unroll"))

        unknown = active / "keep-me.txt"
        unknown.write_text("unmanaged")
        with pytest.raises(RuntimeError, match=r"unknown"):
            publish_run(second, active)
        assert unknown.read_text() == "unmanaged"
        assert first_unroll.exists()
        assert second_unroll.exists()

        unknown.unlink()
        publish_run(second, active)
        assert first_unroll.exists()
        assert second_unroll.exists()
        assert len(list(active.glob("*.mj_unroll"))) == 1
    finally:
        rscope_config.BASE_PATH, rscope_config.TEMP_PATH, rscope_config.META_PATH = original_config
        rollout.rollouts.clear()
        env.close()
