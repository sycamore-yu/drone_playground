"""All-case archive identity, episode boundaries and tamper detection."""

from types import SimpleNamespace

import numpy as np
import pytest

from drone_playground.artifacts.traces import load_navigation_case, save_navigation_traces


@pytest.mark.parametrize("scenarios", [None, {"medium": [5, 0]}])
def test_archive_round_trip_keeps_every_case_and_terminal_frame(tmp_path, scenarios):
    env = SimpleNamespace(bank=SimpleNamespace(num_instances=6))
    pos = np.arange(18, dtype=np.float32).reshape(3, 2, 3)
    trace = dict(
        pos=pos,
        quat=np.zeros((3, 2, 4)),
        time=np.full((3, 2), 0.02),
        actions=np.zeros((3, 2, 4)),
        obs=np.ones((3, 2, 2420)),
        reward=np.arange(6).reshape(3, 2),
        done=np.array([[1, 0], [1, 1], [1, 1]]),
        outcome=np.array([[2, 0], [2, 3], [2, 3]]),
        active=np.array([[1, 1], [0, 1], [0, 0]], bool),
        metrics={"clearance": np.full((3, 2), 0.2)},
    )
    index = save_navigation_traces(env, {"medium": trace}, tmp_path, scenarios)
    assert index["cells"]["medium"]["episode_steps"] == [1, 2]
    for case, length in enumerate((1, 2)):
        restored, scenario = load_navigation_case(tmp_path, "medium", case)
        assert scenario == (2 + case if scenarios is None else scenarios["medium"][case])
        np.testing.assert_array_equal(restored["pos"][:, 0], pos[:length, case])
        assert restored["obs"].shape == (length, 1, 20)
        assert restored["done"][-1, 0] == 1
        assert restored["outcome"][-1, 0] == 2 + case
    archive = tmp_path / "medium.npz"
    archive.write_bytes(archive.read_bytes() + b"unexpected")
    with pytest.raises(ValueError, match=r"digest mismatch"):
        load_navigation_case(tmp_path, "medium", 0)


def test_archive_rejects_noncontiguous_episode_frames(tmp_path):
    env = SimpleNamespace(bank=SimpleNamespace(num_instances=3))
    with pytest.raises(ValueError, match=r"contiguous"):
        save_navigation_traces(env, {"easy": {"active": np.array([[1], [0], [1]])}}, tmp_path)


def test_native_case_survives_a_later_worker_failure_and_keeps_partial_frames(tmp_path):
    import json

    from drone_playground.artifacts.traces import record_native_case

    env = SimpleNamespace(bank=SimpleNamespace(num_instances=8))
    identity = dict(difficulty="medium", case=5, scenario_id=1, seed=30041)
    row = dict(
        pos=np.array([1.0, 2.0, 3.0]),
        quat=np.array([0.0, 0.0, 0.0, 1.0]),
        time=0.02,
        actions=np.zeros(4),
        obs=np.arange(20),
        reward=1.0,
        done=0.0,
        outcome=0,
        active=True,
        metrics=dict(clearance=0.2),
    )
    first = tmp_path / "complete"
    with record_native_case(env, first, identity) as rows:
        rows.append(dict(row, done=1.0, outcome=1))
    before = (first / "case-record.json").read_bytes()
    broken = tmp_path / "interrupted"
    with (
        pytest.raises(RuntimeError, match=r"RPC"),
        record_native_case(env, broken, dict(identity, case=6, seed=30049)) as rows,
    ):
        rows.append(row)
        raise RuntimeError("RPC process died")
    assert (first / "case-record.json").read_bytes() == before
    record = json.loads((broken / "case-record.json").read_text())
    assert not record["completed_execution"] and record["recorded_frames"] == 1
    assert record["identity"]["seed"] == 30049 and "RPC" in record["error"]
    trace, scenario = load_navigation_case(broken / "case-trace", "medium", 0)
    np.testing.assert_array_equal(trace["pos"][0, 0], row["pos"])
    assert scenario == 1 and trace["outcome"][-1, 0] == 0  # Not relabeled as arrival/collision.
    startup = tmp_path / "startup-failure"
    with pytest.raises(TimeoutError), record_native_case(env, startup, dict(identity, case=7)):
        raise TimeoutError("Sensor subscribers not ready")
    record = json.loads((startup / "case-record.json").read_text())
    assert record["recorded_frames"] == 0 and record["archive"] is None
    assert not record["completed_execution"]
