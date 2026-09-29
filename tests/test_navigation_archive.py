"""All-case archive identity, episode boundaries and tamper detection."""

from types import SimpleNamespace

import numpy as np
import pytest

from drone_playground.evaluation.trace_archive import load_navigation_case, save_navigation_traces


@pytest.mark.parametrize('scenarios', [None, {'medium': [5, 0]}])
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
        assert scenario == (2 + case if scenarios is None else scenarios['medium'][case])
        np.testing.assert_array_equal(restored["pos"][:, 0], pos[:length, case])
        assert restored["obs"].shape == (length, 1, 20)
        assert restored["done"][-1, 0] == 1
        assert restored["outcome"][-1, 0] == 2 + case
    archive = tmp_path / "medium.npz"
    archive.write_bytes(archive.read_bytes() + b"unexpected")
    with pytest.raises(ValueError, match="digest mismatch"):
        load_navigation_case(tmp_path, "medium", 0)


def test_archive_rejects_noncontiguous_episode_frames(tmp_path):
    env = SimpleNamespace(bank=SimpleNamespace(num_instances=3))
    with pytest.raises(ValueError, match="contiguous"):
        save_navigation_traces(env, {"easy": {"active": np.array([[1], [0], [1]])}}, tmp_path)
