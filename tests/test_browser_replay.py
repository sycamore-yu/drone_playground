"""Export recorded MuJoCo state to WASM without running fresh dynamics."""

import json

import numpy as np
import pytest

from drone_playground.simulation.browser_replay import FORMAT, export
from drone_playground.simulation.replay import export_replay


def test_trusted_native_record_produces_portable_browser_assets(tmp_path):
    """Render the same qpos, mocap and timestamps as the actual RScope rollout."""
    scene = tmp_path / "fixture.xml"
    scene.write_text(
        '<mujoco><worldbody><body name="moving" mocap="true">'
        '<geom size=".2 .4 .2" type="box"/></body></worldbody></mujoco>'
    )
    time = np.array([0.0, 0.06, 0.12, 0.18, 0.24])
    positions = np.column_stack((time, np.zeros_like(time), np.full(len(time), 1.5)))
    quats = np.tile([0, 0, 0, 1], (len(time), 1))
    moving = np.column_stack((time + 2, time * 3, np.full(len(time), 2.3)))
    path = export_replay(
        tmp_path / "recorded",
        scene,
        time,
        positions,
        quats,
        scene_body_positions={"moving": moving},
    )
    record = export(path.parent, max_frames=3)
    published = json.loads(record.read_text())
    assert published["format"] == FORMAT
    assert published["original_frame_count"] == 5
    assert len(published["frames"]) == 3
    np.testing.assert_array_equal(published["frames"][0]["qpos"][:3], positions[0])
    np.testing.assert_array_equal(published["frames"][-1]["qpos"][:3], positions[-1])
    np.testing.assert_allclose(published["frames"][-1]["mocap_pos"][0], moving[-1])
    assert (path.parent / published["model"]).is_file()


def test_missing_native_record_cannot_be_exported(tmp_path):
    """Input validation prevents silently generating empty browser recordings."""
    with pytest.raises(FileNotFoundError, match=r"scene\.xml"):
        export(tmp_path)
    with pytest.raises(ValueError, match="two frames"):
        export(tmp_path, max_frames=1)
