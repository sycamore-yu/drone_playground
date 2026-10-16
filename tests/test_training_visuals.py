"""Native MuJoCo training previews reuse Autoresearch's frame/manifest protocol."""

import json
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp

from drone_playground.simulation.training_visuals import TrainingPreview


def test_preview_only_spawns_on_explicit_dashboard_request(monkeypatch, tmp_path):
    """Training remains headless unless the user changes the Dashboard mode."""
    monkeypatch.setenv("DRONE_PLAYGROUND_DASHBOARD_SESSION", str(tmp_path))
    preview = TrainingPreview()
    preview.capture(None, None, {}, update=1)
    assert preview.process is None
    assert not (tmp_path / "live").exists()


def test_sampled_rollout_renders_native_procedural_scene(monkeypatch, tmp_path):
    """A real EGL subprocess writes an image and a browser-readable manifest."""
    monkeypatch.setenv("DRONE_PLAYGROUND_DASHBOARD_SESSION", str(tmp_path))
    live = tmp_path / "live"
    live.mkdir()
    (live / "control.json").write_text('{"visual_mode": "sampled_trajectory"}')
    centers = jnp.array([[[2, 2, 2], [3, -2, 2]]], dtype=jnp.float32)
    scene_state = SimpleNamespace(
        centers=centers,
        sizes=jnp.array([[[0.5, 1.0, 0.0], [0.4, 0.6, 0.3]]]),
        target=centers,
        speed=jnp.zeros((1, 2)),
        phase=jnp.zeros((1, 2)),
        active=jnp.array([[True, True]]),
        cylinder=jnp.array([[True, False]]),
        moving=jnp.array([[False, False]]),
    )
    state = SimpleNamespace(time=jnp.array([0.04]), scene_state=scene_state)
    poses = {
        "_preview_position": jnp.array([[0, 0, 1.5], [0.2, 0, 1.5]]),
        "_preview_quaternion": jnp.array([[0, 0, 0, 1], [0, 0, 0, 1]]),
        "_preview_time": jnp.array([0.02, 0.04]),
    }
    preview = TrainingPreview()
    preview.capture(state, SimpleNamespace(name="train/procedural"), poses, update=7)
    assert preview.process is not None
    assert preview.process.wait(timeout=60) == 0, (live / "preview-render.log").read_text()
    visual = json.loads((live / "visual.json").read_text())
    assert visual["mode"] == "sampled_trajectory" and visual["update"] == 7
    assert (live / "current_run_frame.jpg").stat().st_size > 1_000
    manifest = json.loads(Path(visual["trajectory_manifest_path"]).read_text())
    assert len(manifest["frames"]) == 2
    assert all(Path(frame).stat().st_size > 1_000 for frame in manifest["frames"])
    assert not list(live.glob("preview_input_*.npz"))
