"""Opt-in MuJoCo frames for Autoresearch's existing Dashboard media protocol.

Training exports only one world's pose samples. A short-lived renderer process
owns OpenGL and JPEG encoding, so the JAX training loop never constructs a viewer.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


class TrainingPreview:
    """Submit bounded visual samples when a Dashboard user requests them."""

    def __init__(self):
        """Locate the current Dashboard session without starting any renderer."""
        location = os.environ.get("DRONE_PLAYGROUND_DASHBOARD_SESSION")
        self.session = Path(location).resolve() if location else None
        self.process = None
        self.last_capture = 0.0

    def capture(self, env_state, scene, samples, *, update: int, final: bool = False):
        """Render a requested preview or retain the final actual training rollout."""
        if self.session is None or not all(
            name in samples
            for name in ("_preview_position", "_preview_quaternion", "_preview_time")
        ):
            return
        control = self.session / "live/control.json"
        if not final and not control.is_file():
            return
        if final:
            mode = "sampled_trajectory"
        else:
            try:
                mode = json.loads(control.read_text()).get("visual_mode", "off")
            except (OSError, ValueError):
                return
        if mode not in {"live_frame", "sampled_trajectory"}:
            return
        if self.process is not None and self.process.poll() is None:
            if not final:
                return
            self.process.wait(timeout=45)
        if not final and time.monotonic() - self.last_capture < 8:
            return

        positions = np.asarray(samples["_preview_position"])
        quaternions = np.asarray(samples["_preview_quaternion"])
        times = np.asarray(samples["_preview_time"])
        current_time = float(np.asarray(env_state.time[0]))
        # The learner can reset a world at the end of a rollout. In that case
        # the new scene cannot be used to depict the previous episode.
        if (
            current_time + 1e-5 < float(times[-1])
            and samples.get("_preview_scene") is None
            and env_state.scene_state is not None
        ):
            positions = np.asarray(env_state.physics.states.pos[0, 0])[None]
            quaternions = np.asarray(env_state.physics.states.quat[0, 0])[None]
            times = np.asarray([current_time])

        scene_state = samples.get("_preview_scene")
        if scene_state is None:
            scene_state = env_state.scene_state
        payload = {
            "scene_name": np.asarray(scene.name),
            "update": np.asarray(update),
            "mode": np.asarray(mode),
            "final": np.asarray(final),
            "positions": positions,
            "quaternions": quaternions,
            "times": times,
        }
        if scene_state is not None:
            for key in (
                "centers",
                "sizes",
                "target",
                "speed",
                "phase",
                "active",
                "cylinder",
                "moving",
            ):
                payload[key] = np.asarray(getattr(scene_state, key)[0])
            if scene_state.motion is not None:
                payload["motion"] = np.asarray(scene_state.motion[0])
        visual_root = self.session / "live"
        visual_root.mkdir(parents=True, exist_ok=True)
        input_file = visual_root / f"preview_{'final' if final else 'input'}_{update:06d}.npz"
        np.savez_compressed(input_file, **payload)
        log = (visual_root / "preview-render.log").open("a")
        try:
            self.process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "drone_playground.simulation.training_visuals",
                    str(input_file),
                    str(self.session),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                env={**os.environ, "MUJOCO_GL": "egl", "JAX_PLATFORMS": "cpu"},
            )
        finally:
            log.close()
        self.last_capture = time.monotonic()
        if final:
            self.process.wait(timeout=60)


def _visual_model(sample):
    """Construct a native MuJoCo scene from its fixed MJCF or sampled primitives."""
    import xml.etree.ElementTree as ET
    from importlib import resources

    import mujoco

    from drone_playground.simulation.replay import _attach_drone_mesh, _scene_xml

    name = str(sample["scene_name"].item())
    procedural = name == "train/procedural"
    if procedural or name == "empty":
        root = ET.Element("mujoco", model="training-preview")
        assets = {}
        world = ET.SubElement(root, "worldbody")
    else:
        path = Path(name)
        if not path.is_file():
            base = resources.files("drone_playground").joinpath("assets/scenes")
            if not base.is_dir():
                base = Path(__file__).resolve().parents[3] / "assets/scenes"
            path = base.joinpath(
                "racing/lsy_level0.xml" if name == "racing" else f"navigation/{name}.xml"
            )
        root, assets = _scene_xml(Path(path))
        world = root.find("worldbody")
        if world is None:
            world = ET.SubElement(root, "worldbody")

    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    if visual.find("headlight") is None:
        ET.SubElement(
            visual,
            "headlight",
            ambient="0.5 0.5 0.5",
            diffuse="0.85 0.85 0.85",
            specular="0.15 0.15 0.15",
        )

    if procedural or name == "empty":
        ET.SubElement(
            world,
            "geom",
            name="preview_floor",
            type="plane",
            pos="50 0 0" if procedural else "0 0 0",
            size="110 45 .1" if procedural else "12 12 .1",
            rgba="0.16 0.2 0.23 1",
            contype="0",
            conaffinity="0",
        )
    movable = []
    if procedural:
        if "motion" in sample:
            ET.SubElement(root, "size", nuser_body="6")
        for index in np.flatnonzero(sample["active"]):
            body = ET.SubElement(world, "body", name=f"obstacle_{index}", mocap="true")
            if "motion" in sample:
                body.set("pos", " ".join(map(str, sample["centers"][index])))
                body.set("user", " ".join(map(str, sample["motion"][index])))
            cylinder = bool(sample["cylinder"][index])
            size = sample["sizes"][index, :2] if cylinder else sample["sizes"][index]
            ET.SubElement(
                body,
                "geom",
                type="cylinder" if cylinder else "box",
                size=" ".join(map(str, size)),
                rgba="0.87 0.42 0.21 1" if sample["moving"][index] else "0.3 0.55 0.66 1",
                contype="0",
                conaffinity="0",
            )
            movable.append(index)

    drone = ET.SubElement(world, "body", name="drone")
    ET.SubElement(drone, "freejoint", name="preview_drone_joint")
    ET.SubElement(
        drone,
        "geom",
        type="sphere",
        size="0.07",
        mass="0.027",
        rgba="0 0 0 0",
        contype="0",
        conaffinity="0",
    )
    _attach_drone_mesh(root, assets, drone)
    model = mujoco.MjModel.from_xml_string(ET.tostring(root).decode(), assets=assets)
    if model.nq != 7:
        raise ValueError("Training preview must contain exactly one free-joint drone")
    return model, movable, root, assets


def _render(sample, session):
    """Render the most recent contiguous trajectory in a separate EGL process."""
    import mujoco
    from PIL import Image

    positions = np.asarray(sample["positions"], dtype=float)
    quaternions = np.asarray(sample["quaternions"], dtype=float)
    times = np.asarray(sample["times"], dtype=float)
    if (
        positions.ndim != 2
        or positions.shape[1] != 3
        or quaternions.shape
        != (
            len(positions),
            4,
        )
        or times.shape != (len(positions),)
    ):
        raise ValueError("Training preview poses must have shapes [T,3], [T,4], and [T]")
    if not np.isfinite(positions).all() or not np.isfinite(quaternions).all():
        raise ValueError("Training preview has nonfinite poses")
    # Only the most recent episode fragment is continuous across resets.
    resets = np.flatnonzero(np.diff(times) < -1e-6)
    first = int(resets[-1]) + 1 if len(resets) else 0
    if str(sample["mode"].item()) == "live_frame":
        indices = [len(times) - 1]
    else:
        stride = max(1, (len(times) - first) // 16)
        indices = list(range(first, len(times), stride))[-16:]
        if indices[-1] != len(times) - 1:
            indices.append(len(times) - 1)

    model, obstacle_indices, model_xml, assets = _visual_model(sample)
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = 1.6 if str(sample["scene_name"].item()) != "train/procedural" else 5.0
    camera.azimuth = 135
    camera.elevation = -28
    root = session / "live"
    final = bool(sample["final"]) if "final" in sample else False
    sequence = (
        root
        / "trajectories"
        / ("last_training_rollout" if final else f"sample_{int(sample['update']):06d}")
    )
    if str(sample["mode"].item()) == "sampled_trajectory":
        sequence.mkdir(parents=True, exist_ok=True)
    images = []
    rendered = []
    browser_frames = []
    with mujoco.Renderer(model, width=640, height=360) as renderer:
        for index, row in enumerate(indices):
            data.time = float(times[row])
            data.qpos[:3] = positions[row]
            data.qpos[3:7] = quaternions[row, [3, 0, 1, 2]]
            if obstacle_indices and "motion" not in sample:
                centers = sample["centers"][obstacle_indices]
                targets = sample["target"][obstacle_indices]
                speed = sample["speed"][obstacle_indices]
                phase = sample["phase"][obstacle_indices]
                moving = sample["moving"][obstacle_indices]
                delta = targets - centers
                omega = speed / np.maximum(np.linalg.norm(delta, axis=-1), 0.2)
                current = centers + np.where(
                    moving[:, None], delta * np.sin(data.time * omega + phase)[:, None], 0
                )
                for slot, center in zip(obstacle_indices, current, strict=True):
                    body = model.body(f"obstacle_{slot}")
                    data.mocap_pos[int(body.mocapid[0])] = center
            elif model.nuser_body == 6:
                from drone_playground.simulation.replay import _motion

                mocap = data.mocap_pos[None, None].copy()
                _motion(model, np.asarray([data.time]), mocap)
                data.mocap_pos[:] = mocap[0, 0]
            mujoco.mj_kinematics(model, data)
            mujoco.mj_comPos(model, data)
            mujoco.mj_camlight(model, data)
            if final:
                browser_frames.append(
                    {
                        "time": float(data.time),
                        "qpos": np.asarray(data.qpos).round(8).tolist(),
                        "mocap_pos": np.asarray(data.mocap_pos).round(8).tolist(),
                        "mocap_quat": np.asarray(data.mocap_quat).round(8).tolist(),
                    }
                )
            camera.lookat[:] = positions[row]
            renderer.update_scene(data, camera=camera)
            # Render the sampled path as MuJoCo debug geometry, not a second
            # simulated flight or a separately generated trajectory.
            for before, after in zip(
                range(indices[0], row), range(indices[0] + 1, row + 1), strict=True
            ):
                if np.linalg.norm(positions[after] - positions[before]) < 1e-5:
                    continue
                if renderer.scene.ngeom >= renderer.scene.maxgeom:
                    break
                geom = renderer.scene.geoms[renderer.scene.ngeom]
                mujoco.mjv_connector(
                    geom,
                    mujoco.mjtGeom.mjGEOM_CAPSULE,
                    0.014,
                    positions[before],
                    positions[after],
                )
                geom.rgba[:] = [0.1, 0.75, 1.0, 0.9]
                renderer.scene.ngeom += 1
            image = Image.fromarray(renderer.render())
            rendered.append(image)
            if len(indices) > 1:
                path = sequence / f"frame_{index:04d}.jpg"
                image.save(path, "JPEG", quality=75)
                images.append(path)
            if row == indices[-1]:
                temporary = root / ".current_run_frame.jpg"
                image.save(temporary, "JPEG", quality=75)
                os.replace(temporary, root / "current_run_frame.jpg")

    def relative(path):
        root = Path.cwd().resolve()
        return str(path.relative_to(root)) if path.is_relative_to(root) else str(path)

    final_video = None
    browser_replay = None
    if final and rendered:
        video_path = root / "last_training_rollout.webp"
        temporary = root / ".last_training_rollout.webp"
        rendered[0].save(
            temporary,
            format="WEBP",
            save_all=True,
            append_images=rendered[1:],
            duration=125,
            loop=0,
            quality=78,
        )
        os.replace(temporary, video_path)
        final_video = relative(video_path)
        import xml.etree.ElementTree as ET

        from drone_playground.simulation.browser_replay import FORMAT

        browser_xml = root / "browser_scene.xml"
        browser_xml.write_bytes(ET.tostring(model_xml, encoding="utf-8", xml_declaration=True))
        for name, content in assets.items():
            if "/" in name or "\\" in name or name in {"browser_scene.xml", "browser_replay.json"}:
                raise ValueError("Browser MJCF assets require flat portable names")
            (root / name).write_bytes(content)
        path = root / "browser_replay.json"
        record = {
            "format": FORMAT,
            "model": browser_xml.name,
            "assets": sorted(assets),
            "source": "actual final Crazyflow training rollout fragment",
            "original_frame_count": len(times),
            "frames": browser_frames,
        }
        temporary = root / ".browser_replay.json"
        temporary.write_text(json.dumps(record, separators=(",", ":")) + "\n")
        os.replace(temporary, path)
        browser_replay = relative(path)

    status = {
        "mode": str(sample["mode"].item()),
        "sampled_status": "completed" if images else "waiting",
        "update": int(sample["update"]),
        "live_frame_path": relative(root / "current_run_frame.jpg"),
        "trajectory_manifest_path": None,
        "final_rollout_path": final_video,
        "browser_replay_path": browser_replay,
    }
    if images:
        manifest = {
            "sample_index": int(sample["update"]),
            "status": "completed",
            "frames": [relative(path) for path in images],
            "playback_fps": 8,
        }
        manifest_path = sequence / "manifest.json"
        temporary = sequence / ".manifest.json"
        temporary.write_text(json.dumps(manifest))
        os.replace(temporary, manifest_path)
        status["trajectory_manifest_path"] = relative(manifest_path)
    temporary = root / ".visual.json"
    temporary.write_text(json.dumps(status))
    os.replace(temporary, root / "visual.json")


if __name__ == "__main__":
    with np.load(sys.argv[1], allow_pickle=False) as sample:
        try:
            _render(sample, Path(sys.argv[2]).resolve())
        finally:
            Path(sys.argv[1]).unlink(missing_ok=True)
