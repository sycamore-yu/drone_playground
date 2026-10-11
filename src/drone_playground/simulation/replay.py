"""Export recorded flight poses and overlays in the RScope 0.0.8 format."""

from __future__ import annotations

import hashlib
import itertools
import pickle
import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING

import mujoco
import numpy as np
from numpy.typing import ArrayLike
from rscope.rollout import Rollout

if TYPE_CHECKING:
    from drone_playground.simulation.scene import Scene


def _points(frames: Sequence | None, count: int, key: str) -> list[np.ndarray]:
    """Validate frame-aligned world points, retaining only explicit sensor hits."""
    if frames is None:
        return [np.empty((0, 3)) for _ in range(count)]
    if len(frames) != count:
        raise ValueError(f"{key} must have one entry per replay timestamp")
    result = []
    for frame in frames:
        if frame is None:
            result.append(np.empty((0, 3)))
            continue
        points = np.asarray(frame[key] if isinstance(frame, Mapping) else frame, dtype=float)
        if points.size == 0:
            points = points.reshape(0, 3)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"{key} must contain arrays of shape (points, 3)")
        if isinstance(frame, Mapping) and "mask" in frame:
            mask = np.asarray(frame["mask"])
            if mask.shape != points.shape[:1] or mask.dtype != np.bool_:
                raise ValueError(f"{key} mask must be a boolean array of shape (points,)")
            points = points[mask]
        if not np.isfinite(points).all():
            raise ValueError(f"{key} contains nonfinite valid points")
        result.append(points)
    return result


def _scene_xml(scene: Scene | str | Path) -> tuple[ET.Element, dict[str, bytes]]:
    """Copy XML includes and file assets into a closed, portable resource bundle."""
    source = scene if isinstance(scene, (str, Path)) else scene.xml_path
    if source is None:
        if scene.model.ngeom or scene.model.nbody != 1:
            raise ValueError("A nonempty scene must expose its source xml_path")
        return ET.Element("mujoco", model="empty"), {}
    assets = {}
    names = {}

    def read_xml(path: Path, inherited: dict[str, Path]) -> ET.Element:
        root = ET.parse(path).getroot()
        directories = inherited.copy()
        compiler = root.find("compiler")
        if compiler is not None:
            for key in ("assetdir", "meshdir", "texturedir"):
                if key in compiler.attrib:
                    directories[key] = path.parent / compiler.attrib.pop(key)
        for element in root.iter():
            for attr in (
                "file",
                "fileleft",
                "fileright",
                "filefront",
                "fileback",
                "fileup",
                "filedown",
            ):
                if attr not in element.attrib:
                    continue
                base = path.parent
                if element.tag != "include":
                    key = "meshdir" if element.tag == "mesh" else "texturedir"
                    base = directories.get(key, directories.get("assetdir", base))
                target = (base / element.get(attr)).resolve()
                if target not in names:
                    name = f"asset_{len(names):04d}_{target.name}"
                    names[target] = name
                    assets[name] = (
                        ET.tostring(read_xml(target, directories))
                        if element.tag == "include"
                        else target.read_bytes()
                    )
                element.set(attr, names[target])
        return root

    return read_xml(Path(source).resolve(), {}), assets


def _motion(model: mujoco.MjModel, times: np.ndarray, mocap: np.ndarray) -> None:
    """Evaluate the Navigation8 body.user laws at the recorded physics times."""
    if not model.nuser_body:
        return
    if model.nuser_body != 6:
        raise ValueError("Replay body.user requires motion type and five parameters")
    for body in range(1, model.nbody):
        kind, sx, sy, sz, fourth, fifth = model.body_user[body]
        if kind == 0:
            continue
        slot = model.body_mocapid[body]
        if slot < 0 or kind not in (1, 2):
            raise ValueError("Replay supports prescribed mocap motion types 0, 1 and 2")
        if not np.isfinite(model.body_user[body]).all():
            raise ValueError("Nonfinite motion parameters")
        if kind == 1:
            if fifth <= 0:
                raise ValueError("Trefoil slower must be positive")
            u = 2 * times / fifth + fourth
            offset = np.column_stack(
                (
                    sx / 6 * (np.sin(u) + 2 * np.sin(2 * u)),
                    sy / 5 * (np.cos(u) - 2 * np.cos(2 * u)),
                    -sz / 2 * np.sin(3 * u),
                )
            )
        else:
            if fourth <= 0:
                raise ValueError("Bounce period must be positive")
            u = times / fourth + fifth
            triangle = 2 * np.abs(2 * (u - np.floor(u + 0.5))) - 1
            offset = triangle[:, None] * np.array([sx, sy, sz])
        mocap[:, 0, slot] = model.body_pos[body] + offset


def _attach_drone_mesh(root: ET.Element, assets: dict[str, bytes], body: ET.Element) -> None:
    """Attach the existing Crazyflie 2.x visuals to the recorded free-joint body."""
    source = resources.files("drone_playground").joinpath("assets/robots/crazyflie2x")
    model = ET.fromstring(source.joinpath("replay.xml").read_bytes())
    mesh_assets = root.find("asset")
    if mesh_assets is None:
        mesh_assets = ET.SubElement(root, "asset")
    for element in model.find("asset"):
        visual = ET.fromstring(ET.tostring(element))
        if visual.tag == "mesh":
            filename = visual.get("file")
            bundled_name = f"crazyflie_{filename}"
            assets[bundled_name] = source.joinpath(filename).read_bytes()
            visual.set("file", bundled_name)
        mesh_assets.append(visual)
    for geom in model.find("worldbody/body"):
        if geom.tag == "geom" and geom.get("type") == "mesh":
            visual = ET.fromstring(ET.tostring(geom))
            visual.set("mass", "0")
            body.append(visual)
    for filename in ("LICENSE", "SOURCE.txt"):
        assets[f"crazyflie_{filename}"] = source.joinpath(filename).read_bytes()


def export_replay(
    directory: str | Path,
    scene: Scene | str | Path,
    times: ArrayLike,
    positions: ArrayLike,
    quaternions_xyzw: ArrayLike,
    measurements: Sequence | None = None,
    plans: Sequence | None = None,
    *,
    sensor_times: ArrayLike | None = None,
) -> Path:
    """Export one recorded episode without stepping or mutating its simulation.

    Args:
        directory: New or empty output directory; existing records are never replaced.
        scene: Scene with xml_path/model, or an explicit MJCF path. Empty Scene is supported.
        times: Strictly increasing physics timestamps in seconds, shape (T,), T >= 2.
        positions: Recorded world positions in metres, shape (T, 3).
        quaternions_xyzw: Recorded unit body-to-world quaternions, shape (T, 4).
        measurements: T world-point arrays (N, 3), None entries, or mappings containing
            points_world and an optional boolean mask. Only valid hits are displayed.
        plans: T world-position arrays (N, 3), None entries, or mappings containing
            positions and optional received_time/valid_until. Only a valid plan is shown.
        sensor_times: Actual acquisition times for recorded sensor hits, when available.

    Returns:
        Path to episode.mj_unroll, beside scene.xml, assets and rscope_meta.pkl.

    Raises:
        ValueError: Invalid shapes, timestamps, poses or prescribed motion.
        FileExistsError: The output directory is not empty.
    """
    times = np.asarray(times, dtype=float)
    positions = np.asarray(positions, dtype=float)
    quaternions = np.asarray(quaternions_xyzw, dtype=float)
    if (
        times.ndim != 1
        or len(times) < 2
        or not np.isfinite(times).all()
        or np.any(np.diff(times) <= 0)
    ):
        raise ValueError("times must be finite, strictly increasing, with at least two frames")
    count = len(times)
    if positions.shape != (count, 3) or not np.isfinite(positions).all():
        raise ValueError("positions must be finite with shape (T, 3)")
    if (
        quaternions.shape != (count, 4)
        or not np.isfinite(quaternions).all()
        or not np.allclose(np.linalg.norm(quaternions, axis=1), 1, atol=1e-5, rtol=0)
    ):
        raise ValueError("quaternions_xyzw must be unit quaternions with shape (T, 4)")
    quaternions = quaternions / np.linalg.norm(quaternions, axis=1, keepdims=True)
    hits = _points(measurements, count, "points_world")
    trajectories = _points(plans, count, "positions")
    if plans is not None:
        for i, entry in enumerate(plans):
            if not isinstance(entry, Mapping):
                continue
            received = float(entry.get("received_time", times[i]))
            valid_until = float(entry.get("valid_until", times[i]))
            if not np.isfinite([received, valid_until]).all() or received > valid_until:
                raise ValueError("Plan reception and validity times must be finite and ordered")
            if not received - 1e-7 <= times[i] <= valid_until + 1e-7:
                trajectories[i] = np.empty((0, 3))
    if sensor_times is not None:
        sensor_times = np.asarray(sensor_times, dtype=float)
        if sensor_times.shape != (count,) or not np.isfinite(sensor_times).all():
            raise ValueError("Sensor acquisition times must be finite and frame-aligned")
        if np.any(sensor_times > times + 1e-7):
            raise ValueError("Sensor acquisition cannot be later than its replay frame")
    plan_segments = [np.stack((p[:-1], p[1:]), axis=1) for p in trajectories]
    root, assets = _scene_xml(scene)
    if any(
        e.get("name", "").startswith("replay_") or (e.tag == "body" and e.get("name") == "drone")
        for e in root.iter()
    ):
        raise ValueError("Body name drone and names beginning with replay_ are reserved for replay")
    world = ET.SubElement(root, "worldbody")
    drone = ET.SubElement(world, "body", name="drone")
    ET.SubElement(drone, "freejoint", name="replay_drone_joint")
    ET.SubElement(
        drone,
        "geom",
        name="replay_drone",
        type="sphere",
        size="0.07",
        rgba="0 0 0 0",
        contype="0",
        conaffinity="0",
        group="3",
        mass="0.027",
    )
    _attach_drone_mesh(root, assets, drone)
    for i, (start, end) in enumerate(itertools.pairwise(positions)):
        if np.array_equal(start, end):
            continue
        ET.SubElement(
            world,
            "geom",
            name=f"replay_actual_{i}",
            type="capsule",
            size="0.01",
            fromto=" ".join(map(str, np.r_[start, end])),
            rgba="0.1 0.4 1 0.6",
            contype="0",
            conaffinity="0",
        )
    overlays = (("hit", hits, "0.2 1 0.3 1"), ("plan", trajectories, "1 0.2 0.1 1"))
    for label, frames, color in overlays:
        for i in range(max(map(len, frames))):
            body = ET.SubElement(world, "body", name=f"replay_{label}_{i}", mocap="true")
            ET.SubElement(
                body,
                "geom",
                name=f"replay_{label}_geom_{i}",
                type="sphere",
                size="0.025",
                rgba=color,
                contype="0",
                conaffinity="0",
            )
    segment_count = max(map(len, plan_segments))
    if segment_count:
        tendons = root.find("tendon")
        if tendons is None:
            tendons = ET.SubElement(root, "tendon")
        for i in range(segment_count):
            endpoints = []
            for end in ("start", "end"):
                name = f"replay_plan_segment_{i}_{end}"
                marker = ET.SubElement(world, "body", name=name, mocap="true")
                site = f"{name}_site"
                ET.SubElement(marker, "site", name=site, size=".001", rgba="0 0 0 0")
                endpoints.append(site)
            line = ET.SubElement(
                tendons,
                "spatial",
                name=f"replay_plan_line_{i}",
                width=".015",
                rgba="1 .35 .1 .9",
            )
            for site in endpoints:
                ET.SubElement(line, "site", site=site)
    assets["scene.xml"] = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    model = mujoco.MjModel.from_xml_string(assets["scene.xml"].decode(), assets=assets)
    if model.nq != 7 or model.nv != 6:
        raise ValueError("Scene must be jointless; only the recorded drone may have a free joint")
    data = mujoco.MjData(model)
    qpos = np.concatenate((positions, quaternions[:, [3, 0, 1, 2]]), axis=1)[:, None, :]
    qvel = np.zeros((count, 1, model.nv))
    for i, dt in enumerate(np.diff(times)):
        mujoco.mj_differentiatePos(model, qvel[i, 0], dt, qpos[i, 0], qpos[i + 1, 0])
    qvel[-1] = qvel[-2]
    mocap_pos = np.tile(data.mocap_pos, (count, 1, 1, 1))
    mocap_quat = np.tile(data.mocap_quat, (count, 1, 1, 1))
    _motion(model, times, mocap_pos)
    for label, frames, _ in overlays:
        slots = [model.body(f"replay_{label}_{i}").mocapid[0] for i in range(max(map(len, frames)))]
        # ponytail: park unused slots below the scene until RScope supports per-frame visibility.
        mocap_pos[:, 0, slots] = [0, 0, -1e6]
        for t, points in enumerate(frames):
            mocap_pos[t, 0, slots[: len(points)]] = points
    if segment_count:
        slots = [
            model.body(f"replay_plan_segment_{i}_{end}").mocapid[0]
            for i in range(segment_count)
            for end in ("start", "end")
        ]
        mocap_pos[:, 0, slots] = [0, 0, -1e6]
        for t, segments in enumerate(plan_segments):
            mocap_pos[t, 0, slots[: 2 * len(segments)]] = segments.reshape(-1, 3)
    record = Rollout(
        qpos=qpos,
        qvel=qvel,
        mocap_pos=mocap_pos,
        mocap_quat=mocap_quat,
        obs=np.empty((count, 1, 0)),
        reward=np.zeros((count, 1)),
        time=times[:, None],
        metrics={
            "sensor_hits": np.array([len(p) for p in hits])[:, None],
            "plan_points": np.array([len(p) for p in trajectories])[:, None],
            "plan_segments": np.array([len(p) for p in plan_segments])[:, None],
            **(
                {"sensor_acquisition_time": sensor_times[:, None]}
                if sensor_times is not None
                else {}
            ),
        },
    )
    meta = {
        "xml_path": "scene.xml",
        "model_assets": assets,
        "replay": {
            "scene": str(getattr(scene, "name", scene)),
            "geometry_hash": getattr(scene, "geometry_hash", None),
            "asset_sha256": {k: hashlib.sha256(v).hexdigest() for k, v in assets.items()},
            "quaternion_order": "wxyz",
            "qvel": "forward finite difference; last held",
            "reward": "not supplied; zero placeholder",
            "frames": count,
        },
    }
    directory = Path(directory).resolve()
    if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
        raise FileExistsError(f"Replay directory must be new or empty: {directory}")
    directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".replay-", dir=directory.parent) as staging:
        staging = Path(staging)
        for name, content in assets.items():
            path = staging / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        (staging / "rscope_meta.pkl").write_bytes(pickle.dumps(meta, protocol=4))
        (staging / "episode.mj_unroll").write_bytes(pickle.dumps(record, protocol=4))
        staging.replace(directory)
    return directory / "episode.mj_unroll"
