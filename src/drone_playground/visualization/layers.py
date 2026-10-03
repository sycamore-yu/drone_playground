"""Replay-only sensor hit points and causally received planner geometry."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np

from drone_playground.planning.corridors import ConvexPolytope
from drone_playground.references import Trajectory


@dataclass(frozen=True)
class SensorView:
    name: str
    kind: str
    near_m: float
    far_m: float
    display_range_m: float
    rotation: np.ndarray
    translation: np.ndarray
    angles_deg: tuple[float, float]

    def __post_init__(self):
        rotation = np.asarray(self.rotation, float)
        translation = np.asarray(self.translation, float)
        angles = np.asarray(self.angles_deg, float)
        if (
            rotation.shape != (3, 3)
            or translation.shape != (3,)
            or angles.shape != (2,)
            or not np.isfinite(rotation).all()
            or not np.isfinite(translation).all()
            or not np.isfinite(angles).all()
            or not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6)
        ):
            raise ValueError("Sensor view requires finite rigid body extrinsics and angles")
        if (
            not np.isfinite([self.near_m, self.far_m, self.display_range_m]).all()
            or not 0 <= self.near_m < self.display_range_m <= self.far_m
        ):
            raise ValueError("Sensor display range must lie inside the physical range")
        if self.kind == "depth":
            valid = np.all((angles > 0) & (angles < 180))
        else:
            valid = self.kind == "lidar" and -90 <= angles[0] < angles[1] <= 90
        if not valid:
            raise ValueError("Sensor view has invalid field-of-view angles")
        object.__setattr__(self, "rotation", rotation)
        object.__setattr__(self, "translation", translation)

    def line_segments(self):
        if self.kind == "depth":
            hx, hy = np.tan(np.deg2rad(self.angles_deg) / 2)
            corners = np.array([[-hx, -hy, 1], [hx, -hy, 1], [hx, hy, 1], [-hx, hy, 1]])
            near, far = corners * self.near_m, corners * self.display_range_m
            lines = [np.array([near[i], near[(i + 1) % 4]]) for i in range(4)]
            lines += [np.array([near[i], far[i]]) for i in range(4)]
            lines += [np.array([far[i], far[(i + 1) % 4]]) for i in range(4)]
        elif self.kind == "lidar":
            azimuth = np.linspace(0, 2 * np.pi, 49)
            low, high = np.deg2rad(self.angles_deg)
            lines = []
            for elevation in (low, high):
                ring = (
                    self.display_range_m
                    * np.c_[
                        np.cos(elevation) * np.cos(azimuth),
                        np.cos(elevation) * np.sin(azimuth),
                        np.full(49, np.sin(elevation)),
                    ]
                )
                lines.extend(np.stack([ring[:-1], ring[1:]], axis=1))
            for az in np.linspace(0, 2 * np.pi, 12, endpoint=False):
                elevation = np.linspace(low, high, 7)
                arc = (
                    self.display_range_m
                    * np.c_[
                        np.cos(elevation) * np.cos(az),
                        np.cos(elevation) * np.sin(az),
                        np.sin(elevation),
                    ]
                )
                lines.extend(np.stack([arc[:-1], arc[1:]], axis=1))
                for end in (arc[0], arc[-1]):
                    lines.append(np.array([end * self.near_m / self.display_range_m, end]))
        else:
            raise ValueError("Unknown sensor view kind")
        return np.asarray(lines) @ self.rotation.T + self.translation


def sensor_hit_markers(points_world, *, size=0.03):
    """Convert world-frame range-sensor hits to MuJoCo marker descriptors.

    MID360 and D435 are both shown as their surface-hit samples. Ray geometry and
    view-volume outlines are intentionally absent from normal replay rendering.
    """
    points = np.asarray(points_world, dtype=float).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    return [(point, size) for point in points]


def sensor_view(calibration, *, lidar_display_range_m=10.0):
    """Translate actual sensor calibration into the common replay descriptor."""
    if calibration is None:
        return None
    if isinstance(calibration, SensorView):
        return calibration
    optical = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)
    intrinsics = calibration.get("intrinsics", calibration)
    if "horizontal_fov_deg" in intrinsics:
        extrinsics = calibration.get("extrinsics", {})
        rotation = extrinsics.get("rotation_body_from_optical")
        if rotation is None:
            angle = math.radians(calibration.get("pitch_degrees", 0))
            rotation = (
                np.array(
                    [
                        [math.cos(angle), 0, -math.sin(angle)],
                        [0, 1, 0],
                        [math.sin(angle), 0, math.cos(angle)],
                    ]
                )
                @ optical
            )
        translation = extrinsics.get(
            "translation_m", calibration.get("mount_translation_m", [0, 0, 0])
        )
        near, far = intrinsics["near_m"], intrinsics["far_m"]
        return SensorView(
            calibration.get("sensor", "depth"),
            "depth",
            near,
            far,
            far,
            np.asarray(rotation, float),
            np.asarray(translation, float),
            (intrinsics["horizontal_fov_deg"], intrinsics["vertical_fov_deg"]),
        )
    if "range_m" in calibration:
        if "pattern" in calibration:
            angles = tuple(calibration["pattern"]["elevation_deg"])
        else:
            low = calibration["elevation_start_deg"]
            high = low + calibration["elevation_span_deg"] * (
                1 - 1 / calibration["elevation_count"]
            )
            angles = (low, high)
        near, far = calibration["range_m"]
        return SensorView(
            calibration.get("sensor", calibration.get("name", "lidar")),
            "lidar",
            near,
            far,
            min(far, lidar_display_range_m),
            np.eye(3),
            np.zeros(3),
            angles,
        )
    return None


def polytope_edges(polytope: ConvexPolytope):
    """Enumerate bounded convex geometry; omit triangulation diagonals."""
    from scipy.optimize import linprog
    from scipy.spatial import ConvexHull, HalfspaceIntersection, QhullError

    vertices = polytope.vertices
    if vertices is None:
        planes = polytope.halfspaces
        a, b = planes[:, :3], planes[:, 3]
        for direction in np.r_[np.eye(3), -np.eye(3)]:
            result = linprog(
                direction,
                A_ub=a,
                b_ub=-b,
                bounds=[(None, None)] * 3,
                method="highs",
            )
            if not result.success:
                raise ValueError("SFC must have a bounded nonempty 3D interior")
        interior = linprog(
            [0, 0, 0, -1],
            A_ub=np.c_[a, np.linalg.norm(a, axis=1)],
            b_ub=-b,
            bounds=[(None, None)] * 3 + [(0, None)],
            method="highs",
        )
        if not interior.success or interior.x[3] <= 1e-8:
            raise ValueError("SFC must have a bounded nonempty 3D interior")
        try:
            vertices = HalfspaceIntersection(planes, interior.x[:3]).intersections
        except QhullError as exc:
            raise ValueError("Cannot enumerate SFC interior") from exc
    hull = ConvexHull(vertices)
    edges = {}
    for face, normal in zip(hull.simplices, hull.equations[:, :3]):
        for a, b in zip(face, np.roll(face, -1)):
            edges.setdefault(tuple(sorted((int(a), int(b)))), []).append(normal)
    indices = [
        edge
        for edge, normals in edges.items()
        if len(normals) < 2 or not np.allclose(normals[0], normals[1], atol=1e-7, rtol=0)
    ]
    return vertices[np.asarray(indices)]


@dataclass(frozen=True)
class PlanningFrame:
    time: float  # First received on the simulation clock.
    valid_until: float
    layer: str
    trajectory: Trajectory | None = None
    polytopes: tuple[ConvexPolytope, ...] = ()

    def __post_init__(self):
        if not self.layer or not np.isfinite([self.time, self.valid_until]).all() or self.time < 0:
            raise ValueError("Planning frames require a name and finite simulation times")
        object.__setattr__(self, "polytopes", tuple(self.polytopes))


@dataclass
class ReplayLayers:
    sensor: SensorView | None = None
    point_cloud: np.ndarray | None = None
    point_cloud_sequence: np.ndarray | None = None
    point_cloud_metadata: dict | None = None
    planning: list[PlanningFrame] = field(default_factory=list)
    trajectory_samples: int = 64

    def __post_init__(self):
        if not 2 <= self.trajectory_samples <= 512:
            raise ValueError("Trajectory rendering needs 2..512 samples")
        if self.point_cloud is not None and self.point_cloud_sequence is not None:
            raise ValueError("Choose either a static point cloud or a point-cloud sequence")

    def add_to_model(self, xml, native_trace, drone_body):
        """Add nonphysical geometry and animate endpoints in the native file."""
        metadata = dict(schema_version=1, frame="world", units="m", sensor=None, layers=[])
        if self.sensor is not None:
            sensor = self.sensor
            metadata["sensor"] = dict(
                name=sensor.name,
                kind=sensor.kind,
                near_m=sensor.near_m,
                far_m=sensor.far_m,
                display_range_m=sensor.display_range_m,
                angles_deg=sensor.angles_deg,
                rotation_body_from_sensor=sensor.rotation.tolist(),
                translation_m=sensor.translation.tolist(),
                meaning="calibration descriptor only; replay renders hit points, never a field-of-view envelope",
            )
        if self.point_cloud is not None:
            world = xml.find("worldbody")
            if world is None:
                raise ValueError("Replay model has no worldbody")
            assets = xml.find("asset")
            if assets is None:
                assets = ET.SubElement(xml, "asset")
            ET.SubElement(
                assets,
                "material",
                name="viz_sensor_hits",
                rgba=".1 .75 1 .92",
                emission="1",
            )
            for index, (point, size) in enumerate(sensor_hit_markers(self.point_cloud)):
                ET.SubElement(
                    world,
                    "geom",
                    name=f"viz_sensor_hit_{index}",
                    type="sphere",
                    pos=" ".join(map(str, point)),
                    size=str(size),
                    material="viz_sensor_hits",
                    contype="0",
                    conaffinity="0",
                    density="0",
                    group="4",
                )
            metadata["point_cloud"] = dict(
                renderer="MuJoCo markers",
                points=len(sensor_hit_markers(self.point_cloud)),
                frame="world",
            )
        if self.point_cloud_sequence is not None:
            points = np.asarray(self.point_cloud_sequence, dtype=float)
            if points.ndim != 3 or points.shape[-1] != 3:
                raise ValueError("point_cloud_sequence requires [T, N, 3] world-frame points")
            world = xml.find("worldbody")
            if world is None:
                raise ValueError("Replay model has no worldbody")
            assets = xml.find("asset")
            if assets is None:
                assets = ET.SubElement(xml, "asset")
            ET.SubElement(
                assets,
                "material",
                name="viz_sensor_hit_sequence",
                rgba=".1 .75 1 .92",
                emission="1",
            )
            valid = np.isfinite(points).all(axis=-1)
            count = points.shape[1]
            for index in range(count):
                body = ET.SubElement(
                    world,
                    "body",
                    name=f"viz_sensor_hit_{index}",
                    mocap="true",
                    pos="0 0 -1000",
                )
                ET.SubElement(
                    body,
                    "geom",
                    name=f"viz_hit_geom_{index}",
                    type="sphere",
                    size="0.03",
                    material="viz_sensor_hit_sequence",
                    contype="0",
                    conaffinity="0",
                    density="0",
                    group="4",
                )
            steps, count = points.shape[:2]
            replay_steps, batch = native_trace["mocap_pos"].shape[:2]
            if steps != replay_steps:
                raise ValueError("point cloud sequence length differs from replay")
            if batch != 1:
                raise ValueError("Point-cloud replay requires one episode per file")
            extra = np.full((steps, batch, count, 3), -1000.0, dtype=float)
            extra[:, 0] = np.where(valid[..., None], points, -1000.0)
            native_trace["mocap_pos"] = np.concatenate([native_trace["mocap_pos"], extra], axis=2)
            native_trace["mocap_quat"] = np.concatenate(
                [
                    native_trace["mocap_quat"],
                    np.broadcast_to(
                        np.array([1.0, 0.0, 0.0, 0.0]),
                        (steps, batch, count, 4),
                    ),
                ],
                axis=2,
            )
            metadata["point_cloud_sequence"] = {
                **(self.point_cloud_metadata or {}),
                "renderer": "MuJoCo mocap markers",
                "points_per_frame": count,
                "frames": steps,
                "frame": "world",
                "valid_points_per_frame": [int(value) for value in valid.sum(axis=1)],
            }
        if not self.planning:
            return metadata
        world = xml.find("worldbody")
        tendon = xml.find("tendon")
        if tendon is None:
            tendon = ET.SubElement(xml, "tendon")
        times = native_trace["time"]
        steps, batch = times.shape
        if batch != 1:
            raise ValueError("Planner layers require one episode per replay")
        coordinates = []
        layer_names = list(dict.fromkeys(frame.layer for frame in self.planning))
        for layer_index, name in enumerate(layer_names):
            frames = sorted(
                (f for f in self.planning if f.layer == name),
                key=lambda f: f.time,
            )
            geometries = [
                np.concatenate([polytope_edges(p) for p in f.polytopes])
                if f.polytopes
                else np.empty((0, 2, 3))
                for f in frames
            ]
            has_curve = any(f.trajectory is not None for f in frames)
            capacity = max(
                len(g) + (self.trajectory_samples - 1 if f.trajectory is not None else 0)
                for f, g in zip(frames, geometries)
            )
            if not capacity:
                continue
            endpoints = np.zeros((steps, batch, capacity, 2, 3))
            endpoints[..., 2] = -10000
            endpoints[..., 1, 0] = 0.001
            received = np.array([f.time for f in frames])
            for t in range(steps):
                for case in range(batch):
                    now = float(times[t, case])
                    index = np.searchsorted(received, now, side="right") - 1
                    if index < 0:
                        continue
                    frame = frames[index]
                    if now > frame.valid_until:
                        continue
                    segments = []
                    if frame.trajectory is not None:
                        curve = frame.trajectory
                        start = max(now, curve.start_time)
                        if start < min(curve.end_time, frame.valid_until):
                            samples = np.linspace(
                                start,
                                min(curve.end_time, frame.valid_until),
                                self.trajectory_samples,
                            )
                            points = curve.sample_many(samples)["position"]
                            segments.extend(np.stack([points[:-1], points[1:]], axis=1))
                    segments.extend(geometries[index])
                    if segments:
                        endpoints[t, case, : len(segments)] = segments
            for i in range(capacity):
                prefix = f"viz_plan_{layer_index}_{i}"
                sites = []
                for end in range(2):
                    body = ET.SubElement(
                        world,
                        "body",
                        name=f"{prefix}_{end}",
                        mocap="true",
                        pos=f"{end * 0.001} 0 0",
                    )
                    site = f"{prefix}_site_{end}"
                    sites.append(site)
                    ET.SubElement(body, "site", name=site, size=".001", rgba="0 0 0 0")
                line = ET.SubElement(
                    tendon,
                    "spatial",
                    name=prefix,
                    width=".022",
                    rgba=".15 .9 .25 .9" if has_curve else "1 .55 .05 .6",
                )
                for site in sites:
                    ET.SubElement(line, "site", site=site)
            coordinates.append(endpoints.reshape(steps, batch, capacity * 2, 3))
            metadata["layers"].append(
                dict(
                    name=name,
                    renderer_prefix=f"viz_plan_{layer_index}_",
                    segments=capacity,
                    received_snapshots=len(frames),
                    kind="trajectory" if has_curve else "sfc",
                )
            )
        if coordinates:
            extra = np.concatenate(coordinates, axis=2)
            native_trace["mocap_pos"] = np.concatenate([native_trace["mocap_pos"], extra], axis=2)
            quat = np.zeros((*extra.shape[:-1], 4))
            quat[..., 0] = 1
            native_trace["mocap_quat"] = np.concatenate([native_trace["mocap_quat"], quat], axis=2)
        return metadata


def layers_from_decisions(rows, *, sensor=None, trajectory_samples=64):
    """Turn causally received, verified decision records into deduplicated layers."""
    import hashlib
    import json
    from dataclasses import asdict

    frames, previous = [], {}

    def emit(frame):
        data = dict(
            valid_until=frame.valid_until,
            trajectory=None if frame.trajectory is None else asdict(frame.trajectory),
            polytopes=[asdict(p) for p in frame.polytopes],
        )
        digest = hashlib.sha256(
            json.dumps(
                data,
                sort_keys=True,
                default=lambda v: v.tolist(),
                allow_nan=False,
            ).encode()
        ).hexdigest()
        if previous.get(frame.layer) != digest:
            frames.append(frame)
            previous[frame.layer] = digest

    known = set()
    for row in rows:
        now, reply = float(row["time"]), row["reply"]
        stages = reply.get("stages", [dict(reply, stage="output")])
        present = set()
        for stage in stages:
            prefix = f"stage-{stage['stage']}"
            value = stage.get("trajectory", stage.get("output"))
            name = prefix + "/trajectory"
            if isinstance(value, Trajectory):
                emit(
                    PlanningFrame(
                        now,
                        min(
                            stage.get("valid_until", value.end_time),
                            value.end_time,
                        ),
                        name,
                        trajectory=value,
                    )
                )
                present.add(name)
            if stage.get("corridors") or stage.get("trajectory_previews"):
                for corridor in stage.get("corridors", ()):
                    name = prefix + "/sfc/" + corridor.name
                    emit(
                        PlanningFrame(
                            now,
                            corridor.valid_until,
                            name,
                            polytopes=corridor.polytopes,
                        )
                    )
                    present.add(name)
                for preview in stage.get("trajectory_previews", ()):
                    name = prefix + "/preview/" + preview.name
                    emit(
                        PlanningFrame(
                            now,
                            preview.valid_until,
                            name,
                            trajectory=preview.trajectory,
                        )
                    )
                    present.add(name)
        for name in known - present:
            emit(PlanningFrame(now, now, name))
        known = present
    return ReplayLayers(sensor=sensor, planning=frames, trajectory_samples=trajectory_samples)
