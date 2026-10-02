"""Safe rscope export and publication for recorded Drone Playground runs."""

from __future__ import annotations

import copy
import json
import os
import shutil
import tempfile
import threading
import uuid
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np

_RSCOPE_LOCK = threading.RLock()
_PUBLISH_MARKER = ".drone_playground_publish.json"


@contextmanager
def _exclusive_rscope() -> Iterator[None]:
    """Serialize rscope's mutable module-level configuration across threads and processes."""
    lock_path = Path("/tmp/rscope/.drone_playground.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with _RSCOPE_LOCK, lock_path.open("a+b") as lock_file:
        try:
            import fcntl

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        except ImportError:
            fcntl = None
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


@contextmanager
def _temporary_rscope_config(
    base_path: Path, temp_path: Path
) -> Iterator[None]:
    """Point rscope at a private staging directory and restore its global config."""
    from rscope import config

    previous = (config.BASE_PATH, config.TEMP_PATH, config.META_PATH)
    config.BASE_PATH = base_path
    config.TEMP_PATH = temp_path
    config.META_PATH = base_path / "rscope_meta.pkl"
    try:
        yield
    finally:
        config.BASE_PATH, config.TEMP_PATH, config.META_PATH = previous


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        tmp_path = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)


def _batch_base(array: np.ndarray, batch: int) -> np.ndarray:
    if array.ndim == 0:
        raise ValueError(
            "rscope base array is missing its environment dimension"
        )
    if array.shape[0] >= batch:
        return np.array(array[:batch], copy=True)
    if array.shape[0] == 1:
        return np.repeat(array, batch, axis=0)
    raise ValueError(
        f"simulation has {array.shape[0]} worlds but rollout needs {batch}"
    )


def _validated_trace(trace: dict[str, Any]) -> tuple[int, int, dict[str, Any]]:
    required = ("pos", "quat", "time", "obs", "reward", "metrics")
    missing = [name for name in required if name not in trace]
    if missing:
        raise KeyError(
            f"trace is missing required fields: {', '.join(missing)}"
        )

    converted = {
        "pos": np.asarray(trace["pos"]),
        "quat": np.asarray(trace["quat"]),
        "time": np.asarray(trace["time"]),
        "obs": np.asarray(trace["obs"]),
        "reward": np.asarray(trace["reward"]),
        "metrics": {
            name: np.asarray(value) for name, value in trace["metrics"].items()
        },
    }
    if "actions" in trace:
        converted["actions"] = np.asarray(trace["actions"])

    pos = converted["pos"]
    if pos.ndim != 3 or pos.shape[-1] != 3:
        raise ValueError(f"pos must have shape [T, B, 3], got {pos.shape}")
    steps, batch = pos.shape[:2]
    expected = {
        "quat": (steps, batch, 4),
        "time": (steps, batch),
        "reward": (steps, batch),
    }
    for name, shape in expected.items():
        if converted[name].shape != shape:
            raise ValueError(
                f"{name} must have shape {shape}, got {converted[name].shape}"
            )
    if converted["obs"].ndim != 3 or converted["obs"].shape[:2] != (
        steps,
        batch,
    ):
        raise ValueError(
            f"obs must have shape [T, B, O] with T={steps}, B={batch}, got {converted['obs'].shape}"
        )
    for name, value in converted["metrics"].items():
        if value.shape != (steps, batch):
            raise ValueError(
                f"metric {name!r} must have shape {(steps, batch)}, got {value.shape}"
            )
    if "actions" in converted:
        actions = converted["actions"]
        if (
            actions.ndim != 3
            or actions.shape[:2] != (steps, batch)
            or actions.shape[-1] < 1
        ):
            raise ValueError(
                f"actions must have shape [T={steps}, B={batch}, A>0], got {actions.shape}"
            )
    return steps, batch, converted


def _model_bundle(sim: Any, directory: Path) -> tuple[Path, dict[str, bytes]]:
    """Serialize a replayable XML and copy referenced mesh assets beside it."""
    xml = ET.fromstring(sim.spec.to_xml())
    # Attached MjSpecs can emit repeated empty root defaults (e.g. main:0 for
    # the drone and each gate). Only empty, direct-child duplicates are redundant.
    # Keep all actual default properties; reject ambiguous populated duplicates.
    defaults = xml.find("default")
    if defaults is not None:
        seen = {}
        for node in list(defaults):
            name = node.get("class") if node.tag == "default" else None
            if name is None:
                continue
            if name not in seen:
                seen[name] = node
                continue
            previous = seen[name]
            empty = len(node) == 0 and set(node.attrib) == {"class"}
            previous_empty = len(previous) == 0 and set(previous.attrib) == {
                "class"
            }
            if empty:
                defaults.remove(node)
            elif previous_empty:
                defaults.remove(previous)
                seen[name] = node
            else:
                raise ValueError(
                    f"Conflicting nonempty attached defaults: {name}"
                )
    dummy = xml.find(".//body[@name='_dummy']")
    if dummy is not None and dummy.find("inertial") is None:
        body = sim.mj_model.body("_dummy")
        ET.SubElement(
            dummy,
            "inertial",
            {
                "pos": " ".join(map(str, np.asarray(body.ipos))),
                "quat": " ".join(map(str, np.asarray(body.iquat))),
                "mass": str(float(np.asarray(body.mass).item())),
                "diaginertia": " ".join(map(str, np.asarray(body.inertia))),
            },
        )

    raw_assets = {
        str(name): bytes(value) for name, value in dict(sim.spec.assets).items()
    }
    model_assets = dict(raw_assets)
    meshdir_value = getattr(sim.spec.compiler, "meshdir", "")
    meshdir = Path(str(meshdir_value)) if meshdir_value else Path(".")
    compiler = xml.find("compiler")
    if compiler is not None:
        compiler.set("meshdir", "")

    for index, mesh in enumerate(xml.findall("./asset/mesh")):
        source_name = mesh.get("file")
        if not source_name:
            continue
        if source_name in raw_assets:
            payload = raw_assets[source_name]
        else:
            source = Path(source_name)
            if not source.is_absolute():
                source = meshdir / source
            payload = source.read_bytes()
        relative = Path("assets") / f"mesh_{index:03d}_{Path(source_name).name}"
        mesh.set("file", relative.as_posix())
        _atomic_write_bytes(directory / relative, payload)
        model_assets[relative.as_posix()] = payload

    texturedir_value = getattr(sim.spec.compiler, "texturedir", "")
    texturedir = Path(str(texturedir_value)) if texturedir_value else Path(".")
    if compiler is not None:
        compiler.set("texturedir", "")
    for index, texture in enumerate(xml.findall("./asset/texture")):
        source_name = texture.get("file")
        if not source_name:
            continue
        if source_name in raw_assets:
            payload = raw_assets[source_name]
        else:
            source = Path(source_name)
            if not source.is_absolute():
                source = texturedir / source
            payload = source.read_bytes()
        relative = (
            Path("assets") / f"texture_{index:03d}_{Path(source_name).name}"
        )
        texture.set("file", relative.as_posix())
        _atomic_write_bytes(directory / relative, payload)
        model_assets[relative.as_posix()] = payload

    xml_payload = ET.tostring(xml, encoding="utf-8")
    xml_path = directory / "scene.xml"
    _atomic_write_bytes(xml_path, xml_payload)
    return xml_path, model_assets


def _native_rollout(
    sim: Any, trace: dict[str, Any]
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    steps, batch, values = _validated_trace(trace)
    drone_mocap_ids = np.asarray(sim.data.core.drone_mocap_ids).reshape(-1)
    if drone_mocap_ids.size != 1:
        raise ValueError(
            "export_rollout currently requires one drone per environment because trace pos/quat "
            "have no drone axis"
        )
    mocap_id = int(drone_mocap_ids[0])

    qpos_base = _batch_base(np.asarray(sim.mjx_data.qpos), batch)
    qvel_base = _batch_base(np.asarray(sim.mjx_data.qvel), batch)
    mocap_pos_base = _batch_base(np.asarray(sim.mjx_data.mocap_pos), batch)
    mocap_quat_base = _batch_base(np.asarray(sim.mjx_data.mocap_quat), batch)
    qpos = np.repeat(qpos_base[None, ...], steps, axis=0)
    qvel = np.repeat(qvel_base[None, ...], steps, axis=0)
    mocap_pos = np.repeat(mocap_pos_base[None, ...], steps, axis=0)
    mocap_quat = np.repeat(mocap_quat_base[None, ...], steps, axis=0)
    mocap_pos[:, :, mocap_id, :] = values["pos"]
    mocap_quat[:, :, mocap_id, :] = np.roll(values["quat"], 1, axis=-1)

    # Analytic scene obstacles are animated through their own mocap bodies, so a
    # replay shows the geometry that was actually sensed, not a static snapshot.
    obstacle_ids = getattr(sim.data.core, "obstacle_mocap_ids", None)
    if obstacle_ids is not None:
        if "obstacle_pos" not in trace:
            raise KeyError(
                "the replay model declares obstacles but the trace has no obstacle_pos"
            )
        obstacle_pos = np.asarray(trace["obstacle_pos"])
        obstacle_ids = np.asarray(obstacle_ids).reshape(-1)
        if obstacle_pos.shape[:2] != (steps, batch) or obstacle_pos.shape[
            2
        ] != len(obstacle_ids):
            raise ValueError(
                f"obstacle_pos must have shape {(steps, batch, len(obstacle_ids), 3)}, "
                f"got {obstacle_pos.shape}"
            )
        mocap_pos[:, :, obstacle_ids, :] = obstacle_pos
        mocap_quat[:, :, obstacle_ids, :] = np.broadcast_to(
            np.array([1.0, 0.0, 0.0, 0.0]), (steps, batch, len(obstacle_ids), 4)
        )

    metrics = dict(values["metrics"])
    actions = values.get("actions")
    if actions is not None:
        for axis in range(actions.shape[-1]):
            metrics[f"action/{axis}"] = actions[:, :, axis]
    native_trace = {
        "qpos": qpos,
        "qvel": qvel,
        "mocap_pos": mocap_pos,
        "mocap_quat": mocap_quat,
        "time": values["time"],
        "metrics": metrics,
    }
    return native_trace, values["obs"], values["reward"]


def trim_episode(trace: dict[str, Any], case: int) -> dict[str, Any]:
    """Take one contiguous active episode, including its terminal transition."""
    steps, batch = np.asarray(trace["pos"]).shape[:2]
    if not 0 <= case < batch:
        raise IndexError("Replay case index is out of range")
    active = np.asarray(trace.get("active", np.ones((steps, batch), bool)))[
        :, case
    ].astype(bool)
    length = int(active.sum())
    if length < 1 or not np.array_equal(active, np.arange(steps) < length):
        raise ValueError(
            "Replay requires contiguous active frames including the terminal frame"
        )

    def take(value):
        if isinstance(value, dict):
            return {key: take(item) for key, item in value.items()}
        return np.asarray(value)[:length, case : case + 1]

    return take(trace)


def export_rollout(
    sim: Any, directory: Path, trace: dict[str, Any], *, visualization=None
) -> Path:
    """Export one rollout with rscope's native atomic writer and a self-contained model bundle.

    Args:
        sim: Crazyflow simulation object that owns the MuJoCo spec and compiled model.
        directory: Run-specific rollout directory. Existing unrelated files are preserved.
        trace: Recorded arrays using Drone Playground's ``[T, B, ...]`` trace contract.

    Returns:
        Path to the exported ``.mj_unroll`` file.
    """
    directory = Path(directory).resolve()
    layers = (
        visualization
        if visualization is not None
        else getattr(sim, "replay_visualization", None)
    )
    if (
        layers is not None
        and layers.planning
        and np.asarray(trace["pos"]).shape[1] != 1
    ):
        raise ValueError(
            "Planner layers require one episode; export each case separately"
        )
    if "active" in trace:
        # A file contains complete trajectories with a single time dimension.
        # Split unequal episode lengths so padding never appears in a replay.
        active = np.asarray(trace["active"], dtype=bool)
        paths = []
        for case in range(active.shape[1]):
            single = trim_episode(trace, case)
            single.pop("active", None)
            target = directory if case == 0 else directory / f"case-{case:03d}"
            path = export_rollout(
                sim, target, single, visualization=visualization
            )
            paths.append(
                dict(
                    case=case,
                    frames=len(single["time"]),
                    final_time_s=float(single["time"][-1, 0]),
                    file=str(path.relative_to(directory)),
                )
            )
        _atomic_write_bytes(
            directory / "episodes.json",
            (json.dumps(paths, indent=2) + "\n").encode(),
        )
        return directory / paths[0]["file"]
    directory.mkdir(parents=True, exist_ok=True)
    identity = getattr(sim, "component_identity", None)
    if identity is not None:
        _atomic_write_bytes(
            directory / "components.json",
            (
                json.dumps(identity, ensure_ascii=False, indent=2) + "\n"
            ).encode(),
        )
    xml_path, model_assets = _model_bundle(sim, directory)
    native_trace, obs, reward = _native_rollout(sim, trace)
    sensor_context = getattr(sim, "replay_sensor_context", None)
    if (
        sensor_context is not None
        and layers is not None
        and layers.point_cloud_sequence is None
    ):
        from drone_playground.visualization.sensor_hits import sensor_hit_sequence

        cloud, cloud_metadata = sensor_hit_sequence(sensor_context, trace)
        layers = copy.deepcopy(layers)
        layers.point_cloud_sequence = cloud
        layers.point_cloud_metadata = cloud_metadata
    if layers is not None:
        xml = ET.fromstring(xml_path.read_bytes())
        mocap_id = int(np.asarray(sim.data.core.drone_mocap_ids).reshape(-1)[0])
        body_id = int(np.flatnonzero(sim.mj_model.body_mocapid == mocap_id)[0])
        body_name = sim.mj_model.body(body_id).name
        drone_body = xml.find(f".//body[@name='{body_name}']")
        if drone_body is None:
            raise ValueError("Replay model is missing its drone mocap body")
        metadata = layers.add_to_model(xml, native_trace, drone_body)
        _atomic_write_bytes(xml_path, ET.tostring(xml, encoding="utf-8"))
        _atomic_write_bytes(
            directory / "replay-visualization.json",
            (json.dumps(metadata, indent=2) + "\n").encode(),
        )

    return _write_native(
        directory, xml_path, model_assets, native_trace, obs, reward
    )


def _write_native(directory, xml_path, model_assets, native_trace, obs, reward):
    from rscope import rscope_utils

    with _exclusive_rscope():
        stage_root = Path(
            tempfile.mkdtemp(prefix=".rscope-export-", dir=directory.parent)
        )
        stage_base = stage_root / "base"
        stage_temp = stage_root / "temp"
        try:
            with _temporary_rscope_config(stage_base, stage_temp):
                rscope_utils.rscope_init(xml_path, model_assets=model_assets)
                rscope_utils.dump_eval(native_trace, obs, reward)
            unrolls = list(stage_base.glob("*.mj_unroll"))
            if len(unrolls) != 1:
                raise RuntimeError(
                    f"rscope writer produced {len(unrolls)} rollout files"
                )
            destination = directory / unrolls[0].name
            if destination.exists():
                destination = (
                    directory
                    / f"{unrolls[0].stem}-{uuid.uuid4().hex[:8]}.mj_unroll"
                )
            os.replace(unrolls[0], destination)
            _atomic_write_bytes(
                directory / "rscope_meta.pkl",
                (stage_base / "rscope_meta.pkl").read_bytes(),
            )
            return destination
        finally:
            shutil.rmtree(stage_root, ignore_errors=True)


def _publication_files(directory: Path) -> list[Path]:
    meta = directory / "rscope_meta.pkl"
    unrolls = sorted(directory.glob("*.mj_unroll"))
    if not meta.is_file():
        raise FileNotFoundError(f"missing rscope metadata: {meta}")
    if not unrolls:
        raise FileNotFoundError(f"no .mj_unroll files found in {directory}")
    files = [meta, *unrolls]
    xml = directory / "scene.xml"
    if xml.is_file():
        files.append(xml)
    for name in ("components.json", "replay-visualization.json"):
        identity = directory / name
        if identity.is_file():
            files.append(identity)
    assets = directory / "assets"
    if assets.is_dir():
        files.extend(
            sorted(path for path in assets.rglob("*") if path.is_file())
        )
    return files


def _validate_active_directory(active_dir: Path) -> None:
    existing = {
        path.relative_to(active_dir).as_posix()
        for path in active_dir.rglob("*")
        if path.is_file()
    }
    if not existing:
        return
    marker_path = active_dir / _PUBLISH_MARKER
    if not marker_path.is_file():
        raise RuntimeError(
            "active rscope directory contains unknown files and has no publish marker: "
            f"{active_dir}"
        )
    marker = json.loads(marker_path.read_text())
    managed = set(marker.get("files", [])) | {_PUBLISH_MARKER}
    unknown = sorted(existing - managed)
    if unknown:
        raise RuntimeError(
            f"active rscope directory contains unknown files: {', '.join(unknown)}"
        )


def publish_run(
    directory: Path,
    active_dir: Path = Path("/tmp/rscope/active_run"),
) -> Path:
    """Publish a selected run into rscope's active directory without mutating the source run."""
    directory = Path(directory).resolve()
    active_dir = Path(active_dir).resolve()
    if directory == active_dir:
        raise ValueError(
            "source rollout directory and active directory must differ"
        )
    files = _publication_files(directory)

    with _exclusive_rscope():
        active_dir.parent.mkdir(parents=True, exist_ok=True)
        if active_dir.exists():
            _validate_active_directory(active_dir)
        stage = Path(
            tempfile.mkdtemp(
                prefix=f".{active_dir.name}.publish-", dir=active_dir.parent
            )
        )
        backup = (
            active_dir.parent / f".{active_dir.name}.backup-{uuid.uuid4().hex}"
        )
        try:
            published: list[str] = []
            for source in files:
                relative = source.relative_to(directory)
                destination = stage / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                published.append(relative.as_posix())
            marker = {
                "source": str(directory),
                "files": sorted(published),
            }
            (stage / _PUBLISH_MARKER).write_text(
                json.dumps(marker, indent=2, ensure_ascii=False, sort_keys=True)
                + "\n"
            )

            had_active = active_dir.exists()
            if had_active:
                os.replace(active_dir, backup)
            try:
                os.replace(stage, active_dir)
            except BaseException:
                if had_active and backup.exists() and not active_dir.exists():
                    os.replace(backup, active_dir)
                raise
            if backup.exists():
                shutil.rmtree(backup)
            return active_dir
        finally:
            if stage.exists():
                shutil.rmtree(stage, ignore_errors=True)
            if backup.exists() and active_dir.exists():
                shutil.rmtree(backup, ignore_errors=True)


def append_rollout(
    directory: Path, active_dir: Path = Path("/tmp/rscope/active_run")
) -> Path:
    """Append complete snapshots without replacing the directory watched by rscope.

    Native rscope 0.0.8 handles created events, not moved events. A hard link to
    a fully written temporary file emits creation only after the payload exists.
    """
    directory, active_dir = (
        Path(directory).resolve(),
        Path(active_dir).resolve(),
    )
    files = _publication_files(directory)
    with _exclusive_rscope():
        _validate_active_directory(active_dir)
        if (directory / "scene.xml").read_bytes() != (
            active_dir / "scene.xml"
        ).read_bytes():
            raise ValueError(
                "Live snapshots must use the same model; select a new run explicitly"
            )
        for source in files:
            if "assets" in source.relative_to(directory).parts:
                target = active_dir / source.relative_to(directory)
                if (
                    not target.exists()
                    or source.read_bytes() != target.read_bytes()
                ):
                    raise ValueError(
                        "Live snapshots must use identical model assets"
                    )
        marker_path = active_dir / _PUBLISH_MARKER
        marker = json.loads(marker_path.read_text())
        names = set(marker["files"])
        for source in files:
            if source.suffix != ".mj_unroll":
                continue
            target = active_dir / source.name
            if target.exists():
                target = (
                    active_dir
                    / f"{source.stem}-{uuid.uuid4().hex[:8]}.mj_unroll"
                )
            fd, temporary_name = tempfile.mkstemp(
                prefix=".complete-", dir=active_dir
            )
            os.close(fd)
            temporary = Path(temporary_name)
            try:
                shutil.copy2(source, temporary)
                os.link(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
            names.add(target.name)
        marker.update(files=sorted(names), latest_source=str(directory))
        _atomic_write_bytes(
            marker_path, (json.dumps(marker, indent=2) + "\n").encode()
        )
    return active_dir


def publish_snapshot(
    directory: Path,
    run_directory: Path,
    *,
    first: bool = False,
    active_dir: Path = Path("/tmp/rscope/active_run"),
) -> dict:
    """Publish optional live output without changing the outcome of saved training.

    Selecting another run is an ordinary viewer operation. Its directory remains
    selected until an explicit first publication chooses a new run. Publication
    failures are returned for logging; checkpoint and evaluation errors stay fatal
    at their own, mandatory boundaries.
    """
    directory, run_directory, active_dir = (
        Path(directory).resolve(),
        Path(run_directory).resolve(),
        Path(active_dir).resolve(),
    )
    try:
        if not directory.is_relative_to(run_directory):
            raise ValueError("Snapshot directory must belong to its run")
        if not first:
            marker_path = active_dir / _PUBLISH_MARKER
            if not marker_path.is_file():
                return {
                    "status": "not-selected",
                    "reason": "no active owned selection",
                }
            marker = json.loads(marker_path.read_text())
            selected = Path(marker["source"]).resolve()
            if not selected.is_relative_to(run_directory):
                return {
                    "status": "not-selected",
                    "selected_source": str(selected),
                }
        (publish_run if first else append_rollout)(directory, active_dir)
        return {
            "status": "published",
            "directory": str(directory),
            "active_dir": str(active_dir),
        }
    except Exception as error:
        return {
            "status": "error",
            "error": repr(error),
            "directory": str(directory),
        }


def enhance_replay(
    source: Path, directory: Path, visualization, *, drone_mocap_id=0
) -> Path:
    """Copy a trusted local RScope replay and add layers without reevaluating it."""
    import hashlib
    import pickle

    import mujoco

    source, directory = Path(source).resolve(), Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    with source.open("rb") as handle:
        recorded = pickle.load(handle)
    with (source.parent / "rscope_meta.pkl").open("rb") as handle:
        meta = pickle.load(handle)
    assets = dict(meta["model_assets"])
    xml_name = Path(meta["xml_path"]).name
    xml = ET.fromstring(assets[xml_name])
    model = mujoco.MjModel.from_xml_string(
        assets[xml_name].decode(), assets=assets
    )
    ids = np.flatnonzero(model.body_mocapid == drone_mocap_id)
    if len(ids) != 1:
        raise ValueError("Replay must identify one original drone mocap body")
    body = xml.find(f".//body[@name='{model.body(int(ids[0])).name}']")
    native_trace = {
        name: np.array(getattr(recorded, name), copy=True)
        for name in ("qpos", "qvel", "mocap_pos", "mocap_quat", "time")
    }
    native_trace["metrics"] = recorded.metrics
    metadata = visualization.add_to_model(xml, native_trace, body)
    metadata["source"] = dict(
        file=str(source), sha256=hashlib.sha256(source.read_bytes()).hexdigest()
    )
    xml_path = directory / "scene.xml"
    for name, payload in assets.items():
        if name == xml_name:
            continue
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            continue  # In-memory assets remain available under their original keys.
        _atomic_write_bytes(directory / relative, payload)
    _atomic_write_bytes(xml_path, ET.tostring(xml, encoding="utf-8"))
    _atomic_write_bytes(
        directory / "replay-visualization.json",
        (json.dumps(metadata, indent=2) + "\n").encode(),
    )
    if (source.parent / "components.json").is_file():
        shutil.copy2(
            source.parent / "components.json", directory / "components.json"
        )
    return _write_native(
        directory, xml_path, assets, native_trace, recorded.obs, recorded.reward
    )
