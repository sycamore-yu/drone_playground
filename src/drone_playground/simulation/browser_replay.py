"""Export a trusted RScope episode as a portable MuJoCo WASM browser replay.

The source `.mj_unroll` is a Python pickle produced by this project. Never load
records obtained from an untrusted source. The browser loads JSON, native MJCF
and its referenced mesh/texture assets; it does not unpickle or resimulate.
"""

from __future__ import annotations

import json
import pickle
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

FORMAT = "drone-playground-mujoco-replay-v1"


def export(directory: Path, *, max_frames: int = 350) -> Path:
    """Serialize time-aligned qpos, mocap poses, and required MJCF file names."""
    if max_frames < 2:
        raise ValueError("At least two frames are required for interactive replay")
    source = directory.resolve()
    episode = source / "episode.mj_unroll"
    xml = source / "scene.xml"
    if not episode.is_file() or not xml.is_file():
        raise FileNotFoundError("Need an exported RScope scene.xml and episode.mj_unroll")
    # The pickle is read only from our locally generated, trusted RScope result.
    replay = pickle.loads(episode.read_bytes())
    times = np.asarray(replay.time)
    qpos = np.asarray(replay.qpos)
    mocap_pos = np.asarray(replay.mocap_pos)
    mocap_quat = np.asarray(replay.mocap_quat)
    if (
        times.ndim != 2
        or times.shape[1] != 1
        or qpos.shape != (len(times), 1, 7)
        or mocap_pos.ndim != 4
        or mocap_pos.shape[:2] != (len(times), 1)
        or mocap_pos.shape[-1] != 3
        or mocap_quat.shape != (*mocap_pos.shape[:-1], 4)
        or not np.isfinite(times).all()
        or not np.isfinite(qpos).all()
        or np.any(np.diff(times[:, 0]) <= 0)
    ):
        raise ValueError("Unexpected native RScope frame or timestamp contract")
    sample = np.unique(np.linspace(0, len(times) - 1, min(max_frames, len(times)), dtype=int))
    model = ET.parse(xml).getroot()
    filenames = sorted({element.get("file") for element in model.iter() if element.get("file")})
    for name in filenames:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or not (source / name).is_file():
            raise ValueError(f"Non-portable or missing MJCF asset: {name!r}")
    manifest = {
        "format": FORMAT,
        "model": "scene.xml",
        "assets": filenames,
        "source": "locally exported RScope flight data",
        "original_frame_count": len(times),
        "frames": [
            {
                "time": float(times[i, 0]),
                "qpos": np.round(qpos[i, 0], 8).tolist(),
                "mocap_pos": np.round(mocap_pos[i, 0], 8).tolist(),
                "mocap_quat": np.round(mocap_quat[i, 0], 8).tolist(),
            }
            for i in sample
        ],
    }
    target = source / "browser_replay.json"
    temporary = source / ".browser_replay.json"
    temporary.write_text(json.dumps(manifest, separators=(",", ":")) + "\n")
    temporary.replace(target)
    return target
