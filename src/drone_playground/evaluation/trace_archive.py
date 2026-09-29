"""Compact, unpadded trajectories for every independent navigation episode."""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from drone_playground.environments.scenes.navigation import DIFFICULTIES


@contextmanager
def record_native_case(env, directory, identity):
    """Persist each completed or interrupted case before batch aggregation.

    A service failure is an incomplete execution, not an arrival/collision label.
    Existing per-case records remain independent if a later worker fails.
    """
    import jax

    directory = Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    rows, error = [], None
    try:
        yield rows
    except BaseException as exception:
        error = repr(exception)
        raise
    finally:
        archive = None
        if rows:
            trace = jax.tree.map(lambda *values: np.stack(values)[:,None],*rows)
            archive = save_navigation_traces(env,{identity['difficulty']:trace},directory/'case-trace',
                                              {identity['difficulty']:[identity['scenario_id']]})
        record = dict(identity=identity,completed_execution=error is None,
                      recorded_frames=len(rows),error=error,archive=archive)
        target = directory/'case-record.json'
        temporary = target.with_suffix('.tmp')
        temporary.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
        temporary.replace(target)


def save_navigation_traces(env, traces: dict, directory: Path, scenario_groups: dict | None = None) -> dict:
    """Keep all poses/actions/events; raw sensor frames can be reconstructed.

    Frames are post-transition, as in the evaluator. The initial state and
    sensor phase are defined by the run's reset seed and scene manifest. The
    packed offsets retain episode identity without batch-padding frames.
    """
    directory.mkdir(parents=True, exist_ok=True)
    index_path = directory / "index.json"
    index = (
        json.loads(index_path.read_text())
        if index_path.exists()
        else {
            "schema_version": 1,
            "timing": "post-transition; includes terminal transition",
            "sensor_storage": "20-D proprioception; reconstruct ideal measurements from recorded poses, initial reset seed, scene and calibration",
            "cells": {},
        }
    )
    per_difficulty = env.bank.num_instances // len(DIFFICULTIES)
    for difficulty, trace in traces.items():
        active = np.asarray(trace["active"], dtype=bool)
        count = active.shape[1]
        lengths = active.sum(axis=0).astype(np.int64)
        if np.any(lengths == 0) or not np.array_equal(
            active, np.arange(active.shape[0])[:, None] < lengths[None, :]
        ):
            raise ValueError("Navigation archive requires contiguous active frames per episode")
        offsets = np.r_[0, np.cumsum(lengths)]
        fields = {
            key: np.asarray(trace[key])
            for key in ("pos", "quat", "time", "actions", "reward", "done", "outcome")
        }
        fields["proprioception"] = np.asarray(trace["obs"])[..., :20]
        fields.update(
            {"metric_" + key: np.asarray(value) for key, value in trace["metrics"].items()}
        )
        arrays = {
            key: np.concatenate([value[: lengths[case], case] for case in range(count)])
            for key, value in fields.items()
        }
        scenarios = (np.asarray(scenario_groups[difficulty], dtype=np.int32)
                     if scenario_groups is not None
                     else DIFFICULTIES.index(difficulty) * per_difficulty + np.arange(count))
        if scenarios.shape != (count,):
            raise ValueError("Each navigation trace requires its actual scenario identity")
        arrays.update(offsets=offsets, case_ids=np.arange(count), scenario_ids=scenarios)
        target = directory / (difficulty + ".npz")
        temporary = target.with_suffix(".npz.tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        temporary.replace(target)
        index["cells"][difficulty] = {
            "path": target.name,
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "episodes": count,
            "frames": int(offsets[-1]),
            "episode_steps": lengths.tolist(),
            "fields": sorted(arrays),
        }
    temporary_index = index_path.with_suffix(".json.tmp")
    temporary_index.write_text(json.dumps(index, indent=2) + "\n")
    temporary_index.replace(index_path)
    return index


def load_navigation_case(directory: Path, difficulty: str, case: int) -> tuple[dict, int]:
    """Read one exact recorded episode, checking archive integrity first."""
    directory = Path(directory)
    index = json.loads((directory / "index.json").read_text())
    cell = index["cells"][difficulty]
    path = directory / cell["path"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != cell["sha256"]:
        raise ValueError("Navigation archive digest mismatch")
    with np.load(path, allow_pickle=False) as archive:
        if not 0 <= case < len(archive["case_ids"]) or archive["case_ids"][case] != case:
            raise IndexError("Navigation archive case does not exist")
        start, stop = archive["offsets"][case : case + 2]
        trace = {
            key: archive[key][start:stop, None]
            for key in ("pos", "quat", "time", "actions", "reward", "done", "outcome")
        }
        trace["obs"] = archive["proprioception"][start:stop, None]
        trace["metrics"] = {
            key.removeprefix("metric_"): archive[key][start:stop, None]
            for key in archive.files
            if key.startswith("metric_")
        }
        scenario = int(archive["scenario_ids"][case])
    return trace, scenario
