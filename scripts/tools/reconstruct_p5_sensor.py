#!/usr/bin/env python3
"""Reconstruct one ideal raw sensor sample from an archived navigation episode."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("SCIPY_ARRAY_API", "1")

import jax
import jax.numpy as jnp
import numpy as np

from drone_playground.composition import build_environment
from drone_playground.evaluation.native_planners import sensor_function
from drone_playground.evaluation.trace_archive import load_navigation_case

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    parser.add_argument("difficulty", choices=("easy", "medium", "hard"))
    parser.add_argument("case", type=int)
    parser.add_argument("--tick", type=int, default=0)
    args = parser.parse_args()
    run = ROOT / "experiments" / args.run_id
    config = json.loads((run / "manifest.json").read_text())["config"]
    trace, scenario = load_navigation_case(run / "traces", args.difficulty, args.case)
    env = build_environment(
        config, "cpu", config["evaluation"]["split"], int(config["evaluation"]["episodes"])
    )
    try:
        if not 0 <= args.tick < len(trace["time"]) or args.tick % env.sensor_period:
            raise ValueError("Tick must be an actual sensor publication before episode termination")
        seed = config["evaluation"].get("seed_start")
        seed = (
            seed
            if seed is not None
            else (20000 if config["evaluation"]["split"] == "dev" else 30000)
        )
        state = env.reset(jax.random.PRNGKey(seed + args.case), jnp.int32(scenario))
        data = state.pipeline_state
        if args.tick:
            pos = jnp.asarray(trace["pos"][args.tick - 1, 0])
            quat = jnp.asarray(trace["quat"][args.tick - 1, 0])
            states = data.sim_data.states.replace(pos=pos[None, None], quat=quat[None, None])
            data = data.replace(
                sim_data=data.sim_data.replace(states=states), step_index=jnp.int32(args.tick)
            )
        stamp = args.tick * env.dt
        if config["observation"]["name"] == "navigation_depth":
            depth, camera_pos, camera_rotation = jax.tree.map(
                np.asarray, sensor_function(env, "ego")(data)
            )
            arrays = dict(
                depth=depth.reshape(env.sensor.width, env.sensor.height).T,
                camera_position=camera_pos,
                camera_rotation=camera_rotation,
            )
        else:
            frame = jax.tree.map(np.asarray, sensor_function(env, "super")(data))
            arrays = dict(
                points_world=frame.points_world,
                points_sensor=frame.points_sensor,
                valid=frame.valid,
                ranges=frame.distance,
            )
        directory = run / "sensor-reconstruction" / args.difficulty / f"case-{args.case:03d}"
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f"tick-{args.tick:04d}.npz"
        np.savez_compressed(target, **arrays)
        metadata = dict(
            run_id=args.run_id,
            difficulty=args.difficulty,
            case=args.case,
            scenario_id=scenario,
            tick=args.tick,
            simulation_time_s=stamp,
            storage="Reconstructed ideal full-resolution measurement, not a new flight",
            scene_bank_sha256=env.bank.digest(),
            sensor_calibration=env.sensor_calibration,
            sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        )
        target.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(target)
    finally:
        env.close()


if __name__ == "__main__":
    main()
