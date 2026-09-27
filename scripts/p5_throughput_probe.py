"""P5 throughput probe: measure, do not guess, the cost of one environment step.

The P5 spec requires an actual sensor-plus-environment throughput measurement
before any formal training budget is committed. This script measures the raw
environment step (dynamics plus scene plus sensor) at 1, 16 and 128 parallel
environments, decomposes the cost into control/physics and sensor terms, and
projects the wall clock of the frozen 8,388,608-interaction unit budget.

Usage:
    pixi run python scripts/p5_throughput_probe.py --output docs/verification/p5-throughput.json
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from drone_playground.composition import build_environment, compose_config

BATCHES = (1, 16, 128)
INTERACTIONS_PER_UNIT = 8388608
REPEATS = 30


def _git_revision(root: Path) -> str:
    return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()


def measure(experiment: str, batch: int, device: str, root: Path) -> dict:
    config = compose_config(experiment)
    build_start = time.monotonic()
    env = build_environment(config, device, "train", 4)
    build_seconds = time.monotonic() - build_start
    try:
        keys = jax.vmap(jax.random.PRNGKey)(jnp.arange(batch))
        ids = jnp.arange(batch) % env.bank.num_instances
        state = jax.vmap(env.reset)(keys, ids)
        action = jnp.broadcast_to(env.hover_action, (batch, env.action_size))

        def rollout(state, length):
            def body(carry, _):
                nxt = jax.vmap(env.step)(carry, action)
                return nxt, nxt.reward

            return jax.lax.scan(body, state, None, length=length)

        length = 8
        compiled = jax.jit(lambda s: rollout(s, length))
        tic = time.monotonic()
        final, rewards = compiled(state)
        jax.block_until_ready(rewards)
        compile_seconds = time.monotonic() - tic
        del final

        timings = []
        for _ in range(REPEATS):
            tic = time.monotonic()
            _, rewards = compiled(state)
            jax.block_until_ready(rewards)
            timings.append((time.monotonic() - tic) / length)
        per_step = float(np.median(timings))
        interactions_per_second = batch / per_step

        sensor_seconds = float("nan")
        if env.sensor is not None:
            from drone_playground.tasks.sensors.depth import cast_depth
            from drone_playground.tasks.sensors.lidar import Mid360Lidar, cast_lidar

            state_d = state.pipeline_state
            positions = state_d.sim_data.states.pos[:, 0, 0]
            quats = state_d.sim_data.states.quat[:, 0, 0]
            ids_d = state_d.scenario_id
            windows = state_d.sensor_sequence

            def sensor_only(position, quat, scenario, window, fresh):
                del fresh
                if isinstance(env.sensor, Mid360Lidar):
                    frame = cast_lidar(
                        env.sensor,
                        env.bank,
                        scenario,
                        position,
                        quat,
                        jnp.float32(0.0),
                        window,
                    )
                    return frame.distance
                return cast_depth(
                    env.sensor, env.bank, scenario, position, quat, jnp.float32(0.0)
                ).depth

            compiled_sensor = jax.jit(jax.vmap(sensor_only))
            operands = (positions, quats, ids_d, windows, jnp.arange(batch))
            tic = time.monotonic()
            out = compiled_sensor(*operands)
            jax.block_until_ready(out)
            sensor_timings = []
            for _ in range(REPEATS):
                tic = time.monotonic()
                out = compiled_sensor(*operands)
                jax.block_until_ready(out)
                sensor_timings.append(time.monotonic() - tic)
            sensor_seconds = float(np.median(sensor_timings))
        else:
            sensor_timings = []

        return {
            "experiment": experiment,
            "batch": batch,
            "device": device,
            "scene_instances": env.bank.num_instances,
            "scene_capacity": env.bank.capacity,
            "observation_size": env.observation_size,
            "episode_length": env.episode_length,
            "sensor": None if env.sensor is None else env.sensor.calibration_id,
            "realised_sensor_rate_hz": None
            if env.sensor is None
            else env.realised_sensor_rate_hz,
            "points_per_frame": env.points_per_frame,
            "env_build_seconds": build_seconds,
            "compile_seconds": compile_seconds,
            "milliseconds_per_env_step": per_step * 1e3,
            "interactions_per_second": interactions_per_second,
            "milliseconds_per_sensor_frame": sensor_seconds * 1e3,
            "sensor_share_of_step": (
                None if env.sensor is None else sensor_seconds / per_step
            ),
            "projected_unit_wall_seconds": INTERACTIONS_PER_UNIT / interactions_per_second,
        }
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="gpu")
    parser.add_argument(
        "--experiments",
        default="p5_navigation_static,p5_static_depth_ppo",
        help="comma separated experiment names",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    experiments = [name for name in args.experiments.split(",") if name]
    report = {
        "kind": "p5-throughput-probe",
        "git_revision": _git_revision(root),
        "jax_version": jax.__version__,
        "devices": [str(device) for device in jax.devices()],
        "platform": platform.platform(),
        "interactions_per_unit": INTERACTIONS_PER_UNIT,
        "batches": list(BATCHES),
        "measurements": [],
    }
    for experiment in experiments:
        for batch in BATCHES:
            row = measure(experiment, batch, args.device, root)
            report["measurements"].append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
