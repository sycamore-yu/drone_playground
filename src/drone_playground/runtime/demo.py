"""Full flight using Crazyflow's existing Mellinger state controller."""

from __future__ import annotations

import time
from pathlib import Path

import crazyflow  # noqa: F401
import jax
import jax.numpy as jnp
import numpy as np
from crazyflow.control import Control
from crazyflow.control.transform import motor_force2rotor_vel
from crazyflow.dynamics import load_params
from crazyflow.envs import FigureEightEnv
from crazyflow.sim import Sim
from crazyflow.sim import functional as F

from drone_playground.artifacts.record import RunRecorder
from drone_playground.visualization.rscope_io import export_rollout


def run_demo(root: Path, run_id: str, duration: float = 10.0, device: str = "cpu") -> dict:
    """Execute and record a real state-controller flight for the requested duration."""
    freq = 50
    if duration <= 0 or duration > 10:
        raise ValueError("Native figure-eight demo duration must lie in (0, 10] seconds")
    config = {
        "task": "figure8",
        "controller": "Crazyflow Mellinger state controller",
        "duration": duration,
        "dynamics": "first_principles",
        "drone": "cf2x_L250",
        "device": device,
        "freq": freq,
        "action_channels": "rotor RPM",
    }
    with RunRecorder(root, run_id, config, task_id="01") as rec:
        rec.phase("initializing", step=0)
        reference = FigureEightEnv(num_envs=1, freq=freq, device=device)
        trajectory = np.asarray(reference.trajectory)
        reference.close()
        velocity = np.gradient(trajectory, 1 / freq, axis=0)
        acceleration = np.gradient(velocity, 1 / freq, axis=0)
        frames = round(duration * freq)
        commands = np.zeros((frames, 1, 1, 16), dtype=np.float32)
        commands[:, 0, 0, :3] = trajectory[:frames]
        commands[:, 0, 0, 3:6] = velocity[:frames]
        commands[:, 0, 0, 6:9] = acceleration[:frames]
        commands[:, 0, 0, 12] = 1.0
        sim = Sim(
            control=Control.state,
            dynamics="first_principles",
            drone="cf2x_L250",
            freq=500,
            state_freq=freq,
            attitude_freq=500,
            device=device,
        )
        try:
            sim.reset()
            params = load_params("first_principles", "cf2x_L250")
            rotor = motor_force2rotor_vel(
                np.full(4, float(params["mass"]) * 9.81 / 4, np.float32),
                params["rpm2thrust"],
            )
            data = sim.data.replace(
                states=sim.data.states.replace(
                    pos=sim.data.states.pos.at[0, 0].set(trajectory[0]),
                    rotor_vel=sim.data.states.rotor_vel.at[0, 0].set(rotor),
                )
            )
            step_fn = sim.build_step_fn()

            def one_step(state, cmd):
                state = step_fn(F.state_control(state, cmd), 500 // freq)
                s = state.states
                error = jnp.linalg.norm(s.pos[0, 0] - cmd[0, 0, :3])
                obs = jnp.concatenate((s.pos[0, 0], s.quat[0, 0], s.vel[0, 0], s.ang_vel[0, 0]))
                return state, {
                    "pos": s.pos[:, 0],
                    "quat": s.quat[:, 0],
                    "time": state.core.steps[:, 0] / 500,
                    "obs": obs[None],
                    "reward": jnp.exp(-2 * error)[None],
                    "actions": state.controls.rotor_vel[:, 0],
                    "metrics": {
                        "tracking_error": error[None],
                        "reference_x": cmd[:, 0, 0],
                        "reference_y": cmd[:, 0, 1],
                        "reference_z": cmd[:, 0, 2],
                    },
                }

            rec.phase("compiling", step=0)
            started = time.monotonic()
            _final, trace = jax.jit(lambda initial, cmds: jax.lax.scan(one_step, initial, cmds))(
                data, jnp.asarray(commands)
            )
            trace = jax.tree.map(np.asarray, trace)
            rec.phase("recording", step=frames)
            for i in range(frames):
                rec.log(
                    i,
                    {
                        "flight/time_s": float(trace["time"][i, 0]),
                        "flight/tracking_error_m": float(trace["metrics"]["tracking_error"][i, 0]),
                        "flight/altitude_m": float(trace["pos"][i, 0, 2]),
                        **{
                            f"flight/rotor_{j}_rpm": float(trace["actions"][i, 0, j])
                            for j in range(4)
                        },
                    },
                )
            export_started = time.monotonic()
            record = export_rollout(sim, rec.path / "rollouts" / "native-flight", trace)
            error = trace["metrics"]["tracking_error"]
            result = {
                "frames": frames,
                "duration_s": frames / freq,
                "rmse_m": float(np.sqrt(np.mean(error**2))),
                "finite_states": bool(np.isfinite(trace["pos"]).all()),
                "record": str(record),
                "elapsed_seconds": time.monotonic() - started,
            }
            if not result["finite_states"]:
                raise FloatingPointError("Demo produced a non-finite state")
            rec.log(
                frames,
                {"record/export_seconds": time.monotonic() - export_started},
            )
            rec.finish("completed", **result)
            return result
        finally:
            sim.close()


if __name__ == "__main__":
    import sys

    from drone_playground.cli import main

    main(["demo", *sys.argv[1:]])
