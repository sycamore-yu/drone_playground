"""Frozen closed-loop execution and independent episode evaluation."""

import math
import time
from functools import lru_cache
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.spatial.transform import Rotation

from drone_playground.simulation.controllers import MellingerController
from drone_playground.simulation.methods import PlannerController
from drone_playground.simulation.records import atomic_json, write_episodes
from drone_playground.simulation.replay import export_replay
from drone_playground.simulation.tasks import Event


def measurement_points(env, state, count=128):
    """Sample recorded valid sensor returns in world coordinates for replay."""
    if env.sensor is None:
        return jnp.zeros((env.num_envs, 0, 3)), jnp.zeros((env.num_envs, 0), bool)
    observation = state.observation
    points = observation.measurement.points_body.reshape(env.num_envs, -1, 3)
    mask = observation.measurement.mask.reshape(env.num_envs, -1)
    indices = jnp.linspace(0, points.shape[1] - 1, min(count, points.shape[1])).astype(jnp.int32)
    points, mask = points[:, indices], mask[:, indices]
    positions = observation.acquisition_pose[:, None, :3]
    quaternions = observation.acquisition_pose[:, None, 3:]
    world = (
        jax.vmap(lambda q, cloud: Rotation.from_quat(q).apply(cloud))(quaternions[:, 0], points)
        + positions
    )
    return world, mask


def _sample(env, state, command):
    points, mask = measurement_points(env, state)
    return {
        "time": state.time,
        "position": state.physics.states.pos[:, 0],
        "velocity": state.physics.states.vel[:, 0],
        "quaternion": state.physics.states.quat[:, 0],
        "command": command,
        "event": state.task.event,
        "gates": state.task.gates,
        "sensor_points": points,
        "sensor_mask": mask,
        "sensor_time": (
            state.time
            if state.observation is None
            else state.observation.measurement.acquisition_time
        ),
    }


@lru_cache(maxsize=16)
def _rollout_kernel(env, method, actor, chunk_steps, record):
    method_step = method.build_step(env, actor)

    def run(carry, parameters):
        def advance(carry, _):
            state, memory = method_step(*carry, parameters)
            return (state, memory), _sample(env, state, state.commands.applied) if record else None

        return jax.lax.scan(advance, carry, None, length=chunk_steps)

    return jax.jit(run)


def rollout(
    env,
    *,
    seed: int,
    method=None,
    actor=None,
    parameters=None,
    chunk_steps: int = 25,
    method_name=None,
    record=True,
) -> tuple[list[dict], dict[str, np.ndarray]]:
    """Run one complete independent episode per world through the public Environment.

    No auto-reset is performed during evaluation. The initial seed and batch index
    identify each case; every outcome appears once in the returned denominator.
    """
    if method is None:
        method = PlannerController(
            MellingerController(drone=env.sim.drone, frequency=env.method_hz)
        )
    if method.execution != "jax" or chunk_steps <= 0:
        raise ValueError("Use a constructed JAX method and positive chunk_steps")
    if method.trainable and (actor is None or parameters is None):
        raise ValueError("Frozen policy evaluation requires an actor and explicit parameters")
    state = env.reset(jax.random.PRNGKey(seed))
    memory = method.initialize(env, seed, actor)
    run_chunk = _rollout_kernel(env, method, actor, chunk_steps, record)
    initial = _sample(env, state, state.commands.applied) if record else {}
    samples = [{key: np.asarray(value)[None] for key, value in initial.items()}]
    carry = (state, memory)
    started = time.monotonic()
    for _ in range(math.ceil((env.task.duration / env.dt + 1) / chunk_steps)):
        carry, chunk = run_chunk(carry, parameters)
        if record:
            samples.append(jax.device_get(chunk))
        if np.all(np.asarray(carry[0].done)):
            break
    state = jax.device_get(carry[0])
    elapsed = time.monotonic() - started
    if not np.all(state.done):
        raise RuntimeError("Task deadline did not produce an episode event")
    traces = (
        {key: np.concatenate([sample[key] for sample in samples], 0) for key in samples[0]}
        if record
        else {}
    )
    label = method_name or ("policy" if method.trainable else type(method.controller).__name__)
    return episode_records(env, state, seed, label, elapsed), traces


def episode_records(env, state, seed, method, elapsed, failure=None):
    """Build one row per actual episode, independent of execution or transport."""
    episodes = []
    for index in range(env.num_envs):
        event = Event(int(state.task.event[index]))
        clearance = float(state.task.min_clearance[index])
        episodes.append(
            {
                "episode": index,
                "initialization_seed": seed,
                "world_index": index,
                "scene": env.scene.name,
                "geometry_sha256": env.scene.geometry_hash,
                "method": method,
                "event": event.name,
                "terminated": bool(state.terminated[index]),
                "truncated": bool(state.truncated[index]),
                "flight_seconds": float(state.time[index]),
                "physics_steps": int(state.physics.core.steps[index, 0]),
                "position_rmse": (
                    float(np.sqrt(state.task.error_sum[index] / max(1, state.task.samples[index])))
                    if env.task.name == "tracking"
                    else None
                ),
                "gates_passed": int(state.task.gates[index]),
                "gate_order": ",".join(
                    str(i + 1)
                    for i in np.asarray(env.task.gate_order)[: int(state.task.gates[index])]
                ),
                "min_clearance": clearance if np.isfinite(clearance) else None,
                "evaluation_wall_seconds": elapsed,
                "action_delay_seconds": float(state.commands.delay_ticks[index]) / env.sim.freq,
                "requested_action_delay_seconds": float(state.commands.requested_delay[index]),
                "execution": "ideal_tracking" if env.control_level == "ideal" else "crazyflow",
                "method_failure": "" if failure is None else failure["status"],
            }
        )
    return episodes


def save_evaluation(
    directory: Path,
    env,
    episodes,
    traces,
    *,
    report: dict,
    replay_episodes: int = 2,
    plans=None,
    name="",
    write_tables=True,
    record_trajectories=True,
) -> dict:
    """Persist the selected report, actual trajectories and success/failure examples."""
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if write_tables:
        write_episodes(directory / "episodes.csv", episodes)
    if record_trajectories and traces:
        np.savez_compressed(directory / "trajectories.npz", **traces)
    report = dict(report)
    report.update(
        task=env.task.name,
        task_contract=env.task.contract,
        scene=env.scene.name,
        geometry=env.scene.geometry_identity,
        method=episodes[0]["method"],
    )
    replays = []
    indices = []
    for success in (True, False):
        eligible = [
            i for i, episode in enumerate(episodes) if (episode["event"] == "SUCCESS") == success
        ]
        if eligible:
            indices.append(eligible[0])
    indices += [i for i in range(len(episodes)) if i not in indices]
    for index in indices[:replay_episodes]:
        times = traces["time"][:, index]
        keep = np.r_[True, np.diff(times) > 0]
        positions, quaternions = traces["position"][keep, index], traces["quaternion"][keep, index]
        finite = np.isfinite(positions).all(-1) & np.isfinite(quaternions).all(-1)
        frames = np.flatnonzero(keep)[finite]
        if len(frames) < 2:
            continue
        measurements = [
            {
                "points_world": traces["sensor_points"][frame, index],
                "mask": traces["sensor_mask"][frame, index],
            }
            for frame in frames
        ]
        if plans is not None:
            episode_plans = [plans[frame] for frame in frames]
        elif not episodes[index]["method"].startswith("policy"):
            episode_plans = [
                np.asarray(env.task.target(jnp.asarray(times[frame]) + jnp.linspace(0, 1, 12))[0])
                for frame in frames
            ]
        else:
            episode_plans = None
        output = export_replay(
            directory / "replays" / f"{name or env.scene.name}-{episodes[index]['episode']:04d}",
            env.scene,
            times[frames],
            traces["position"][frames, index],
            traces["quaternion"][frames, index],
            measurements,
            episode_plans,
        )
        replays.append(str(output.relative_to(directory)))
    report["replays"] = replays
    if write_tables:
        atomic_json(directory / "report.json", report)
    return report


def rollout_host(env, method, *, seed: int, method_name=None, record=True):
    """Run a constructed planner/controller using the common physical and task step."""
    if env.num_envs != 1 or method.execution != "host":
        raise ValueError("Host execution requires one independent world and a host method")
    state = env.reset(jax.random.PRNGKey(seed))
    memory = method.initialize(env, seed)
    records = [jax.device_get(_sample(env, state, state.commands.applied))] if record else [{}]
    plans, decisions = [None], []
    started, failure = time.monotonic(), None
    for _ in range(math.ceil(env.task.duration / env.dt) + 1):
        try:
            command, memory, trajectory, decision = method.host_command(env, state, memory)
            if decision is not None:
                decisions.append(decision)
            state = env.step_setpoint(state, command)
        except (RuntimeError, FloatingPointError) as error:
            failure = {
                "time": float(state.time[0]),
                "status": type(error).__name__,
                "detail": str(error),
            }
            decisions.append(failure)
            state = state.replace(
                task=state.task.replace(event=jnp.array([Event.METHOD_FAILURE], jnp.int32))
            )
            if record:
                records.append(jax.device_get(_sample(env, state, state.commands.applied)))
            plans.append(None)
            break
        if record:
            records.append(jax.device_get(_sample(env, state, state.commands.applied)))
        plans.append(None if trajectory is None else trajectory.positions)
        if bool(state.done[0]):
            break
    if not bool(state.done[0]):
        raise RuntimeError("Host rollout did not reach the task deadline")
    traces = {key: np.stack([record[key] for record in records], 0) for key in records[0]}
    label = method_name or type(method.controller).__name__
    episodes = episode_records(env, state, seed, label, time.monotonic() - started, failure)
    return episodes, traces, plans, decisions
