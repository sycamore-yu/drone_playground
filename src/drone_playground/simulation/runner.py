"""Frozen closed-loop execution and independent episode evaluation."""

import math
import time
from functools import lru_cache
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.spatial.transform import Rotation

from drone_playground.simulation.records import atomic_json, write_episodes
from drone_playground.simulation.replay import export_replay
from drone_playground.simulation.tasks import Event


def measurement_points(env, state, count=128):
    """Sample recorded valid sensor returns in world coordinates for replay."""
    if env.sensor is None:
        return jnp.zeros((env.num_envs, 0, 3)), jnp.zeros((env.num_envs, 0), bool)
    observation = state.observation
    points = observation.points_at_completion.reshape(env.num_envs, -1, 3)
    mask = observation.measurement.mask.reshape(env.num_envs, -1)
    indices = jnp.linspace(0, points.shape[1] - 1, min(count, points.shape[1])).astype(jnp.int32)
    points, mask = points[:, indices], mask[:, indices]
    positions, quaternions = env.sensor._pose_at(
        observation.pose_history, observation.measurement.acquisition_time[:, None]
    )
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
def _rollout_kernel(env, method, actor, chunk_steps):
    def run(carry, parameters):
        def advance(carry, _):
            state, memory, integral = carry
            if method == "policy":
                raw, memory, _ = actor.apply(parameters, env.observe(state), memory)
                action = jnp.tanh(raw)
                command = env.action_setpoint(state, action).value
                state = env.step(state, action)
                memory = jnp.where(state.done[:, None], 0.0, memory)
            else:
                reference = env.task.target(state.time)
                setpoint, integral = env.controller(state.physics, reference, integral)
                command = setpoint.value
                state = env.step_setpoint(state, setpoint)
            return (state, memory, integral), _sample(env, state, command)

        return jax.lax.scan(advance, carry, None, length=chunk_steps)

    return jax.jit(run)


def rollout(
    env, *, seed: int, method: str = "mellinger", actor=None, parameters=None, chunk_steps: int = 25
) -> tuple[list[dict], dict[str, np.ndarray]]:
    """Run one complete independent episode per world through the public Environment.

    No auto-reset is performed during evaluation. The initial seed and batch index
    identify each case; every outcome appears once in the returned denominator.
    """
    if method not in {"mellinger", "policy"} or chunk_steps <= 0:
        raise ValueError("rollout supports mellinger/policy and positive chunk_steps")
    if method == "policy" and (actor is None or parameters is None):
        raise ValueError("Frozen policy evaluation requires an actor and explicit parameters")
    state = env.reset(jax.random.PRNGKey(seed))
    memory = jnp.zeros((env.num_envs, 192))
    integral = jnp.zeros((env.num_envs, 3))
    run_chunk = _rollout_kernel(env, method, actor, chunk_steps)
    samples = [_sample(env, state, jnp.zeros((env.num_envs, 4)))]
    samples = [{key: np.asarray(value)[None] for key, value in samples[0].items()}]
    carry = (state, memory, integral)
    started = time.monotonic()
    for _ in range(math.ceil((env.task.duration / env.dt + 1) / chunk_steps)):
        carry, chunk = run_chunk(carry, parameters)
        chunk = jax.device_get(chunk)
        samples.append(chunk)
        if np.all(np.asarray(carry[0].done)):
            break
    state = jax.device_get(carry[0])
    elapsed = time.monotonic() - started
    if not np.all(state.done):
        raise RuntimeError("Task deadline did not produce an episode event")
    traces = {key: np.concatenate([sample[key] for sample in samples], 0) for key in samples[0]}
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
                    str(i) for i in [1, 2, 3, 4, 2][: int(state.task.gates[index])]
                ),
                "min_clearance": clearance if np.isfinite(clearance) else None,
                "evaluation_wall_seconds": elapsed,
                "action_delay_seconds": 0.0,
            }
        )
    return episodes, traces


def save_evaluation(
    directory: Path, env, episodes, traces, *, report: dict, replay_episodes: int = 2, plans=None
) -> dict:
    """Persist the selected report, actual trajectories and success/failure examples."""
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    write_episodes(directory / "episodes.csv", episodes)
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
        elif episodes[index]["method"] == "mellinger":
            episode_plans = [
                np.asarray(env.task.target(jnp.asarray(times[frame]) + jnp.linspace(0, 1, 12))[0])
                for frame in frames
            ]
        else:
            episode_plans = None
        output = export_replay(
            directory / "rollouts" / f"episode-{index:04d}",
            env.scene,
            times[frames],
            traces["position"][frames, index],
            traces["quaternion"][frames, index],
            measurements,
            episode_plans,
        )
        replays.append(str(output.relative_to(directory)))
    report["replays"] = replays
    atomic_json(directory / "report.json", report)
    return report


@lru_cache(maxsize=8)
def _ros_kernel(env):
    @jax.jit
    def advance(current, reference, memory):
        setpoint, memory = env.controller(current.physics, reference, memory)
        current = env.step_setpoint(current, setpoint)
        return current, memory, _sample(env, current, setpoint.value)

    return advance, jax.jit(env.sensor._pose_at)


def rollout_ros(env, planner, *, seed: int, planner_name: str):
    """Run a genuine host planner with sensor-only map ingress and Crazyflow execution."""
    from scipy.spatial.transform import Rotation as HostRotation

    from drone_playground.simulation.ros_planner import (
        RosPlannerError,
        cloud_measurement,
        depth_measurement,
    )

    if env.num_envs != 1 or env.task.name != "navigation" or env.sensor is None:
        raise ValueError("Native rollout requires one Navigation world and a configured sensor")
    state = env.reset(jax.random.PRNGKey(seed))
    planner.reset(seed)
    integral = jnp.zeros((1, 3))
    records = [jax.device_get(_sample(env, state, jnp.zeros((1, 4))))]
    plans, decisions = [None], []
    previous_velocity = np.asarray(state.physics.states.vel[0, 0])

    advance, sensor_pose = _ros_kernel(env)
    started = time.monotonic()
    failure = None
    trajectory = None
    last_decision = -math.inf
    config = env.sensor.config
    for _ in range(math.ceil(env.task.duration / env.dt) + 1):
        timestamp = int(state.physics.core.steps[0, 0]) / env.sim.freq
        position = np.asarray(state.physics.states.pos[0, 0])
        quaternion = np.asarray(state.physics.states.quat[0, 0])
        velocity = np.asarray(state.physics.states.vel[0, 0])
        ready = config.profile == "d435i" or int(state.observation.frame[0]) > 0
        try:
            if ready and timestamp - last_decision + 1e-9 >= planner.replan_interval:
                measurement = jax.device_get(state.observation.measurement)
                acquisition_time = int(state.observation.frame[0]) / config.frequency_hz
                sensor_position, sensor_quaternion = sensor_pose(
                    state.observation.pose_history, jnp.array([[acquisition_time]])
                )
                sensor_position = np.asarray(sensor_position[0, 0])
                sensor_rotation = HostRotation.from_quat(np.asarray(sensor_quaternion[0, 0]))
                if config.profile == "d435i":
                    mounting = HostRotation.from_euler(
                        "xyz", [config.roll_deg, -config.pitch_deg, config.yaw_deg], degrees=True
                    )
                    optical_to_body = HostRotation.from_matrix([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
                    optical_rotation = sensor_rotation * mounting * optical_to_body
                    sensor_message = depth_measurement(
                        measurement.values[0],
                        acquisition_time,
                        fx=config.width / (2 * np.tan(np.deg2rad(config.horizontal_fov_deg) / 2)),
                        fy=config.height / (2 * np.tan(np.deg2rad(config.vertical_fov_deg) / 2)),
                        cx=(config.width - 1) / 2,
                        cy=(config.height - 1) / 2,
                        position=sensor_position + sensor_rotation.apply(config.translation),
                        quaternion=np.roll(optical_rotation.as_quat(), 1),
                        available_time=acquisition_time + config.latency,
                    )
                else:
                    points = np.asarray(state.observation.points_at_completion[0])
                    sensor_message = cloud_measurement(
                        points[np.asarray(measurement.mask[0])],
                        acquisition_time,
                        position=sensor_position,
                        quaternion=np.roll(sensor_rotation.as_quat(), 1),
                        available_time=acquisition_time + config.latency,
                    )
                observation = {
                    "position": position,
                    "quaternion": np.roll(quaternion, 1),
                    "velocity": velocity,
                    "acceleration": (velocity - previous_velocity) / env.dt,
                    "goal": np.asarray(env.task.goal),
                    "measurement": sensor_message,
                }
                trajectory = planner.plan(observation, timestamp, force_replan=True)
                last_decision = timestamp
                decisions.append(
                    {
                        "sequence": trajectory.sequence,
                        "time": timestamp,
                        "status": trajectory.status,
                        "upstream_status": trajectory.upstream_status,
                        "valid_until": trajectory.valid_until,
                        "sensor_time": acquisition_time,
                        **trajectory.timings,
                    }
                )
            if trajectory is not None:
                reference = tuple(
                    jnp.asarray(value)[None] for value in trajectory.sample(timestamp)
                )
            else:
                reference = (state.physics.states.pos[:, 0], jnp.zeros((1, 3)), jnp.zeros((1, 3)))
        except RosPlannerError as error:
            failure = {"time": timestamp, "status": error.status, "detail": str(error)}
            decisions.append(failure)
            state = state.replace(
                task=state.task.replace(event=jnp.array([Event.METHOD_FAILURE], jnp.int32))
            )
            break
        previous_velocity = velocity
        state, integral, sample = advance(state, reference, integral)
        records.append(jax.device_get(sample))
        plans.append(None if trajectory is None else trajectory.positions)
        if bool(state.done[0]):
            break
    event = Event(int(state.task.event[0]))
    if event == Event.RUNNING:
        raise RuntimeError("Native rollout did not reach the task deadline")
    traces = {key: np.stack([record[key] for record in records], 0) for key in records[0]}
    traces["event"][-1, 0] = int(event)
    clearance = float(state.task.min_clearance[0])
    episodes = [
        {
            "episode": 0,
            "initialization_seed": seed,
            "world_index": 0,
            "scene": env.scene.name,
            "geometry_sha256": env.scene.geometry_hash,
            "method": planner_name,
            "event": event.name,
            "terminated": bool(state.terminated[0]),
            "truncated": bool(state.truncated[0]),
            "flight_seconds": float(state.time[0]),
            "physics_steps": int(state.physics.core.steps[0, 0]),
            "position_rmse": None,
            "gates_passed": 0,
            "gate_order": "",
            "min_clearance": clearance if np.isfinite(clearance) else None,
            "evaluation_wall_seconds": time.monotonic() - started,
            "action_delay_seconds": 0.0,
            "method_failure": "" if failure is None else failure["status"],
        }
    ]
    return episodes, traces, plans, decisions
