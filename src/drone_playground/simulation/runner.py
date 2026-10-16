"""Frozen closed-loop execution and independent episode evaluation."""

import math
import time
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.spatial.transform import Rotation

from drone_playground.simulation.controllers import MellingerController
from drone_playground.simulation.methods import PlannerController
from drone_playground.simulation.scene import SceneBatch
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


def _sample(env, state, command, record_sensors=True):
    sample = {
        "time": state.time,
        "position": state.physics.states.pos[:, 0],
        "velocity": state.physics.states.vel[:, 0],
        "quaternion": state.physics.states.quat[:, 0],
        "command": command,
        "event": state.task.event,
        "gates": state.task.gates,
    }
    if record_sensors:
        points, mask = measurement_points(env, state)
        sample.update(
            sensor_points=points,
            sensor_mask=mask,
            sensor_time=(
                state.time
                if state.observation is None
                else state.observation.measurement.acquisition_time
            ),
        )
    return sample


def _init_navigation_progress(env, state):
    """Track only low-dimensional episode facts, never costly sensor replay arrays."""
    position = state.physics.states.pos[:, 0]
    goal = jnp.broadcast_to(env.goal_for(state), position.shape)
    distance = jnp.linalg.norm(goal - position, axis=-1)
    height = position[:, 2]
    return {
        "initial_distance": distance,
        "closest_distance": distance,
        "path_length": jnp.zeros_like(distance),
        "minimum_height": height,
        "maximum_height": height,
        "lower_margin": height - env.task.low[2],
        "upper_margin": env.task.high[2] - height,
    }


def _step_navigation_progress(env, progress, before, after):
    """Accumulate active-policy-step displacement and extrema without host transfers."""
    current = after.physics.states.pos[:, 0]
    previous = before.physics.states.pos[:, 0]
    goal = jnp.broadcast_to(env.goal_for(after), current.shape)
    active = ~before.done
    distance = jnp.linalg.norm(goal - current, axis=-1)
    displacement = jnp.linalg.norm(current - previous, axis=-1)
    height = current[:, 2]
    return {
        **progress,
        "closest_distance": jnp.minimum(
            progress["closest_distance"], jnp.where(active, distance, jnp.inf)
        ),
        "path_length": progress["path_length"] + jnp.where(active, displacement, 0.0),
        "minimum_height": jnp.minimum(
            progress["minimum_height"], jnp.where(active, height, jnp.inf)
        ),
        "maximum_height": jnp.maximum(
            progress["maximum_height"], jnp.where(active, height, -jnp.inf)
        ),
        "lower_margin": jnp.minimum(
            progress["lower_margin"], jnp.where(active, height - env.task.low[2], jnp.inf)
        ),
        "upper_margin": jnp.minimum(
            progress["upper_margin"], jnp.where(active, env.task.high[2] - height, jnp.inf)
        ),
    }


def _append_navigation_progress(env, episodes, state, progress):
    """Attach comparable, finite terminal facts to each original episode record."""
    goal = np.broadcast_to(np.asarray(env.goal_for(state)), (env.num_envs, 3))
    position = np.asarray(state.physics.states.pos[:, 0])
    final_distance = np.linalg.norm(goal - position, axis=-1)
    initial_distance = np.asarray(progress["initial_distance"])
    for i, row in enumerate(episodes):
        original = max(float(initial_distance[i]), 1e-6)
        final = float(final_distance[i])
        closest = float(progress["closest_distance"][i])
        height = float(position[i, 2])
        row.update(
            initial_goal_distance_metres=original,
            final_goal_distance_metres=final,
            minimum_goal_distance_metres=closest,
            normalized_final_goal_progress=(original - final) / original,
            normalized_best_goal_progress=max(0.0, (original - closest) / original),
            path_length_metres=float(progress["path_length"][i]),
            min_height_metres=float(progress["minimum_height"][i]),
            max_height_metres=float(progress["maximum_height"][i]),
            minimum_lower_boundary_margin_metres=float(progress["lower_margin"][i]),
            minimum_upper_boundary_margin_metres=float(progress["upper_margin"][i]),
            lower_height_out_of_bounds=int(
                row["event"] == Event.OUT_OF_BOUNDS.name and height <= float(env.task.low[2]) + 0.01
            ),
        )


@lru_cache(maxsize=16)
def _rollout_kernel(
    env, method, actor, chunk_steps, record, progress_diagnostics=False, record_sensors=True
):
    method_step = method.build_step(env, actor)

    if progress_diagnostics:

        def diagnostic_chunk(carry, parameters):
            def advance(carry, _):
                state, memory, progress = carry
                next_state, next_memory = method_step(state, memory, parameters)
                progress = _step_navigation_progress(env, progress, state, next_state)
                return (next_state, next_memory, progress), (
                    _sample(env, next_state, next_state.commands.applied, record_sensors)
                    if record
                    else None
                )

            return jax.lax.scan(advance, carry, None, length=chunk_steps)

        return jax.jit(diagnostic_chunk)

    def run(carry, parameters):
        def advance(carry, _):
            state, memory = method_step(*carry, parameters)
            return (state, memory), _sample(
                env, state, state.commands.applied, record_sensors
            ) if record else None

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
    progress_diagnostics=False,
    record_sensors=True,
    initial_state=None,
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
    if progress_diagnostics and env.task.name != "navigation":
        raise ValueError("Progress diagnostics only apply to Navigation")
    state = env.reset(jax.random.PRNGKey(seed)) if initial_state is None else initial_state
    memory = method.initialize(env, seed, actor)
    run_chunk = _rollout_kernel(
        env, method, actor, chunk_steps, record, progress_diagnostics, record_sensors
    )
    initial = _sample(env, state, state.commands.applied, record_sensors) if record else {}
    samples = [{key: np.asarray(value)[None] for key, value in initial.items()}]
    carry = (
        (state, memory, _init_navigation_progress(env, state))
        if progress_diagnostics
        else (state, memory)
    )
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
    episodes = episode_records(env, state, seed, label, elapsed)
    if progress_diagnostics:
        _append_navigation_progress(env, episodes, state, jax.device_get(carry[2]))
    return episodes, traces


def episode_records(env, state, seed, method, elapsed, failure=None):
    """Build one row per actual episode, independent of execution or transport."""
    episodes = []
    for index in range(env.num_envs):
        scene = env.scene.scene_for_world(index) if isinstance(env.scene, SceneBatch) else env.scene
        event = Event(int(state.task.event[index]))
        clearance = float(state.task.min_clearance[index])
        episodes.append(
            {
                "episode": index,
                "initialization_seed": seed,
                "world_index": index,
                "scene": scene.name,
                "geometry_sha256": scene.geometry_hash,
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


def slice_initial_world(batch_state, template, world_index: int, batch_size: int):
    """Select one benchmark world without resampling its initial conditions.

    The single-world environment supplies shape information so that global
    simulator arrays are not accidentally sliced when their size equals B.
    """
    if not 0 <= world_index < batch_size:
        raise ValueError("Benchmark world index is outside the initial-state batch")

    def select(batched, one):
        many_shape, one_shape = getattr(batched, "shape", ()), getattr(one, "shape", ())
        if (
            batch_size > 1
            and one_shape
            and one_shape[0] == 1
            and (many_shape == (batch_size, *one_shape[1:]))
        ):
            return batched[world_index : world_index + 1]
        if many_shape != one_shape:
            raise ValueError("Benchmark batch and single-world state layouts differ")
        return batched

    # Crazyflow's SimCore keeps batch size in PyTree metadata. Reconstruct with
    # the single-world metadata instead of mapping two incompatible treedefs.
    batched_leaves = jax.tree.leaves(batch_state)
    template_leaves, single_layout = jax.tree.flatten(template)
    if len(batched_leaves) != len(template_leaves):
        raise ValueError("Benchmark state layouts have different leaf counts")
    return jax.tree.unflatten(
        single_layout,
        [
            select(batched, one)
            for batched, one in zip(batched_leaves, template_leaves, strict=True)
        ],
    )


def rollout_host(
    env,
    method,
    *,
    seed: int,
    method_name=None,
    record=True,
    record_sensors=True,
    initial_state=None,
):
    """Run a constructed planner/controller using the common physical and task step."""
    if env.num_envs != 1 or method.execution != "host":
        raise ValueError("Host execution requires one independent world and a host method")
    state = env.reset(jax.random.PRNGKey(seed)) if initial_state is None else initial_state
    memory = method.initialize(env, seed)
    records = (
        [jax.device_get(_sample(env, state, state.commands.applied, record_sensors))]
        if record
        else [{}]
    )
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
                records.append(
                    jax.device_get(_sample(env, state, state.commands.applied, record_sensors))
                )
            plans.append(None)
            break
        if record:
            records.append(
                jax.device_get(_sample(env, state, state.commands.applied, record_sensors))
            )
        plans.append(
            None
            if trajectory is None
            else {
                "positions": trajectory.positions,
                "received_time": memory["last_plan"],
                "valid_until": trajectory.valid_until,
            }
        )
        if bool(state.done[0]):
            break
    if not bool(state.done[0]):
        raise RuntimeError("Host rollout did not reach the task deadline")
    traces = {key: np.stack([record[key] for record in records], 0) for key in records[0]}
    label = method_name or type(method.controller).__name__
    episodes = episode_records(env, state, seed, label, time.monotonic() - started, failure)
    return episodes, traces, plans, decisions
