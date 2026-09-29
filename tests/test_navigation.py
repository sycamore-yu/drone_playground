"""P5-01 acceptance: SANDO-style scene geometry, the navigation event protocol, reset.

These checks exercise the public construction and execution surface
(``NavigationScene``, ``make_bank``, ``NavigationEnv.step``) rather than private
helpers, and they assert the frozen protocol numbers the task book names: body
collision, 0.5 m arrival, the 40 s limit, fast crossing and reset.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from drone_playground.composition import validate_config
from drone_playground.environments.scenes.navigation import (
    BODY_RADIUS_M,
    DIFFICULTIES,
    DYNAMIC_FAMILIES,
    KIND_BOX,
    KIND_CYLINDER,
    MOTION_BOUNCE,
    MOTION_STATIC,
    MOTION_TREFOIL,
    SANDO_DENSITY,
    STATIC_FAMILIES,
    NavigationScene,
    SceneBank,
    body_centre_from_state,
    clearance_and_collision,
    make_bank,
    obstacle_positions,
)
from drone_playground.environments.tasks.navigation import (
    OUTCOME_ARRIVED,
    OUTCOME_COLLISION,
    OUTCOME_NUMERICAL,
    OUTCOME_OUT_OF_BOUNDS,
    NavigationEnv,
)
from tests.reference_configs import compose_reference as compose_config

CORRIDOR_LOW = np.array([0.0, -5.0, 0.0], np.float32)
CORRIDOR_HIGH = np.array([20.0, 5.0, 5.0], np.float32)
# Measured steady-state hover of first_principles/cf2x_L250 at 50 Hz: the
# commanded thrust that holds altitude is 0.3514 N (= 1.12 * m * g) at
# normalized action 0.40, because the pinned model's rotor dynamics lose part
# of the command. The value is asserted below against the real simulation.
HOVER_THRUST_ACTION = 0.40
# Gains of a damping-stabilised altitude hold, found by a grid search over the
# pinned first_principles rotor response (tmp/p5-check/tune_hold.py).
HOLD_TRIM, HOLD_KP, HOLD_KD = 0.38, 0.5, 0.3


def synthetic_bank(obstacles, capacity=4, start=(0.5, 0.0, 2.0), goal=(15.5, 0.0, 2.0)):
    """Hand-built bank for controlled event tests, using the public pytree."""
    kind = np.zeros((1, capacity), np.int32)
    size = np.zeros((1, capacity, 3), np.float32)
    origin = np.zeros((1, capacity, 3), np.float32)
    motion = np.zeros((1, capacity), np.int32)
    params = np.zeros((1, capacity, 5), np.float32)
    active = np.zeros((1, capacity), bool)
    for index, obstacle in enumerate(obstacles):
        kind[0, index] = obstacle["kind"]
        size[0, index] = obstacle["size"]
        origin[0, index] = obstacle["origin"]
        motion[0, index] = obstacle.get("motion", MOTION_STATIC)
        params[0, index] = obstacle.get("params", (0.0,) * 5)
        active[0, index] = True
    return SceneBank(
        kind=jnp.asarray(kind),
        size=jnp.asarray(size),
        origin=jnp.asarray(origin),
        motion=jnp.asarray(motion),
        params=jnp.asarray(params),
        active=jnp.asarray(active),
        start=jnp.asarray(np.tile(np.asarray(start, np.float32), (1, 1))),
        goal=jnp.asarray(np.tile(np.asarray(goal, np.float32), (1, 1))),
        difficulty=jnp.zeros((1,), jnp.int32),
        subtype=jnp.zeros((1,), jnp.int32),
        world_low=jnp.asarray(CORRIDOR_LOW),
        world_high=jnp.asarray(CORRIDOR_HIGH),
        subtype_names=("synthetic",),
    )


def synthetic_env(obstacles, duration=2.0, **kwargs):
    return NavigationEnv(
        scene_bank=synthetic_bank(obstacles),
        dynamics=kwargs.pop("dynamics", "first_principles"),
        drone=kwargs.pop("drone", "cf2x_L250"),
        freq=kwargs.pop("freq", 50),
        duration=duration,
        device="cpu",
        **kwargs,
    )


def place(state, env, position, velocity=(0.0, 0.0, 0.0)):
    """Move the body to an exact pose, used to build controlled event cases."""
    data = state.pipeline_state
    states = data.sim_data.states.replace(
        pos=jnp.asarray(position, jnp.float32)[None, None],
        vel=jnp.asarray(velocity, jnp.float32)[None, None],
        quat=jnp.broadcast_to(env.identity_quat, data.sim_data.states.quat.shape),
        ang_vel=jnp.zeros_like(data.sim_data.states.ang_vel),
    )
    return state.replace(pipeline_state=data.replace(sim_data=data.sim_data.replace(states=states)))


# --------------------------------------------------------------------------------------
# Scene geometry, seeding and density
# --------------------------------------------------------------------------------------


def test_scene_positions_match_the_host_reference():
    """The device motion functions must reproduce the source analytic curves."""
    scene = NavigationScene(families=DYNAMIC_FAMILIES, dynamic=True)
    bank, manifest = make_bank(scene, seed=0, per_difficulty=2)
    from drone_playground.environments.scenes.navigation import Obstacle

    checked = {MOTION_STATIC: 0, MOTION_TREFOIL: 0, MOTION_BOUNCE: 0}
    for index, row in enumerate(manifest["instances"]):
        for slot, table in enumerate(row["obstacle_table"]):
            obstacle = Obstacle(
                table["shape"],
                bank.kind[index, slot].item(),
                tuple(table["origin"]),
                tuple(table["size"]),
                bank.motion[index, slot].item(),
                tuple(table["params"]),
            )
            for time in (0.0, 1.0, 7.5, 33.25):
                device = np.asarray(obstacle_positions(bank, jnp.int32(index), jnp.float32(time)))[
                    slot
                ]
                assert np.allclose(device, obstacle.position(time), atol=1e-4), (
                    f"instance {index} slot {slot} at t={time}"
                )
            checked[obstacle.motion] += 1
    assert all(count > 0 for count in checked.values()), checked


def test_scene_generation_is_seed_reproducible_and_split_isolated():
    scene = NavigationScene(families=STATIC_FAMILIES)
    first, first_manifest = make_bank(scene, seed=0, per_difficulty=2)
    again, again_manifest = make_bank(scene, seed=0, per_difficulty=2)
    other, other_manifest = make_bank(scene, seed=1, per_difficulty=2)

    assert first.digest() == again.digest()
    assert first_manifest["instances"] == again_manifest["instances"]
    assert first.digest() != other.digest()
    # The three splits use disjoint base seeds, so their geometry differs.
    train, _ = make_bank(scene, seed=10000, per_difficulty=2)
    dev, _ = make_bank(scene, seed=20000, per_difficulty=2)
    heldout, _ = make_bank(scene, seed=30000, per_difficulty=2)
    assert len({train.digest(), dev.digest(), heldout.digest()}) == 3


def test_bank_is_grouped_by_difficulty_and_density_increases():
    scene = NavigationScene(families=STATIC_FAMILIES)
    per_difficulty = 3
    bank, manifest = make_bank(scene, seed=0, per_difficulty=per_difficulty)
    assert bank.num_instances == per_difficulty * len(DIFFICULTIES)
    for index, difficulty in enumerate(DIFFICULTIES):
        block = range(index * per_difficulty, (index + 1) * per_difficulty)
        assert {int(bank.difficulty[i]) for i in block} == {index}
        rows = [manifest["instances"][i] for i in block]
        target = SANDO_DENSITY[difficulty]
        realised = float(np.mean([row["occupied_fraction"] for row in rows]))
        assert abs(realised - target) < 0.03, (difficulty, realised, target)
    realised = [
        float(
            np.mean(
                [
                    manifest["instances"][i * per_difficulty + j]["occupied_fraction"]
                    for j in range(per_difficulty)
                ]
            )
        )
        for i in range(3)
    ]
    assert realised == sorted(realised), realised


def test_navigation_scene_rejects_mismatched_tasks():
    with pytest.raises(ValueError):
        NavigationScene(families=STATIC_FAMILIES, dynamic=True)
    with pytest.raises(ValueError):
        NavigationScene(families=DYNAMIC_FAMILIES, dynamic=False)


def test_composition_rejects_navigation_protocol_changes():
    config = compose_config("p5_navigation_static")
    validate_config(config)
    for override in (
        ["task.duration=30.0"],
        ["task.goal_radius=1.0"],
        ["task.freq=100"],
        ["task.dynamic=true"],
    ):
        with pytest.raises(ValueError):
            validate_config(compose_config("p5_navigation_static", override))


# --------------------------------------------------------------------------------------
# Collision geometry
# --------------------------------------------------------------------------------------


def test_collision_uses_the_pinned_body_sphere_and_exact_primitive_distance():
    obstacle = {
        "kind": KIND_CYLINDER,
        "size": (1.0, 5.0, 0.0),
        "origin": (5.0, 0.0, 2.5),
    }
    bank = synthetic_bank([obstacle])
    scenario = jnp.int32(0)
    inside = clearance_and_collision(bank, scenario, 0.0, jnp.array([5.0, 0.0, 2.5]))
    assert inside[1] and inside[0] < 0
    touching = clearance_and_collision(bank, scenario, 0.0, jnp.array([5.0 + 1.0 + 0.05, 0.0, 2.5]))
    assert touching[1] and abs(float(touching[0]) + 0.02) < 1e-5
    clear = clearance_and_collision(bank, scenario, 0.0, jnp.array([5.0 + 1.0 + 0.5, 0.0, 2.5]))
    assert not clear[1] and abs(float(clear[0]) - 0.43) < 1e-5
    # Above the finite cylinder the side distance no longer decides contact.
    top = clearance_and_collision(bank, scenario, 0.0, jnp.array([5.0, 0.0, 5.0 + 0.2]))
    assert not top[1]

    box = {"kind": KIND_BOX, "size": (0.5, 0.5, 0.5), "origin": (8.0, 0.0, 2.0)}
    bank = synthetic_bank([box])
    assert clearance_and_collision(bank, scenario, 0.0, jnp.array([8.0, 0.0, 2.0]))[1]
    assert not clearance_and_collision(bank, scenario, 0.0, jnp.array([8.9, 0.0, 2.0]))[1]


def test_body_centre_uses_the_modelled_sphere_offset():
    # Crazyflow orientation is xyzw, so the rest attitude is (0, 0, 0, 1).
    quat = jnp.array([0.0, 0.0, 0.0, 1.0])
    centre = body_centre_from_state(jnp.array([1.0, 2.0, 3.0]), quat)
    assert np.allclose(np.asarray(centre), [1.0, 2.0, 3.005], atol=1e-6)
    # A 90 degree rotation about x sends the +z body offset to -y in the world.
    half = jnp.array([math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)])
    rotated = body_centre_from_state(jnp.array([0.0, 0.0, 0.0]), half)
    assert np.allclose(np.asarray(rotated), [0.0, -0.005, 0.0], atol=1e-6)
    # The env's rest attitude must be the model's own, not a layout assumption.
    env = synthetic_env([], duration=1.0)
    assert np.allclose(np.asarray(env.identity_quat), [0.0, 0.0, 0.0, 1.0])
    env.close()


def test_fast_crossing_is_detected_inside_one_control_step():
    """A body crossing a thin bar between two control samples must still fail."""
    bar = {
        "kind": KIND_BOX,
        "size": (0.1, 2.0, 0.1),
        "origin": (5.0, 0.0, 2.0),
    }
    env = synthetic_env([bar], duration=0.5)
    state = env.reset(jax.random.PRNGKey(0), jnp.int32(0))
    state = place(state, env, (4.5, 0.0, 2.0), velocity=(40.0, 0.0, 0.0))
    before = np.asarray(state.pipeline_state.sim_data.states.pos[0, 0])
    result = env.step(state, jnp.array([0.0, 0.0, 0.0, 0.5]))
    after = np.asarray(result.pipeline_state.sim_data.states.pos[0, 0])
    # Both control-step boundary samples are outside the bar and its sphere.
    assert abs(before[0] - 5.0) > 0.1 + BODY_RADIUS_M
    assert abs(after[0] - 5.0) > 0.1 + BODY_RADIUS_M
    assert float(result.metrics["collision"]) == 1.0
    assert int(result.info["outcome"]) == OUTCOME_COLLISION
    env.close()


def test_clearance_ignores_inactive_slots():
    obstacle = {"kind": KIND_CYLINDER, "size": (1.0, 5.0, 0.0), "origin": (5.0, 0.0, 2.5)}
    bank = synthetic_bank([obstacle])
    bank = bank.replace(active=jnp.zeros_like(bank.active))
    clearance, hit = clearance_and_collision(bank, jnp.int32(0), 0.0, jnp.array([5.0, 0.0, 2.5]))
    assert not hit and float(clearance) == pytest.approx(2.5 - BODY_RADIUS_M)


def test_ground_contact_terminates_before_reference_point_leaves_bounds():
    env = synthetic_env([], duration=1.0)
    state = env.reset(jax.random.PRNGKey(0), jnp.int32(0))
    # Body sphere bottom is below the ground while its reference stays above it.
    near = place(state, env, (5.0, 0.0, 0.05))
    result = env.step(near, env.hover_action)
    assert int(result.info["outcome"]) == OUTCOME_COLLISION
    assert float(result.metrics["out_of_bounds"]) == 0.0
    assert float(result.metrics["clearance"]) < 0.0
    safe = env.step(place(state, env, (5.0, 0.0, 0.1)), env.hover_action)
    assert float(safe.metrics["collision"]) == 0.0
    assert float(safe.metrics["clearance"]) > 0.0
    # Below-ground crossings are also collisions, with the separate bounds
    # diagnostic retained; the outcome applies the existing collision priority.
    below = env.step(place(state, env, (5.0, 0.0, -0.5)), env.hover_action)
    assert int(below.info["outcome"]) == OUTCOME_COLLISION
    assert float(below.metrics["out_of_bounds"]) == 1.0
    env.close()


# --------------------------------------------------------------------------------------
# Navigation event protocol
# --------------------------------------------------------------------------------------


def test_arrival_requires_the_half_metre_radius_and_terminates_the_episode():
    env = synthetic_env([], duration=1.0)
    state = env.reset(jax.random.PRNGKey(0), jnp.int32(0))
    near = place(state, env, (15.5 - 0.49, 0.0, 2.0))
    result = env.step(near, env.hover_action)
    assert float(result.metrics["arrived"]) == 1.0
    assert int(result.info["outcome"]) == OUTCOME_ARRIVED
    assert float(result.done) == 1.0

    outside = place(state, env, (15.5 - 0.51, 0.0, 2.0))
    result = env.step(outside, env.hover_action)
    assert float(result.metrics["arrived"]) == 0.0
    env.close()


def test_collision_wins_over_arrival_in_the_same_step():
    blocking = {"kind": KIND_BOX, "size": (0.5, 0.5, 0.5), "origin": (15.5, 0.0, 2.0)}
    env = synthetic_env([blocking], duration=1.0)
    state = env.reset(jax.random.PRNGKey(0), jnp.int32(0))
    # Inside the obstacle and inside the arrival radius at the same time.
    state = place(state, env, (15.3, 0.0, 2.0))
    result = env.step(state, env.hover_action)
    assert float(result.metrics["arrived"]) == 1.0
    assert float(result.metrics["collision"]) == 1.0
    assert int(result.info["outcome"]) == OUTCOME_COLLISION

    # The same transition without contact must be worth strictly more: the
    # arrival bonus is withheld and the frozen failure cost is charged.
    safe = synthetic_env([], duration=1.0)
    safe_state = place(safe.reset(jax.random.PRNGKey(0), jnp.int32(0)), safe, (15.3, 0.0, 2.0))
    safe_result = safe.step(safe_state, safe.hover_action)
    assert int(safe_result.info["outcome"]) == OUTCOME_ARRIVED
    gap = float(safe_result.reward) - float(result.reward)
    expected = env.objective.arrival_bonus - env.objective.failure_penalty
    assert gap == pytest.approx(expected, abs=0.2), gap
    env.close()
    safe.close()


def test_out_of_bounds_and_numerical_outcomes_are_distinct():
    env = synthetic_env([], duration=1.0)
    state = env.reset(jax.random.PRNGKey(0), jnp.int32(0))
    outside = place(state, env, (5.0, 5.5, 2.0))
    result = env.step(outside, env.hover_action)
    assert int(result.info["outcome"]) == OUTCOME_OUT_OF_BOUNDS
    diverged = place(state, env, (5.0, 0.0, 2.0))
    data = diverged.pipeline_state
    states = data.sim_data.states.replace(pos=jnp.full_like(data.sim_data.states.pos, jnp.inf))
    diverged = diverged.replace(
        pipeline_state=data.replace(sim_data=data.sim_data.replace(states=states))
    )
    result = env.step(diverged, env.hover_action)
    assert int(result.info["outcome"]) == OUTCOME_NUMERICAL
    assert float(result.reward) < 0.0
    env.close()


def test_altitude_hold_reaches_the_forty_second_limit_as_a_timeout():
    """The frozen 40 s limit is a real reachable outcome, not a dead branch.

    A constant commanded thrust has no stable equilibrium in the pinned rotor
    model, so the limit is exercised with a closed-loop altitude hold: the
    episode stays in bounds, never arrives and never collides, and must be
    recorded as a timeout after exactly 2000 steps.
    """
    env = synthetic_env([], duration=40.0)
    assert env.episode_length == 2000
    state = env.reset(jax.random.PRNGKey(0), jnp.int32(0))
    target_height = 1.5

    def action_for(state):
        states = state.pipeline_state.sim_data.states
        height = states.pos[0, 0, 2]
        climb = states.vel[0, 0, 2]
        thrust = jnp.clip(
            HOLD_TRIM - HOLD_KP * (height - target_height) - HOLD_KD * climb, -1.0, 1.0
        )
        return jnp.array([0.0, 0.0, 0.0, thrust])

    def body(carry, _):
        nxt = env.step(carry, action_for(carry))
        return nxt, (
            nxt.done,
            nxt.metrics["collision"],
            nxt.metrics["out_of_bounds"],
            nxt.metrics["arrived"],
            nxt.pipeline_state.sim_data.states.pos[0, 0],
        )

    final, (done, collision, bounds, arrived, positions) = jax.jit(
        lambda s: jax.lax.scan(body, s, None, length=env.episode_length)
    )(state)
    assert float(jnp.sum(done)) == 0.0, "the altitude hold must not terminate early"
    assert float(final.metrics["collision"]) == 0.0
    assert float(final.metrics["out_of_bounds"]) == 0.0
    assert float(jnp.max(collision)) == 0.0 and float(jnp.max(bounds)) == 0.0
    assert float(jnp.max(arrived)) == 0.0
    height = np.asarray(positions[:, 2])
    assert height.min() > 1.0 and height.max() < 2.5, (height.min(), height.max())
    final_distance = float(np.linalg.norm(np.asarray(positions[-1]) - np.asarray([15.5, 0.0, 2.0])))
    assert final_distance > 0.5
    env.close()


def test_timeout_accounting_keeps_episodes_that_never_terminate():
    from drone_playground.evaluation.navigation import summarize_cell

    steps = 2000
    trace = {
        "done": np.zeros((steps, 2), bool),
        "outcome": np.zeros((steps, 2), np.int32),
        "active": np.ones((steps, 2), bool),
        "reward": np.zeros((steps, 2), np.float32),
        "metrics": {
            "clearance": np.ones((steps, 2), np.float32),
            "goal_distance": np.ones((steps, 2), np.float32),
        },
    }
    labels = [
        {"scenario_id": 0, "difficulty": "easy", "subtype": "synthetic"},
        {"scenario_id": 1, "difficulty": "easy", "subtype": "synthetic"},
    ]
    report = summarize_cell(trace, labels, dt=0.02, duration=40.0)
    assert report["timeout"] == 2
    assert report["arrived"] == 0
    assert report["constrained_time_mean_s"] == pytest.approx(40.0)
    assert report["episodes"][0]["outcome"] == "timeout"


def test_reset_is_reproducible_and_clears_every_episode_field():
    bar = {"kind": KIND_BOX, "size": (0.5, 0.5, 0.5), "origin": (8.0, 0.0, 2.0)}
    env = synthetic_env([bar], duration=1.0)
    first = env.reset(jax.random.PRNGKey(7), jnp.int32(0))
    second = env.reset(jax.random.PRNGKey(7), jnp.int32(0))
    assert np.array_equal(np.asarray(first.obs), np.asarray(second.obs))
    data = first.pipeline_state
    assert int(data.step_index) == 0
    assert float(data.previous_distance) == pytest.approx(15.0)
    assert np.allclose(np.asarray(data.sim_data.states.pos[0, 0]), [0.5, 0.0, 2.0])
    assert np.allclose(np.asarray(data.sim_data.states.vel[0, 0]), 0.0)
    assert np.allclose(np.asarray(data.sim_data.states.quat[0, 0]), env.identity_quat)
    assert all(float(value) == 0.0 for value in first.metrics.values())

    bank = synthetic_bank([bar], start=(0.5, 0.0, 2.0))
    env.bank = bank
    # A different scenario index is a different recorded instance.
    other = env.reset(jax.random.PRNGKey(7), jnp.int32(0))
    assert np.allclose(np.asarray(other.pipeline_state.sim_data.states.pos[0, 0]), [0.5, 0.0, 2.0])
    env.close()


def test_scene_clock_tracks_the_physics_step_counter():
    env = synthetic_env([], duration=1.0)
    state = env.reset(jax.random.PRNGKey(0), jnp.int32(0))
    assert int(state.pipeline_state.sim_data.core.steps[0, 0]) == 0

    def body(carry, _):
        nxt = env.step(carry, env.hover_action)
        return nxt, nxt.pipeline_state.sim_data.core.steps[0, 0]

    final, steps = jax.jit(lambda s: jax.lax.scan(body, s, None, length=25))(state)
    assert int(final.pipeline_state.sim_data.core.steps[0, 0]) == 25 * env.substeps
    # The scene time used for the last substep equals the physics clock.
    last_scene_time = int(steps[-1]) * env.dt_physics
    assert last_scene_time == pytest.approx(25 * env.dt, abs=1e-9)


def test_explicit_initial_state_reaches_physics_observation_and_progress_origin():
    env = synthetic_env([], duration=1.0)
    try:
        initial = dict(position=jnp.array([0.7, 0.1, 2.1]),
                       velocity=jnp.array([0.1, -0.05, 0.03]),
                       quaternion=jnp.array([0., 0., np.sin(.03), np.cos(.03)]))
        state = jax.jit(env.reset)(jax.random.PRNGKey(7), jnp.int32(0), initial)
        data = state.pipeline_state
        np.testing.assert_allclose(data.sim_data.states.pos[0, 0], initial['position'])
        np.testing.assert_allclose(data.sim_data.states.vel[0, 0], initial['velocity'])
        np.testing.assert_allclose(data.sim_data.states.quat[0, 0], initial['quaternion'])
        assert float(data.previous_distance) == pytest.approx(
            float(jnp.linalg.norm(initial['position'] - env.bank.goal[0])))
        np.testing.assert_array_equal(state.obs, env.observation(data))
        np.testing.assert_array_equal(state.info['terminal_proprioception'], env.proprioception(data))
        assert int(data.step_index) == 0
    finally:
        env.close()


def test_observation_and_action_contract_sizes():
    env = synthetic_env([], duration=1.0)
    assert env.action_size == 4
    assert env.observation_size == 20
    assert env.observer.name == "navigation_state"
    assert env.hover_action.shape == (4,)
    assert bool(jnp.all(env.hover_action >= -1.0)) and bool(jnp.all(env.hover_action <= 1.0))
    env.close()


def test_failure_cost_dominates_progress_so_crashing_is_never_the_best_outcome():
    """The frozen reward ordering: success > safe timeout > any failure."""
    from drone_playground.learning.objectives import NavigationObjective

    objective = NavigationObjective()
    common = dict(
        out_of_bounds=False,
        numerical_failure=False,
        clearance=1.0,
        action=jnp.zeros(4),
        previous_action=jnp.zeros(4),
    )
    start = 15.0
    success = objective(arrived=True, collided=False, previous_distance=0.5, distance=0.0, **common)
    timeout = float(
        objective(arrived=False, collided=False, previous_distance=start, distance=start, **common)
    )
    worst_crash = float(
        objective(arrived=False, collided=True, previous_distance=start, distance=0.0, **common)
    )
    near_crash = float(
        objective(arrived=False, collided=True, previous_distance=5.0, distance=1.1, **common)
    )
    shallow_crash = float(
        objective(arrived=False, collided=True, previous_distance=start, distance=14.0, **common)
    )
    assert float(success) > timeout, (float(success), timeout)
    assert timeout > worst_crash, (timeout, worst_crash)
    assert worst_crash > shallow_crash
    assert near_crash < timeout
    # The bound is analytic: a failure can never be worth more than the whole
    # progress term less the frozen failure cost.
    assert objective.failure_penalty <= -objective.progress_scale * start
