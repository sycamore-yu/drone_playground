"""Procedural Navigation training never reuses frozen Navigation8 geometry."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hydra import compose, initialize_config_dir

from drone_playground.simulation.evaluation import create_environment
from drone_playground.simulation.procedural_scene import (
    ProceduralScene,
    SceneDistribution,
)
from drone_playground.simulation.scene import Scene
from drone_playground.simulation.sensors import SensorConfig, measure

_FROZEN_GEOMETRY = {
    "S01": "b6341373ebed813d951f2bd20a19a06a05b291bef3f910310217a07a149a107a",
    "S02": "9bb7d3160847b154315d9e9f9c553b813c704f40242869b95b07766b71a7d460",
    "S03": "89a95166549ff57ca41d355e430d826ba183bfd40e35a987da264f0f35114526",
    "D01": "2510663ddf43e2ef1253b1d2667ca95365a9bf09d780d1585355fb226d16e61c",
    "D02": "0b248d4df80b6330cf17eda7c36a639ac419ed524f7170ab3d1c147ca38d6124",
    "D03": "0dbeb0a2cef49c7e67ce8a199beb4dad05108a91fef49254c8d079ecbb9f47f6",
    "S06": "7b5d906220a1ab3cc2e5bea9f699c882b5e0ae88ba21796749386f9793a69d4b",
    "D06": "eb00927dd9c3455ba63dc22880ec3741a0de2a27572e99170b5011dc58f79d0f",
}


def test_fixed_navigation8_geometry_matches_pre_existing_accepted_reports():
    """Never silently alter frozen benchmark geometry in a training refactor."""
    for name, expected_hash in _FROZEN_GEOMETRY.items():
        assert Scene(name).geometry_hash == expected_hash


def _sample(worlds=3, obstacles=12):
    scene = ProceduralScene({"max_obstacles": obstacles})
    return scene, scene.sample(jax.random.PRNGKey(19), worlds)


def test_repeated_seed_exact_and_episode_and_world_distinct():
    """Each world and each episode has a separate deterministic RNG stream."""
    scene, state = _sample()
    repeated = jax.jit(lambda: scene.sample(jax.random.PRNGKey(19), 3))()
    # XLA may fuse the eager/JIT multiply-add at float32 precision.
    np.testing.assert_allclose(state.centers, repeated.centers, atol=2e-6, rtol=1e-6)
    np.testing.assert_allclose(state.phase, repeated.phase, atol=2e-6, rtol=1e-6)
    assert not np.array_equal(state.centers[0], state.centers[1])
    renewed = jax.jit(lambda old: scene.sample(jax.random.PRNGKey(20), 3, old))(state)
    np.testing.assert_array_equal(renewed.episode, [1, 1, 1])
    assert not np.array_equal(state.centers, renewed.centers)


def test_density_families_match_navigation_benchmark_scale_not_geometry():
    """New layouts span benchmark obstacle-count bands without copying positions."""
    scene = ProceduralScene(SceneDistribution(max_obstacles=208))
    state = scene.sample(jax.random.PRNGKey(43), 16)
    counts = np.asarray(state.active.sum(axis=-1))
    assert all(
        any(abs(int(count) - baseline) <= 8 for baseline in (41, 81, 162, 50, 100, 200))
        for count in counts
    )
    moving = np.asarray((state.active & state.moving).sum(axis=-1))
    assert all(
        num == 0 or abs(num - int(total * 0.65)) <= 1
        for total, num in zip(counts, moving, strict=True)
    )
    assert not np.array_equal(state.centers[0], state.centers[1])


def test_corridor_and_start_goal_are_collision_free_for_several_seeds():
    """Constructive outside lane remains traversable independently of obstacles."""
    scene = ProceduralScene()
    for seed in (0, 1, 2):
        state = scene.sample(jax.random.PRNGKey(seed), 3)
        start_clearance = scene.clearance(state.start, scene_state=state)
        goal_clearance = scene.clearance(state.goal, scene_state=state)
        assert (np.asarray(start_clearance) > 1.0).all()
        assert (np.asarray(goal_clearance) > 1.0).all()
        lane = jnp.broadcast_to(
            jnp.stack(
                [
                    jnp.linspace(2, 98, 20),
                    jnp.full(20, scene.distribution.corridor_y),
                    jnp.full(20, 3.0),
                ],
                -1,
            ),
            (3, 20, 3),
        )
        for t in (0.0, 2.0, 12.0):
            distance = scene.clearance(lane, t, scene_state=state)
            assert (np.asarray(distance) > 1.0).all()


def test_mask_and_analytic_first_hit_equal_sensor_and_collision_geometry():
    """Masking, camera hit and contact clearance access identical cylinders."""
    scene, state = _sample(worlds=1, obstacles=3)
    centers = state.centers.at[0, 0].set(jnp.array([10.0, 0.0, 3.0]))
    sizes = state.sizes.at[0, 0].set(jnp.array([1.0, 3.0, 0.0]))
    active = jnp.zeros_like(state.active).at[0, 0].set(True)
    cylinder = state.cylinder.at[0, 0].set(True)
    moving = state.moving.at[0, 0].set(False)
    state = state.replace(
        centers=centers, sizes=sizes, active=active, cylinder=cylinder, moving=moving
    )
    ray_origin = jnp.array([[5.0, 0.0, 3.0]])
    hit = scene.raycast(ray_origin, jnp.array([[[1.0, 0.0, 0.0]]]), scene_state=state)
    np.testing.assert_allclose(hit, [[4.0]], atol=1e-5)
    clearance = scene.clearance(jnp.array([[9.0, 0.0, 3.0]]), scene_state=state)
    np.testing.assert_allclose(clearance, [0.0], atol=1e-5)
    config = SensorConfig.d435i(
        width=1,
        height=1,
        horizontal_fov_deg=60.0,
        vertical_fov_deg=40.0,
        roll_deg=0.0,
        pitch_deg=0.0,
        yaw_deg=0.0,
        min_range=0.1,
        max_range=15.0,
    )
    measurement = measure(
        scene,
        config,
        ray_origin,
        jnp.array([[0.0, 0.0, 0.0, 1.0]]),
        jnp.zeros(1),
        scene_state=state,
    )
    np.testing.assert_allclose(measurement.values, [[[4.0]]], atol=1e-5)
    np.testing.assert_array_equal(np.asarray(measurement.mask), np.array([[[True]]]))
    empty = state.replace(active=jnp.zeros_like(state.active))
    assert np.isinf(scene.clearance(jnp.array([[9.0, 0.0, 3.0]]), scene_state=empty))[0]
    np.testing.assert_allclose(
        scene.raycast(ray_origin, jnp.array([[[1.0, 0.0, 0.0]]]), scene_state=empty),
        [[40.0]],
    )


def test_moving_obstacle_uses_per_world_speed_phase_and_target():
    """Changing phase affects both depth returns and contact geometry."""
    scene, state = _sample(worlds=2, obstacles=3)
    state = state.replace(
        active=jnp.ones_like(state.active),
        moving=jnp.ones_like(state.moving),
        cylinder=jnp.zeros_like(state.cylinder),
    )
    c0 = scene._centers_at(state, jnp.zeros(2))
    c1 = scene._centers_at(state, jnp.ones(2))
    assert not np.array_equal(c0, c1)
    shifted = state.replace(phase=state.phase + 1.0)
    c_shifted = scene._centers_at(shifted, jnp.zeros(2))
    assert not np.array_equal(c0, c_shifted)


def test_procedural_and_benchmark_assets_are_disjoint():
    """No benchmark MJCF enters the training scene generator."""
    scene, state = _sample()
    assert scene.geometry_hash not in {
        Scene(name).geometry_hash for name in ("S01", "D03", "S06", "D06")
    }
    assert len(state.centers) == 3
    with pytest.raises(ValueError):
        SceneDistribution(max_obstacles=0)


@pytest.mark.parametrize(
    "experiment", ["navigation_procedural_depth", "navigation_procedural_lidar"]
)
def test_procedural_environment_reset_and_sensor_observation(experiment):
    """Depth and LiDAR use the same scene state as collision/termination."""
    from pathlib import Path

    import drone_playground

    root = Path(drone_playground.__file__).parent / "configs"
    with initialize_config_dir(version_base=None, config_dir=str(root)):
        cfg = compose(
            config_name="config",
            overrides=[
                "mode=train",
                f"experiment={experiment}",
                "simulation.num_envs=2",
                "simulation.procedural.max_obstacles=6",
                "simulation.device=cpu",
            ],
        )
    assert cfg.learning.scenes is None
    env = create_environment(cfg)
    state = env.reset(jax.random.PRNGKey(1))
    assert state.scene_state.centers.shape == (2, 6, 3)
    np.testing.assert_allclose(state.physics.states.pos[:, 0], state.scene_state.start)
    # An empty sampled forest correctly reports +inf clearance.
    assert (
        np.asarray(env.scene_clearance(state.physics.states.pos[:, 0], 0.0, state.scene_state))
        > 1.0
    ).all()
    observation = env.observe(state)
    assert ("depth" if "depth" in experiment else "points") in observation
    advanced = env.step(state, jnp.zeros((2, env.action_size)))
    assert (np.asarray(advanced.time) > 0).all()
