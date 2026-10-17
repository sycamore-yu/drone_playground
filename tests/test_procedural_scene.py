"""Procedural Navigation training never reuses frozen Navigation8 geometry."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hydra import compose, initialize_config_dir

from drone_playground.simulation.factory import create_environment
from drone_playground.simulation.procedural_scene import (
    ProceduralScene,
    SceneDistribution,
    distance_sampling_weights,
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


def test_matched_crossbeam_training_samples_new_vertical_passages():
    """New static beams vary by reset without changing unaffected scene obstacles."""
    key = jax.random.key(601)
    options = {"max_obstacles": 36, "obstacle_motion": "static", "layout": "matched"}
    baseline = ProceduralScene(options)
    varied = ProceduralScene({**options, "horizontal_beam_probability": 0.7})
    before = baseline.sample(key, 48)
    after = varied.sample(key, 48)
    assert baseline.geometry_hash != varied.geometry_hash
    for field in ("start", "goal", "route"):
        np.testing.assert_array_equal(getattr(before, field), getattr(after, field))
    for field in ("centers", "sizes", "active", "cylinder", "moving", "motion"):
        np.testing.assert_array_equal(getattr(before, field)[:, 2:], getattr(after, field)[:, 2:])
    beams = np.asarray(np.any(before.sizes[:, :2] != after.sizes[:, :2], axis=-1))
    assert 35 <= int(beams.sum()) < 96
    centers, sizes = np.asarray(after.centers), np.asarray(after.sizes)
    for world, slot in np.argwhere(beams):
        center, half_extent = centers[world, slot], sizes[world, slot]
        assert 12 <= center[0] <= 88 and center[1] == 0
        assert 2.35 <= center[2] <= 3.05
        assert 0.25 <= half_extent[0] <= 0.55
        assert 15.5 <= half_extent[1] <= 17.0
        assert 0.45 <= half_extent[2] <= 0.75
        assert center[2] + half_extent[2] < 4.3
        assert abs(after.start[world, 0] - center[0]) > half_extent[0] + 4
        assert not bool(after.cylinder[world, slot])
        assert not bool(after.moving[world, slot])
        assert bool(after.active[world, slot])
    assert np.min(np.asarray(varied.clearance(after.start, scene_state=after))) > 0.45
    assert np.min(np.asarray(varied.clearance(after.goal, scene_state=after))) > 0.45
    np.testing.assert_allclose(
        after.centers, jax.jit(lambda: varied.sample(key, 48))().centers, atol=1e-5, rtol=1e-6
    )
    assert Scene("S06").geometry_hash == _FROZEN_GEOMETRY["S06"]


@pytest.mark.parametrize("invalid", [-0.1, 1.01, "0.5", True, float("nan")])
def test_horizontal_beam_probability_requires_numeric_supported_static_layout(invalid):
    """Reject invalid beam sampling contracts at construction time."""
    with pytest.raises(ValueError, match="Horizontal beam probability"):
        SceneDistribution(
            layout="matched", obstacle_motion="static", horizontal_beam_probability=invalid
        )
    with pytest.raises(ValueError, match="Horizontal beam probability"):
        SceneDistribution(
            layout="legacy", obstacle_motion="static", horizontal_beam_probability=0.7
        )
    with pytest.raises(ValueError, match="Horizontal beam probability"):
        SceneDistribution(
            layout="matched", obstacle_motion="dynamic", horizontal_beam_probability=0.7
        )


def test_zero_beam_probability_preserves_matched_procedural_source_identity():
    """Old frozen matched source and generated arrays remain identical by default."""
    options = {"max_obstacles": 36, "layout": "matched", "obstacle_motion": "static"}
    old = ProceduralScene(options)
    new = ProceduralScene({**options, "horizontal_beam_probability": 0.0})
    assert old.geometry_hash == new.geometry_hash
    first, second = (scene.sample(jax.random.key(13), 6) for scene in (old, new))
    for x, y in zip(jax.tree.leaves(first), jax.tree.leaves(second), strict=True):
        np.testing.assert_array_equal(x, y)


def test_collision_free_start_sampler_provides_goal_distance_curriculum():
    """Shorter episodes vary distance without changing frozen evaluation geometry."""
    scene = ProceduralScene({"max_obstacles": 36, "start_sampling": "collision_free"})
    a = scene.sample(jax.random.PRNGKey(21), 12)
    b = scene.sample(jax.random.PRNGKey(21), 12)
    np.testing.assert_allclose(a.start, b.start, rtol=0.0, atol=2e-6)
    clearance = scene.clearance(a.start, 0.0, scene_state=a)
    assert (np.asarray(clearance) > 0.45).all()
    dx = np.asarray(a.goal[:, 0] - a.start[:, 0])
    assert (dx > 4.0).all()
    assert (dx < 99.0).all()
    assert (dx < 35.0).any()  # Some episodes begin near their terminal goal.


def test_stratified_start_sampler_covers_nominal_start_and_rotates_cohorts():
    """Every batch covers near-start, uniform and near-goal initial positions."""
    scene = ProceduralScene({"max_obstacles": 24, "start_sampling": "stratified"})
    first = scene.sample(jax.random.PRNGKey(21), 12)
    repeated = scene.sample(jax.random.PRNGKey(21), 12)
    np.testing.assert_allclose(first.start, repeated.start, atol=2e-6)
    near_start = np.asarray(first.start)[::3]
    assert np.all((near_start[:, 0] >= 1.8) & (near_start[:, 0] <= 5.5))
    assert np.all(np.abs(near_start[:, 1]) <= 2.5)
    assert np.all(np.asarray(scene.clearance(first.start, scene_state=first)) > 0.45)
    second = scene.sample(jax.random.PRNGKey(21), 12, prior=first)
    assert np.all(np.asarray(second.episode) == 1)
    rotated_start = np.asarray(second.start)[2::3]
    assert np.all((rotated_start[:, 0] >= 1.8) & (rotated_start[:, 0] <= 5.5))
    assert scene.geometry_hash != ProceduralScene({"max_obstacles": 24}).geometry_hash


def test_stratified_start_executes_native_crazyflow_step_under_jit():
    """Keep the scene sampler compatible with the normal compiled Environment."""
    from drone_playground.simulation.environment import make_environment

    env = make_environment(
        task="navigation",
        scene="train/procedural",
        num_envs=2,
        device="cpu",
        sensor="depth",
        procedural={"max_obstacles": 6, "start_sampling": "stratified"},
    )
    before = env.reset(jax.random.key(7))
    np.testing.assert_allclose(before.physics.states.pos[:, 0], before.scene_state.start)
    after = env.step(before, jnp.zeros((2, env.action_size)))
    assert np.isfinite(np.asarray(after.physics.states.pos)).all()


def test_distance_curriculum_rehearses_old_distances_and_has_stationary_probe():
    """Only training start proportions change; probes and obstacle RNG stay paired."""
    phases = np.asarray([distance_sampling_weights(p) for p in (0.0, 1 / 3, 2 / 3, 1.0)])
    np.testing.assert_allclose(phases[:3].mean(axis=0), distance_sampling_weights(), atol=1e-7)
    assert np.all(phases > 0)
    assert phases[0, 0] > phases[-1, 0] and phases[0, 2] < phases[-1, 2]
    control = ProceduralScene({"max_obstacles": 12, "start_sampling": "distance_mixture"})
    treatment = ProceduralScene({"max_obstacles": 12, "start_sampling": "distance_curriculum"})
    key = jax.random.key(29)
    baseline = control.sample(key, 128)
    probe = treatment.sample(key, 128)
    np.testing.assert_array_equal(baseline.start, probe.start)
    early = treatment.sample(key, 128, sampling_progress=0.0)
    late = jax.jit(lambda p: treatment.sample(key, 128, sampling_progress=p))(jnp.array(1.0))
    for state in (early, late):
        distances = np.linalg.norm(np.asarray(state.goal - state.start), axis=-1)
        assert np.max(np.min(np.abs(distances[:, None] - [20.0, 50.0, 80.0]), axis=1)) < 1e-4
        assert np.all(np.asarray(treatment.clearance(state.start, scene_state=state)) > 0.45)
        np.testing.assert_allclose(state.centers, baseline.centers, atol=1e-5, rtol=1e-6)
    assert np.mean(np.linalg.norm(np.asarray(early.goal - early.start), axis=-1)) < 35
    assert np.mean(np.linalg.norm(np.asarray(late.goal - late.start), axis=-1)) > 50


@pytest.mark.parametrize("algorithm", ["ppo", "bptt", "shac"])
def test_curriculum_progress_survives_host_counter_reset_and_world_masks(algorithm):
    """Each real algorithm uses the supplied phase and preserves live worlds."""
    from drone_playground.learning.trainer import Trainer
    from drone_playground.simulation.environment import make_environment

    env = make_environment(
        task="navigation",
        scene="train/procedural",
        num_envs=2,
        device="cpu",
        duration=0.02,
        procedural={"max_obstacles": 6, "start_sampling": "distance_curriculum"},
    )
    trainer = Trainer(
        env,
        algorithm=algorithm,
        config={"horizon": 1, "ppo_epochs": 1, "minibatches": 1, "critic_epochs": 1},
    )
    before = trainer.initialize().replace(updates=1000, interactions=2000)
    with pytest.raises(ValueError, match="explicit finite progress"):
        trainer.update(before)
    after, metrics = trainer.update(before, sampling_progress=0.8)
    assert after.updates == 1001 and after.interactions == 2002
    assert metrics["distance_curriculum_phase"] == 2
    assert sum(float(metrics[f"reset_distance_{d}m_count"]) for d in (20, 50, 80)) == 2
    assert all(np.isfinite(v) for v in metrics.values())
    masked = env.reset(
        jax.random.key(99), before.env_state, mask=jnp.array([True, False]), sampling_progress=1.0
    )
    np.testing.assert_array_equal(
        masked.scene_state.start[1], before.env_state.scene_state.start[1]
    )
    assert masked.scene_state.episode[0] == before.env_state.scene_state.episode[0] + 1


@pytest.mark.parametrize("kind", ["depth", "lidar"])
def test_fixed_validation_cohort_and_stratified_start_experiments_are_paired(kind):
    """Control and intervention differ only in the train-only start distribution."""
    from pathlib import Path

    import drone_playground

    root = Path(drone_playground.__file__).parent / "configs"
    with initialize_config_dir(version_base=None, config_dir=str(root)):
        control = compose(
            config_name="config",
            overrides=[f"experiment=navigation_procedural_{kind}_ppo_collision_free"],
        )
        candidate = compose(
            config_name="config",
            overrides=[f"experiment=navigation_procedural_{kind}_ppo_stratified"],
        )
    assert control.scene.name == candidate.scene.name == "train/procedural"
    assert control.learning.algorithm == candidate.learning.algorithm == "ppo"
    assert control.simulation.procedural.start_sampling == "collision_free"
    assert candidate.simulation.procedural.start_sampling == "stratified"
    control.simulation.procedural.start_sampling = "stratified"
    assert control == candidate


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
    assert cfg.scene.name == "train/procedural"
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


def test_static_and_dynamic_training_distributions_are_independent():
    """Static OA has no moving geometry; Dynamic OA samples moving obstacles."""
    static = ProceduralScene({"obstacle_motion": "static", "max_obstacles": 8})
    dynamic = ProceduralScene({"obstacle_motion": "dynamic", "max_obstacles": 8})
    key = jax.random.key(4)
    first, second = static.sample(key, 4), dynamic.sample(key, 4)
    assert not np.asarray(first.moving).any()
    assert np.asarray(second.moving & second.active).any(axis=1).all()
    assert static.geometry_hash != dynamic.geometry_hash
    np.testing.assert_array_equal(first.centers, static.sample(key, 4).centers)
    np.testing.assert_array_equal(second.centers, dynamic.sample(key, 4).centers)
