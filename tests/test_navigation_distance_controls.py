"""Distance interventions preserve geometry and remove velocity-target incentives."""

import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hydra import compose, initialize_config_dir

from drone_playground.simulation.environment import make_environment
from drone_playground.simulation.procedural_scene import ProceduralScene, SceneDistribution


@pytest.mark.parametrize("distance", [10, 20, 30, 50, 80])
def test_exact_distance_has_safe_start_and_identical_obstacles(distance):
    """Measure radius and clearance while holding every obstacle fixed."""
    options = {
        "start_sampling": "collision_free",
        "start_goal_distance_range_m": [distance, distance],
    }
    scene = ProceduralScene(options)
    state = jax.jit(lambda: scene.sample(jax.random.key(71), 16))()
    control = ProceduralScene({**options, "start_goal_distance_range_m": [20, 20]}).sample(
        jax.random.key(71), 16
    )
    np.testing.assert_allclose(
        np.linalg.norm(state.goal - state.start, axis=-1), distance, atol=2e-5
    )
    assert (np.asarray(scene.clearance(state.start, scene_state=state)) > 0.45).all()
    assert (np.asarray(state.start) >= [0, -20, 0.5]).all()
    assert (np.asarray(state.start) <= [100, 20, 6]).all()
    np.testing.assert_allclose(state.centers, control.centers, atol=2e-6)
    np.testing.assert_array_equal(state.active, control.active)


@pytest.mark.parametrize("interval", [[0, 20], [20, 10], [10, 100], [20], [20, float("nan")]])
def test_invalid_distance_interval_is_rejected(interval):
    """Refuse inconsistent or nonphysical sampling requests."""
    with pytest.raises(ValueError, match="Distance range"):
        SceneDistribution(start_sampling="collision_free", start_goal_distance_range_m=interval)


def test_relative_goal_does_not_saturate_and_time_reward_does_not_penalize_speed():
    """A faster state has the same time/safety cost and an unsaturated goal error."""
    env = make_environment(
        task="navigation",
        scene="S01",
        sensor="state",
        device="cpu",
        num_envs=1,
        action={"level": "acceleration"},
        task_config={"reward_profile": "minimum_time_navigation"},
        observation_config={"navigation_relative_goal": True},
    )
    state = env.reset(jax.random.key(4))
    state = jax.jit(env.step)(state, jnp.zeros((1, env.action_size)))
    assert float(state.time[0]) > 0
    position = state.physics.states.pos.at[0, 0].set(jnp.array([78, 0, 3]))
    state = state.replace(
        physics=state.physics.replace(states=state.physics.states.replace(pos=position))
    )
    nearby = env.observe(state)["state"][0, 3:6]
    far = state.replace(
        physics=state.physics.replace(
            states=state.physics.states.replace(pos=position.at[0, 0, 0].set(58))
        )
    )
    np.testing.assert_allclose(env.observe(far)["state"][0, 3:6], 2 * nearby, atol=1e-7)
    cost, terms = env.task_reward_components(state, state, jnp.zeros((1, 3)))
    fast = state.replace(
        physics=state.physics.replace(
            states=state.physics.states.replace(vel=jnp.array([[[12, 0, 0]]]))
        )
    )
    fast_cost, _ = env.task_reward_components(state, fast, jnp.zeros((1, 3)))
    np.testing.assert_allclose(cost, fast_cost)
    np.testing.assert_allclose(terms["navigation_time_cost"], env.dt)


def test_distance_queue_composes_matched_speed_free_controls():
    """Actual Hydra recipes change only the distance within each sensor family."""
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "distance_scheduler", root / "tools/optimize_paper_ppo.py"
    )
    scheduler = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scheduler)
    scheduler.configure_distance_search()
    with initialize_config_dir(
        version_base=None, config_dir=str(root / "src/drone_playground/configs")
    ):
        for family in ("depth", "lidar"):
            configs = []
            for distance in (20, 80):
                options = scheduler.trial_options(
                    f"{family}_d{distance}", 0, Path("/tmp/test-distance")
                )
                config = compose(config_name="config", overrides=options)
                assert config.learning.loss is None
                assert config.task.options.reward_profile == "minimum_time_navigation"
                assert config.observation.navigation_relative_goal is True
                assert (
                    config.early_stopping.min_updates
                    * config.learning.options.horizon
                    * config.simulation.num_envs
                    == config.learning.max_interactions
                )
                assert config.screening.training_distribution_probe is True
                assert config.simulation.procedural.start_goal_distance_range_m == [
                    distance,
                    distance,
                ]
                configs.append(config)
            configs[1].simulation.procedural.start_goal_distance_range_m = [20, 20]
            assert configs[0] == configs[1]
    assert not scheduler.prune_lidar_candidate("lidar_d20", {"status": "running"})
    records = {
        f"{case}_s{seed}": {"status": "missing"}
        for case in scheduler.CASES
        for seed in range(3)
    }
    for family in ("depth", "lidar"):
        for seed in range(3):
            records[f"{family}_d20_s{seed}"]["status"] = "accepted"
    assert scheduler.next_trial(records) == ("depth_d80", 0)
