"""Behavior at the public configuration, physics and evaluation interfaces."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hydra.errors import InstantiationException

from drone_playground.composition import build_environment, compose_experiment, validate_config


def test_navigation_without_benchmark_accepts_custom_task_conditions():
    config = compose_experiment('navigation/ppo', "navigation/static")
    config["evaluation"].pop("protocol", None)
    config["env"]["task"].update(freq=25, duration=2.0, goal_radius=0.3)
    validate_config(config)


def test_task_frequency_changes_actual_physical_elapsed_time():
    config = compose_experiment('control/ppo', "hovering", ["env.task.freq=25"])
    env = build_environment(config, device="cpu", role="eval", count=1)
    try:
        state = env.reset(jax.random.PRNGKey(11))
        advanced = env.step(state, env.hover_action)
        ticks = np.asarray(advanced.pipeline_state.sim_data.core.steps - state.pipeline_state.sim_data.core.steps)
        np.testing.assert_array_equal(ticks, 20)
        assert env.dt == 0.04
        assert "frequency_hz" not in config["env"]["action"]
    finally:
        env.close()


def test_duplicate_execution_frequency_is_rejected_instead_of_ignored():
    config = compose_experiment('control/ppo', "hovering")
    config["env"]["action"]["frequency_hz"] = 25
    with pytest.raises(ValueError, match=r"frequency_hz|env.task.freq"):
        validate_config(config)


def test_navigation_rejects_a_physics_clock_the_backend_does_not_execute():
    config = compose_experiment('navigation/ppo', "navigation/static")
    config["evaluation"]["protocol"] = None
    config["env"]["task"]["physics_freq"] = 1000
    with pytest.raises(ValueError, match=r"physics frequency.*actual"):
        build_environment(config, device="cpu", role="eval", count=1)


@pytest.mark.parametrize("bounds", [[float("nan"), 1.1], [0.9, float("inf")]])
def test_randomization_rejects_nonfinite_physical_ranges(bounds):
    config = compose_experiment('control/ppo', "hovering")
    config["training"]["domain_randomization"] = {"enabled": True, "dynamics": {"mass": bounds}}
    with pytest.raises((ValueError, InstantiationException), match=r"finite|randomization"):
        build_environment(config, device="cpu", role="train", count=1)


def test_mass_randomization_changes_motion_and_reproduces_per_reset():
    config = compose_experiment('control/ppo', "hovering")
    config["training"]["domain_randomization"] = {"enabled": True, "dynamics": {"mass": [1.5, 2.0]}}
    nominal = build_environment(config, device="cpu", role="eval", count=1)
    randomized = build_environment(config, device="cpu", role="train", count=1)
    try:
        key = jax.random.PRNGKey(11)
        state = randomized.reset(key)
        again = randomized.reset(key)
        other = randomized.reset(jax.random.PRNGKey(12))
        params = state.pipeline_state.sim_data.params
        np.testing.assert_array_equal(params.mass, again.pipeline_state.sim_data.params.mass)
        assert not np.array_equal(params.mass, other.pipeline_state.sim_data.params.mass)
        step = randomized.step(state, nominal.hover_action)
        fixed = nominal.step(nominal.reset(key), nominal.hover_action)
        assert not np.allclose(step.pipeline_state.sim_data.states.vel, fixed.pipeline_state.sim_data.states.vel)
        np.testing.assert_array_equal(step.pipeline_state.sim_data.params.mass, params.mass)
        np.testing.assert_array_equal(state.info["physical_parameters"]["mass_kg"], params.mass)
    finally:
        nominal.close()
        randomized.close()


def test_inertia_randomization_preserves_inverse_and_changes_rigid_body_response():
    config = compose_experiment('control/ppo', "hovering")
    config["env"]["dynamics"]["forward"] = "first_principles"
    config["training"]["domain_randomization"] = {"enabled": True, "dynamics": {"inertia": [2.0, 2.0]}}
    nominal = build_environment(config, device="cpu", role="eval", count=1)
    randomized = build_environment(config, device="cpu", role="train", count=1)
    try:
        key = jax.random.PRNGKey(11)
        fixed, variable = nominal.reset(key), randomized.reset(key)
        parameters = variable.pipeline_state.sim_data.params
        np.testing.assert_allclose(parameters.J @ parameters.J_inv, jnp.eye(3), atol=1e-6)
        action = nominal.hover_action.at[0].set(0.3)
        # The attitude controller produces unequal rotor commands; inertia must
        # affect the integrated body angular velocity, not just saved metadata.
        for _ in range(3):
            fixed = nominal.step(fixed, action)
            variable = randomized.step(variable, action)
        assert not np.allclose(variable.pipeline_state.sim_data.states.ang_vel,
                               fixed.pipeline_state.sim_data.states.ang_vel, atol=1e-7)
    finally:
        nominal.close()
        randomized.close()


def test_fitted_attitude_model_rejects_inertia_randomization_without_torque_dynamics():
    config = compose_experiment('control/ppo', "hovering")
    config["training"]["domain_randomization"] = {"enabled": True, "dynamics": {"inertia": [.9, 1.1]}}
    with pytest.raises(ValueError, match=r"inertia|randomizable"):
        build_environment(config, device="cpu", role="train", count=1)


def test_retired_split_names_are_rejected():
    from drone_playground.environments.environment import build_environment as construct

    for role in ("dev", "heldout", "checkpoint", "checkpoint_eval", "benchmark"):
        with pytest.raises(ValueError, match=r"Choose"):
            construct({}, role=role)


def test_current_artifact_schema_rejects_historical_versions():
    from drone_playground.artifacts.schema import require_current

    for version in (None, 1, 2):
        with pytest.raises(ValueError, match=r"config_version"):
            require_current({"config_version": version})
    assert require_current({"config_version": 3}) == {"config_version": 3}


@pytest.mark.parametrize("kind", ["termination", "truncation"])
def test_task_time_limit_classification_and_pre_reset_observation(kind):
    from drone_playground.learning.wrappers import wrap_for_training

    config = compose_experiment('control/ppo', "hovering")
    config["env"]["task"]["time_limit_kind"] = kind
    config["env"]["task"]["duration"] = .02
    env = build_environment(config, device="cpu", count=1)
    try:
        wrapped = wrap_for_training(env, episode_length=1)
        initial = wrapped.reset(jax.random.split(jax.random.PRNGKey(2), 1))
        final = wrapped.step(initial, jnp.broadcast_to(env.hover_action, (1, 4)))
        assert bool(final.done[0])
        assert bool(final.info["terminated"][0]) == (kind == "termination")
        assert bool(final.info["truncation"][0]) == (kind == "truncation")
        assert not np.array_equal(final.obs, final.info["terminal_observation"])
    finally:
        env.close()


def test_ppo_truncation_bootstraps_terminal_value_without_using_next_episode():
    from brax.training.agents.ppo.losses import compute_gae

    targets, _ = compute_gae(
        truncation=jnp.array([[1.], [0.]]), termination=jnp.array([[0.], [1.]]),
        rewards=jnp.array([[1.], [2.]]), values=jnp.array([[3.], [999.]]),
        bootstrap_value=jnp.array([777.]), next_values=jnp.array([[10.], [888.]]),
        discount=.9, lambda_=.95,
    )
    np.testing.assert_allclose(targets[:, 0], [10., 2.], atol=1e-5)


def test_acceptance_is_separate_and_consumes_configured_thresholds():
    from drone_playground.benchmarks import apply_quality
    from drone_playground.evaluation.tracking.metrics import summarize_trials

    report = summarize_trials(dict(active=np.ones((2, 1), bool),
                                   failed=np.zeros((2, 1), bool), reward=np.zeros((2, 1)),
                                   metrics=dict(tracking_error=np.full((2, 1), .3))), [1], .02)
    assert "quality_passed" not in report
    config = compose_experiment('control/ppo', "hovering")
    apply_quality(report, config)
    assert not report["quality_passed"]
    config["evaluation"]["acceptance"]["tracking_rmse_m"] = .4
    apply_quality(report, config)
    assert report["quality_passed"]


def test_latency_statistics_exclude_warmup_and_count_deadline_misses():
    from drone_playground.runtime.timing import decision_statistics

    report = decision_statistics([1., .001, .003, .007], .005)
    assert report["decision_samples"] == 3
    assert report["warmup_decisions"] == 1
    assert report["decision_p50_ms"] == 3.
    assert report["deadline_miss_fraction"] == 1/3


def test_sensor_metadata_matches_executed_frame_capture_times():
    config = compose_experiment('papers/ego_planner', "navigation/static")
    config["evaluation"].pop("protocol", None)
    env = build_environment(config, device="cpu", role="eval", count=1)
    try:
        state = env.reset(jax.random.PRNGKey(7), jnp.int32(0))
        times = []
        for _ in range(4):
            state = env.step(state, jnp.zeros(env.action_size))
            times.append(float(state.pipeline_state.sensor_time[-1]))
        assert env.sensor_timing["requested_source_hz"] == 30.
        assert env.sensor_timing["effective_capture_hz"] == 25.
        np.testing.assert_allclose(times, [0., .04, .04, .08], atol=1e-6)
    finally:
        env.close()


def test_navigation_protocol_supplies_evaluation_and_checkpoint_budgets():
    config = compose_experiment("navigation/differentiable_pointcloud")
    assert config["evaluation"]["episodes"] == 25
    assert config["evaluation"]["seed_start"] == 80000
    assert config["training"]["checkpoint_eval_episodes"] == 8
    assert config["training"]["checkpoint_eval_seed_start"] == 50000


def test_generic_navigation_executes_configured_cases_and_initial_states():
    from collections import Counter

    from drone_playground.evaluation.run import make_evaluator

    config = compose_experiment(
        'navigation/ppo', "navigation/static", ["evaluation.episodes=2"]
    )
    env = build_environment(config, device="cpu", role="eval", count=2)
    env.duration, env.episode_length = .02, 1
    try:
        def make_policy(params, deterministic=True):
            return lambda obs, key: (jnp.zeros((*obs.shape[:-1], env.action_size)), {})

        evaluator = make_evaluator(env, make_policy, [80000, 80001])
        report, traces = evaluator.run({})
        rows = report["episodes"]
        assert Counter(row["scene_id"] for row in rows) == dict.fromkeys(["S01", "S02", "S03", "S06"], 2)
        assert len({row["seed"] for row in rows}) == 8
        assert report["initial_conditions"]["seeds"] == [row["seed"] for row in rows]
        assert len({tuple(row["initial_position_m"]) for row in rows}) == 8
        for group, trace in traces.items():
            expected = report["cells"][group]["episodes"]
            np.testing.assert_allclose(trace["observation_pos"][0], [row["initial_position_m"] for row in expected])
    finally:
        env.close()
