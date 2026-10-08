"""Executable robot-learning concepts, shared tasks and frozen evaluation."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from drone_playground.configuration import compose_experiment
from drone_playground.environments.factory import build_environment
from tests.helpers.configs import bodyrates_config


@pytest.mark.parametrize("forward", ["lotf_high_fidelity", "lotf_simplified"])
def test_lotf_is_a_backend_of_the_same_training_and_evaluation_task(forward):
    cfg = bodyrates_config(forward=forward)
    train = build_environment(cfg, "cpu", "train", 2)
    evaluation = build_environment(cfg, "cpu", "eval", 2)
    try:
        assert type(train.unwrapped) is type(evaluation.unwrapped)
        assert train.task == evaluation.task == "hovering"
        assert cfg["algorithm"]["name"] == "bptt"
        assert train.sim.freq == 1000
        state = train.reset(jax.random.key(7))
        physical = train.physical_action(train.hover_action)
        np.testing.assert_allclose(physical, [0.192 * 9.81, 0, 0, 0], atol=1e-6)
        result = jax.jit(train.step)(state, train.hover_action)
        assert np.isfinite(result.obs).all()
        assert int(result.pipeline_state.sim_data.core.steps[0, 0]) == 20
    finally:
        train.close()
        evaluation.close()


@pytest.mark.parametrize("backward", ["direct", "simplified_dynamics_jacobian"])
def test_lotf_forward_and_backward_are_selected_independently(backward):
    from drone_playground.environments.factory import build_dynamics

    cfg = bodyrates_config(forward="lotf_high_fidelity")
    cfg["algorithm"]["gradient"]["transition"] = backward
    model = build_dynamics(cfg)
    assert model.forward == "lotf_high_fidelity"
    assert model.backward == backward


@pytest.mark.parametrize(
    "preset,forward",
    [
        ("crazyflow_first_principles", "first_principles"),
        ("crazyflow_so_rpy", "so_rpy"),
        ("crazyflow_so_rpy_rotor", "so_rpy_rotor"),
        ("crazyflow_so_rpy_rotor_drag", "so_rpy_rotor_drag"),
    ],
)
def test_crazyflow_forward_models_are_parallel_dynamics_presets(preset, forward):
    from drone_playground.environments.factory import build_dynamics

    cfg = compose_experiment(
        "control/bptt",
        "tracking",
        [f"dynamics@env.dynamics={preset}", "runtime.action_delay_ms=null"],
    )
    model = build_dynamics(cfg)
    assert model.forward == forward


def test_measurement_noise_changes_policy_input_without_changing_physics_or_reward():
    cfg = compose_experiment(
        "control/bptt", "hovering", ["runtime.action_delay_ms=null", "runtime.device=cpu"]
    )
    cfg["training"]["observation_noise"] = {"position_std_m": 0.1, "velocity_std_mps": 0.2}
    env = build_environment(cfg, "cpu", "train", 2)
    try:
        state = env.reset(jax.random.key(5))
        ideal = env.env.observation(state.pipeline_state)
        assert not np.array_equal(state.obs, ideal)
        physical = env.env.step(state, env.hover_action)
        measured = jax.jit(env.step)(state, env.hover_action)
        np.testing.assert_array_equal(
            measured.pipeline_state.sim_data.states.pos, physical.pipeline_state.sim_data.states.pos
        )
        np.testing.assert_array_equal(measured.reward, physical.reward)
        np.testing.assert_array_equal(env.reset(jax.random.key(5)).obs, state.obs)
    finally:
        env.close()


@pytest.mark.parametrize("forward", ["so_rpy", "lotf_high_fidelity", "lotf_simplified"])
def test_wind_is_a_runtime_force_with_a_real_velocity_response(forward):
    cfg = (
        bodyrates_config(forward=forward)
        if forward.startswith("lotf_")
        else compose_experiment(
            "control/bptt", "hovering", ["runtime.action_delay_ms=null", "runtime.device=cpu"]
        )
    )
    cfg["training"]["disturbance"] = {"force_world_n": [0.1, 0, 0]}
    env = build_environment(cfg, "cpu", "train", 2)
    try:
        state = env.reset(jax.random.key(5))
        result = jax.jit(env.step)(state, env.hover_action)
        assert float(result.pipeline_state.sim_data.states.vel[0, 0, 0]) > 0.0001
        assert "external_force_world_n" in result.pipeline_state.sim_data.plugins
    finally:
        env.close()


def test_eval_role_disables_training_randomization_noise_and_disturbance():
    cfg = compose_experiment(
        "control/bptt", "hovering", ["runtime.action_delay_ms=null", "runtime.device=cpu"]
    )
    cfg["training"]["domain_randomization"] = {
        "enabled": True,
        "dynamics": {"mass": [1.2, 1.2], "motor_strength": [0.9, 0.9]},
    }
    cfg["training"]["observation_noise"] = {"position_std_m": 0.1}
    cfg["training"]["action_noise"] = {"bias_normalized": [0.1, 0, 0, 0]}
    cfg["training"]["disturbance"] = {"force_world_n": [0.1, 0, 0]}
    env = build_environment(cfg, "cpu", "eval", 2)
    try:
        state = env.reset(jax.random.key(5))
        np.testing.assert_allclose(
            state.pipeline_state.sim_data.params.mass, env.default.params.mass
        )
        assert env.role == "eval"
        assert not env.experiment_config["training"]["domain_randomization"]["enabled"]
        assert not any(env.environment_effects.values())
    finally:
        env.close()


def test_training_position_goals_are_separate_from_the_scene_and_nominal_goals():
    cfg = compose_experiment(
        "control/bptt", "hovering", ["runtime.action_delay_ms=null", "runtime.device=cpu"]
    )
    cfg["training"]["command_distribution"] = {
        "kind": "position",
        "distribution": "uniform",
        "low": [-1, -1, 1],
        "high": [1, 1, 2],
    }
    train = build_environment(cfg, "cpu", "train", 2)
    nominal = build_environment(cfg, "cpu", "eval", 2)
    try:
        goals = jax.vmap(train.reset)(jax.random.split(jax.random.key(5), 8)).pipeline_state.command
        assert np.ptp(np.asarray(goals), axis=0).min() > 0.1
        assert nominal.command_distribution["distribution"] == "reference_bank"
        assert train.scene == nominal.scene
    finally:
        train.close()
        nominal.close()


def test_action_uncertainty_reaches_delayed_execution_and_nominal_racing_is_clean():
    cfg = compose_experiment(
        "control/bptt", "hovering", ["runtime.action_delay_ms=[0,0]", "runtime.device=cpu"]
    )
    cfg["training"]["action_noise"] = {"bias_normalized": [0.2, 0, 0, 0]}
    env = build_environment(cfg, "cpu", "train", 1)
    try:
        state = env.reset(jax.random.key(3))
        result = jax.jit(env.step)(state, env.hover_action)
        expected = env.physical_action(env.hover_action + jnp.array([0.2, 0, 0, 0]))
        np.testing.assert_allclose(
            result.pipeline_state.sim_data.controls.attitude.staged_cmd[0, 0], expected, atol=1e-6
        )
    finally:
        env.close()
    race = build_environment(
        compose_experiment(
            "control/bptt", "racing", ["runtime.device=cpu", "runtime.action_delay_ms=null"]
        ),
        "cpu",
        "eval",
        1,
    )
    try:
        assert race.core.settings.disturbances == {}
        assert not any(race.environment_effects.values())
    finally:
        race.close()


def test_point_mass_randomization_reset_and_measurement_have_real_effects():
    from drone_playground.dynamics.point_mass import PointMassLag, PointMassState
    from drone_playground.environments.randomization import (
        noisy_point_mass_state,
        reset_point_mass_state,
    )

    initial = PointMassState.create(jnp.zeros((2, 3))).replace(
        measurement_key=jax.random.split(jax.random.PRNGKey(1), 2)
    )
    model = PointMassLag(
        domain_randomization={
            "enabled": True,
            "dynamics": {"motor_strength": [0.5, 0.5], "lag": [2, 2]},
        }
    )
    randomized = model.randomize(initial, jax.random.PRNGKey(2))
    from drone_playground.control.setpoints import StateSetpoint

    control = StateSetpoint(acceleration=jnp.ones((2, 3)))
    nominal = PointMassLag().step(initial, control, 0.1)
    changed = model.step(randomized, control, 0.1)
    assert np.max(np.asarray(changed.acc)) < np.min(np.asarray(nominal.acc))
    reset = reset_point_mass_state(
        initial,
        jax.random.PRNGKey(3),
        {"position_std_m": 0.1, "orientation_half_width_rad": 0.2, "velocity_std_mps": 0.2},
    )
    assert np.linalg.norm(reset.pos) > 0 and not np.array_equal(reset.rotation, initial.rotation)
    measured = noisy_point_mass_state(reset, jnp.zeros(2), 0.1, {"position_std_m": 0.1})
    assert not np.array_equal(measured.pos, reset.pos)
    np.testing.assert_array_equal(
        reset_point_mass_state(
            initial,
            jax.random.PRNGKey(3),
            {"position_std_m": 0.1, "orientation_half_width_rad": 0.2, "velocity_std_mps": 0.2},
        ).pos,
        reset.pos,
    )
    with pytest.raises(ValueError, match=r"mass/inertia"):
        PointMassLag(domain_randomization={"enabled": True, "dynamics": {"mass": [0.9, 1.1]}})


def test_lotf_full_mass_and_inertia_randomization_change_physical_response():
    cfg = bodyrates_config(forward="lotf_high_fidelity")
    cfg["training"]["domain_randomization"] = {
        "enabled": True,
        "dynamics": {"mass": [1.2, 1.2], "inertia": [1.5, 1.5]},
    }
    env = build_environment(cfg, "cpu", "train", 1)
    try:
        randomized = env.reset(jax.random.key(3))
        nominal = randomized.replace(
            pipeline_state=randomized.pipeline_state.replace(
                sim_data=randomized.pipeline_state.sim_data.replace(params=env.default.params)
            )
        )
        np.testing.assert_allclose(
            randomized.pipeline_state.sim_data.params.J, 1.5 * env.default.params.J
        )
        command = env.hover_action + jnp.array([0, 0.2, 0, 0])
        advance = jax.jit(env.step)
        first, second = advance(randomized, command), advance(nominal, command)
        assert not np.allclose(
            first.pipeline_state.sim_data.states.vel, second.pipeline_state.sim_data.states.vel
        )
        assert not np.allclose(
            first.pipeline_state.sim_data.states.ang_vel,
            second.pipeline_state.sim_data.states.ang_vel,
        )
    finally:
        env.close()


def test_training_internal_pointcloud_evaluation_uses_nominal_conditions():
    import copy

    from drone_playground.learning.algorithms.recurrent_bptt import (
        initialize,
        make_checkpoint_eval_evaluator,
    )

    cfg = compose_experiment(
        "papers/differentiable_pointcloud",
        overrides=[
            "runtime.device=cpu",
            "training.num_envs=2",
            "training.checkpoint_eval_envs=2",
            "algorithm.horizon_length=2",
            "env.sensor.azimuth_count=6",
            "env.sensor.elevation_count=2",
            "env.scene.obstacles_per_kind=1",
            "objective.velocity_window=2",
        ],
    )
    randomized = copy.deepcopy(cfg)
    randomized["training"]["domain_randomization"] = {
        "enabled": True,
        "dynamics": {"motor_strength": [0.5, 0.5]},
    }
    randomized["training"]["observation_noise"] = {"position_std_m": 0.2, "sensor_std_m": 0.1}
    randomized["training"]["action_noise"] = {"std_physical": 0.3}
    results = []
    for selected in (cfg, randomized):
        task = build_environment(selected, "cpu", "train", 2)
        try:
            state, network, _ = initialize(task, selected)
            evaluate, _ = make_checkpoint_eval_evaluator(task, network, selected)
            results.append(evaluate(state.params))
        finally:
            task.close()
    for key in results[0]:
        np.testing.assert_array_equal(results[0][key], results[1][key])


def test_pointcloud_fixed_scene_distribution_is_consumed_and_reproducible():
    from drone_playground.learning.algorithms.recurrent_bptt import initialize, make_update

    cfg = compose_experiment(
        "papers/differentiable_pointcloud",
        overrides=[
            "runtime.device=cpu",
            "training.num_envs=2",
            "algorithm.horizon_length=2",
            "env.sensor.azimuth_count=6",
            "env.sensor.elevation_count=2",
            "objective.velocity_window=2",
        ],
    )
    cfg["training"]["scene_distribution"] = {
        "type": "fixed",
        "scene": {
            "_target_": "drone_playground.environments.scenes.catalog.NavigationCatalogScene",
            "scene_ids": ["S01"],
        },
    }
    cfg["env"]["task"]["command_distribution"]["speed_range_mps"] = [4.0, 4.0]
    task = build_environment(cfg, "cpu", "train", 2)
    try:
        state, network, optimizer = initialize(task, cfg)
        update = make_update(task, network, optimizer, cfg)
        first, a = update(state)
        second, b = update(state.replace(key=jax.random.PRNGKey(9)))
        assert int(first.updates) == int(second.updates) == 1
        assert np.isfinite(float(a["loss"]))
        np.testing.assert_allclose(a["loss"], b["loss"], atol=1e-6)
    finally:
        task.close()


def test_source_rollout_rejects_an_unconsumed_integer_action_delay():
    cfg = compose_experiment(
        "papers/differentiable_pointcloud",
        overrides=["runtime.device=cpu", "runtime.action_delay_steps=1"],
    )
    with pytest.raises(ValueError, match=r"delay"):
        build_environment(cfg, "cpu", "train", 1)


@pytest.mark.parametrize("environment", ["tracking", "racing", "navigation/static"])
@pytest.mark.parametrize("forward", ["lotf_high_fidelity", "lotf_simplified"])
def test_source_dynamics_reuses_tracking_racing_and_navigation(environment, forward):
    cfg = bodyrates_config(environment, forward)
    if environment.startswith("navigation/"):
        cfg = compose_experiment(
            "control/bptt",
            environment,
            [
                "dynamics@env.dynamics=" + forward,
                "controller@env.controller=rates",
                "env.controller.input_kind=rates",
                "sensor@env.sensor=none",
                "env.physics_freq=1000",
                "env.task.duration=0.04",
                "runtime.action_delay_ms=null",
                "runtime.device=cpu",
            ],
        )
        cfg["env"]["task"]["observation"] = {
            "_target_": "drone_playground.environments.observations.state.NavigationObservation",
            "name": "navigation_state",
            "include_goal": True,
            "include_previous_action": True,
            "action_size": 4,
        }
    env = build_environment(cfg, "cpu", "eval", 1)
    try:
        state = env.reset(jax.random.key(1))
        result = jax.jit(env.step)(state, env.hover_action)
        assert np.isfinite(result.obs).all()
        assert env.task in ("tracking", "racing", "navigation")
        assert result.pipeline_state.sim_data.core.freq == 1000
    finally:
        env.close()
