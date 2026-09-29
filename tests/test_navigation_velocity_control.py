"""Velocity-action navigation keeps the plant/events and adds no obstacle-aware controller."""

import jax
import jax.numpy as jnp
import numpy as np


def controller():
    from drone_playground.execution.controllers.velocity import VelocityControl

    obj = VelocityControl()
    obj.bind(jnp.array([-0.7, -0.7, -3.14, 0.0]), jnp.array([0.7, 0.7, 3.14, 0.6]))
    return obj


def test_zero_velocity_is_gravity_compensation_not_a_goal_controller():
    c = controller()
    result = c.attitude_command(jnp.zeros(3), jnp.zeros(4), jnp.float32(0.032))
    np.testing.assert_allclose(result, [0, 0, 0, 0.032 * 9.81], atol=1e-6)


def test_velocity_norm_and_action_derivatives_are_bounded_and_finite():
    c = controller()
    cmd = c.physical_action(jnp.array([1.0, 1.0, 1.0, 0.0]))
    assert np.linalg.norm(cmd[:3]) <= 20.00001
    action = jnp.array([0.1, 0.05, 0.0, 0.0])
    command = c.attitude_command(jnp.zeros(3), c.physical_action(action), jnp.float32(0.032))
    assert command[1] > 0 and command[0] < 0
    derivative = jax.jacrev(
        lambda a: c.attitude_command(jnp.zeros(3), c.physical_action(a), 0.032)
    )(action)
    assert np.isfinite(derivative).all() and np.linalg.norm(derivative) > 0


def test_controller_has_no_scene_goal_or_obstacle_inputs():
    import inspect

    c = controller()
    assert tuple(inspect.signature(c.attitude_command).parameters) == (
        "velocity",
        "command",
        "mass",
    )


def test_public_velocity_recipes_keep_scene_protocol_and_distinct_action_contract():
    from drone_playground.composition import compose_method, validate_config

    for name in ("ppo", "bptt", "shac"):
        cfg = compose_method("learning/navigation_" + name, "navigation/static_velocity")
        validate_config(cfg)
        assert cfg["env"]["task"]["duration"] == 300
        assert cfg["env"]["task"]["goal_radius"] == 0.5
        assert cfg["method"]["output"] == cfg["env"]["execution"]["command"] == "velocity_yaw"
        assert cfg["env"]["execution"]["dynamics"]["forward"] == "first_principles"
        assert cfg["env"]["execution"]["controller"]["max_speed"] == 20.0


def test_velocity_controller_actually_tracks_using_the_original_physical_plant():
    from drone_playground.execution.controllers.velocity import VelocityControl
    from tests.test_navigation import synthetic_env

    env = synthetic_env([], duration=5.0, controller=VelocityControl())
    start = env.reset(jax.random.PRNGKey(0), jnp.int32(0))
    action = jnp.array([0.05, 0.0, 0.0, 0.0])

    def run(initial):
        def step(state, _):
            nxt = env.step(state, action)
            return nxt, nxt.done

        return jax.lax.scan(step, initial, None, length=150)

    final, done = jax.jit(run)(start)
    pos = np.asarray(final.pipeline_state.sim_data.states.pos[0, 0])
    vel = np.asarray(final.pipeline_state.sim_data.states.vel[0, 0])
    assert np.isfinite(pos).all() and np.isfinite(vel).all()
    assert not np.asarray(done).any()
    assert pos[0] > 2.5 and abs(pos[2] - 2.0) < 0.5
    assert abs(vel[0] - 1.0) < 0.2 and abs(vel[2]) < 0.15
    assert 0 < final.metrics["physical_thrust"] < 0.6
    env.close()


def test_public_composition_instantiates_the_selected_velocity_controller():
    from drone_playground.composition import build_environment, compose_method
    from drone_playground.execution.controllers.velocity import VelocityControl

    config = compose_method("learning/navigation_bptt", "navigation/static_velocity")
    env = build_environment(config, "cpu", "dev", 2)
    try:
        assert isinstance(env.controller, VelocityControl)
        assert env.controller.name == config["env"]["execution"]["controller"]["name"]
        np.testing.assert_allclose(env.physical_action(jnp.array([0.05, 0, 0, 0])), [1, 0, 0, 0])
        start = env.reset(jax.random.PRNGKey(0), jnp.int32(0))
        np.testing.assert_array_equal(start.info["delay_command_queue"], 0.0)
        np.testing.assert_array_equal(start.pipeline_state.previous_action, 0.0)
    finally:
        env.close()


def test_velocity_synthetic_reset_has_zero_velocity_and_zero_yaw_command():
    from drone_playground.execution.controllers.velocity import VelocityControl
    from tests.test_navigation import synthetic_env

    env = synthetic_env([], controller=VelocityControl())
    try:
        np.testing.assert_array_equal(env.physical_action(env.hover_action), jnp.zeros(4))
    finally:
        env.close()


def test_checkpoint_recipe_resolves_named_navigation_methods():
    from drone_playground.app import _checkpoint_recipe
    from drone_playground.composition import compose_method

    for method in ("navigation_bptt", "navigation_ppo", "navigation_shac", "pointcloud_navigation"):
        config = compose_method("learning/" + method)
        assert _checkpoint_recipe(config) == "learning/" + method
