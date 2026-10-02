"""Navigation training changes must preserve frozen interfaces and useful gradients."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from drone_playground.networks.perception import SensorLayout, perception_network_factory


def small_network(**options):
    layout = SensorLayout("lidar", 20, 2, 4, 5)
    config = dict(hidden_sizes=[16, 16], distribution_type="tanh_normal", **options)
    return layout, perception_network_factory(layout, config)(layout.total_size, 4)


def test_explicit_proprioception_scale_matches_preconditioned_input():
    scale = [20.0] * 20
    layout, conditioned = small_network(proprio_scale=scale)
    _, base = small_network()
    key = jax.random.PRNGKey(5)
    params = conditioned.policy_network.init(key)
    obs = jnp.ones((2, layout.total_size)).at[:, :20].set(20.0)
    expected = base.policy_network.apply(None, params, obs.at[:, :20].divide(20.0))
    actual = conditioned.policy_network.apply(None, params, obs)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_hover_mean_initialization_is_inside_the_tanh_distribution():
    layout, network = small_network(mean_kernel_scale=0.0, mean_action_bias=[0, 0, 0, 0.4])
    params = network.policy_network.init(jax.random.PRNGKey(8))
    logits = network.policy_network.apply(None, params, jnp.zeros((1, layout.total_size)))
    action = network.parametric_action_distribution.mode(logits)
    np.testing.assert_allclose(action, [[0, 0, 0, 0.4]], atol=1e-6)


def test_sensor_conditioned_value_changes_when_only_obstacles_change():
    layout, network = small_network(critic_uses_sensor=True)
    params = network.value_network.init(jax.random.PRNGKey(0))
    before = jnp.zeros((1, layout.total_size))
    after = before.at[:, layout.proprioception_size :].set(0.7)
    a = network.value_network.apply(None, params, before)
    b = network.value_network.apply(None, params, after)
    assert not np.allclose(a, b, atol=1e-6), "Value must distinguish obstacle observations"
    derivative = jax.grad(lambda obs: network.value_network.apply(None, params, obs).sum())(after)
    assert np.isfinite(derivative).all()
    assert np.linalg.norm(derivative[:, 20:]) > 0


def test_legacy_value_remains_proprioception_only():
    layout, network = small_network()
    params = network.value_network.init(jax.random.PRNGKey(0))
    before = jnp.zeros((1, layout.total_size))
    after = before.at[:, layout.proprioception_size :].set(0.7)
    np.testing.assert_array_equal(
        network.value_network.apply(None, params, before),
        network.value_network.apply(None, params, after),
    )


def test_invalid_conditioning_is_rejected_before_training():
    with pytest.raises(ValueError):
        small_network(proprio_scale=[0.0] * 20)


def motion_reward(**changes):
    from drone_playground.learning.objectives.navigation import MotionNavigationObjective

    args = dict(
        arrived=False,
        collided=False,
        out_of_bounds=False,
        numerical_failure=False,
        previous_distance=20.0,
        distance=20.0,
        clearance=3.0,
        action=jnp.zeros(4),
        previous_action=jnp.zeros(4),
        velocity=jnp.zeros(3),
        goal_delta=jnp.array([20.0, 0.0, 0.0]),
        dt=0.02,
    )
    args.update(changes)
    return MotionNavigationObjective(target_speed=2.0)(**args)


def test_continuous_reward_has_forward_and_altitude_gradients_before_collision():
    forward = jax.grad(lambda v: motion_reward(velocity=v))(jnp.zeros(3))
    assert forward[0] > 0
    downward = jax.grad(lambda z: motion_reward(goal_delta=jnp.array([20.0, 0.0, z])))(
        jnp.float32(1.0)
    )
    assert downward < 0, "Larger vertical target error must reduce reward"


def test_penetrating_and_approaching_obstacles_keep_escape_gradients():
    for clearance in (-0.2, 0.2, 1.0):
        derivative = jax.grad(lambda c: motion_reward(clearance=c))(jnp.float32(clearance))
        assert np.isfinite(derivative) and derivative > 0


def test_arrival_reward_is_withheld_for_any_failure_and_speed_excess_is_penalized():
    safe = float(motion_reward(arrived=True))
    failed = float(motion_reward(arrived=True, numerical_failure=True))
    assert failed < safe - 100
    assert motion_reward(velocity=jnp.array([25.0, 0.0, 0.0])) < motion_reward(
        velocity=jnp.array([20.0, 0.0, 0.0])
    )


def test_motion_terms_are_integrated_in_seconds_not_frame_count():
    value = motion_reward(dt=0.02)
    half = motion_reward(dt=0.01)
    np.testing.assert_allclose(value, 2 * half, rtol=1e-5)


def test_low_speed_command_range_does_not_reward_accelerating_toward_the_nominal_cap():
    from drone_playground.learning.objectives.navigation import MotionNavigationObjective

    objective = MotionNavigationObjective(progress_scale=0.2)

    def reward(speed):
        return objective(
            arrived=False,
            collided=False,
            out_of_bounds=False,
            numerical_failure=False,
            previous_distance=20.0,
            distance=20.0 - speed * 0.02,
            clearance=3.0,
            velocity=jnp.array([speed, 0.0, 0.0]),
            goal_delta=jnp.array([20.0, 0.0, 0.0]),
            action=jnp.zeros(4),
            previous_action=jnp.zeros(4),
            dt=0.02,
        )

    assert reward(2.0) > reward(4.0)
    assert reward(2.0) > reward(10.0)
    assert jax.grad(reward)(jnp.float32(4.0)) < 0


def test_global_reward_scaling_preserves_continuous_and_terminal_relations():
    from drone_playground.learning.objectives.navigation import MotionNavigationObjective

    args = dict(
        arrived=False,
        collided=True,
        out_of_bounds=False,
        numerical_failure=False,
        previous_distance=20.0,
        distance=19.0,
        clearance=-0.1,
        velocity=jnp.ones(3),
        goal_delta=jnp.array([19.0, 0.0, 0.0]),
        action=jnp.zeros(4),
        previous_action=jnp.zeros(4),
        dt=0.02,
    )
    original = MotionNavigationObjective()(**args)
    scaled = MotionNavigationObjective(reward_scale=0.01)(**args)
    np.testing.assert_allclose(scaled, original * 0.01, rtol=1e-6)


def test_motion_objective_receives_real_environment_state():
    from drone_playground.learning.objectives.navigation import MotionNavigationObjective
    from tests.helpers.scenes import place
    from tests.helpers.scenes import synthetic_navigation_env as synthetic_env

    env = synthetic_env([], objective=MotionNavigationObjective())
    state = env.reset(jax.random.PRNGKey(0), jnp.int32(0))
    state = place(state, env, (1.0, 0.0, 2.0))
    nxt = env.step(state, env.hover_action)
    assert np.isfinite(nxt.reward)
    derivative = jax.grad(lambda a: env.step(state, a).reward)(env.hover_action)
    assert np.isfinite(derivative).all()
    assert np.linalg.norm(derivative) > 0
    env.close()


def test_navigation_selection_uses_progress_only_after_real_task_outcomes():
    from drone_playground.benchmarks import checkpoint_eval_score

    def report(success, collision, remaining):
        return dict(
            success_rate=success,
            collision_rate=collision,
            constrained_time_mean_s=300.0,
            cells={"easy": {"episodes": [{"final_goal_distance_m": remaining}]}},
        )

    a, b = report(0.0, 1.0, 96.0), report(0.0, 1.0, 80.0)
    assert checkpoint_eval_score("navigation", a) == checkpoint_eval_score("navigation", b)
    assert checkpoint_eval_score("navigation", b, "navigation-convergence-v1") > checkpoint_eval_score(
        "navigation", a, "navigation-convergence-v1"
    )
    assert checkpoint_eval_score(
        "navigation", report(1.0, 0.0, 0.5), "navigation-convergence-v1"
    ) > checkpoint_eval_score("navigation", report(0.0, 0.0, 0.1), "navigation-convergence-v1")


def test_navigation_v2_selection_cannot_prefer_boundary_escape_to_real_progress():
    from drone_playground.benchmarks import checkpoint_eval_score

    def report(collision, boundary, remaining):
        return dict(
            success_rate=0.0,
            collision_rate=collision,
            failure_rate=collision + boundary,
            constrained_time_mean_s=300.0,
            cells={"easy": {"episodes": [{"final_goal_distance_m": remaining}]}},
        )

    escape = report(0.0, 1.0, 96.0)
    progress = report(1.0, 0.0, 20.0)
    assert checkpoint_eval_score(
        "navigation", progress, "navigation-convergence-v2"
    ) > checkpoint_eval_score("navigation", escape, "navigation-convergence-v2")


def test_navigation_v3_selection_ranks_successful_time_ahead_of_terminal_roundoff():
    from drone_playground.benchmarks import checkpoint_eval_score

    def report(seconds, distance, arrived=True):
        return dict(
            success_rate=float(arrived),
            collision_rate=0.0,
            failure_rate=0.0,
            constrained_time_mean_s=seconds,
            cells={"easy": {"episodes": [{"arrived": arrived, "final_goal_distance_m": distance}]}},
        )

    fast, slow = report(25.0, 0.4999), report(50.0, 0.4991)
    assert checkpoint_eval_score("navigation", fast, "navigation-convergence-v3") > checkpoint_eval_score(
        "navigation", slow, "navigation-convergence-v3"
    )
    assert checkpoint_eval_score(
        "navigation", report(300.0, 2.0, False), "navigation-convergence-v3"
    ) > checkpoint_eval_score("navigation", report(300.0, 20.0, False), "navigation-convergence-v3")


def test_optional_position_cost_keeps_gradient_when_progress_telescopes_to_zero():
    from drone_playground.learning.objectives.navigation import MotionNavigationObjective

    objective = MotionNavigationObjective(
        position_scale=0.1,
        progress_scale=0.0,
        velocity_scale=0.0,
        altitude_scale=0.0,
        clearance_scale=0.0,
        smoothness_scale=0.0,
    )

    def loss(x):
        return objective(
            arrived=False,
            collided=False,
            out_of_bounds=False,
            numerical_failure=False,
            previous_distance=10.0,
            distance=x,
            clearance=5.0,
            action=jnp.zeros(4),
            previous_action=jnp.zeros(4),
            velocity=jnp.zeros(3),
            goal_delta=jnp.array([x, 0.0, 0.0]),
            dt=0.02,
        )

    np.testing.assert_allclose(jax.grad(loss)(jnp.float32(5)), -0.002, atol=1e-6)


@pytest.mark.parametrize("thresholds", [[2, 4], [2, 20], [20, 30]])
def test_early_exit_evaluation_preserves_full_trace_and_individual_terminal_states(thresholds):
    from flax import struct

    from drone_playground.runtime.jax_runner import rollout

    @struct.dataclass
    class ToyState:
        obs: object
        done: object
        count: object
        threshold: object

    class Env:
        action_size = 1

        def step(self, state, action):
            count = state.count + 1
            return state.replace(obs=state.obs + action, count=count, done=count >= state.threshold)

    initial = ToyState(
        jnp.zeros((2, 1)), jnp.zeros(2, bool), jnp.zeros(2, int), jnp.array(thresholds)
    )

    def policy(obs, key):
        return jnp.ones_like(obs), {}

    def project(old, new, action, alive, ended, index):
        return dict(
            obs=new.obs,
            done=new.done,
            actions=action,
            active=alive,
            count=new.count,
            ended=ended,
            time=jnp.full((2,), index + 1),
        )

    expected = jax.jit(lambda s: rollout(Env(), policy, s, 11, project))(initial)
    actual = jax.jit(lambda s: rollout(Env(), policy, s, 11, project, early_exit=True))(initial)
    for name in expected:
        np.testing.assert_array_equal(actual[name], expected[name], err_msg=name)
