"""Numerical paper-reward contracts, independent of training success."""

import math

import jax
import jax.numpy as jnp
import numpy as np

from drone_playground.simulation import paper_fan_lidar, task_rewards
from drone_playground.simulation.methods import Action
from drone_playground.simulation.networks import Actor


def test_visfly_reward_matches_official_navigation_formula():
    """Match NavigationEnv.py:85-99 term by term, without legacy failure costs."""
    assert hasattr(task_rewards, "visfly_reward_terms"), "Official VisFly reward is missing"
    values = dict(
        position=jnp.array([[0.0, 0.0, 1.0], [8.7, 0.0, 1.0]]),
        target=jnp.array([[9.0, 0.0, 1.0], [9.0, 0.0, 1.0]]),
        velocity=jnp.array([[2.0, 0.0, 0.0], [0.5, 0.0, 0.0]]),
        quaternion=jnp.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
        angular_velocity=jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        direction=jnp.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        clearance=jnp.array([0.5, 2.0]),
        collision_vector=jnp.array([[0.5, 0.0, 0.0], [0.0, 2.0, 0.0]]),
        success=jnp.array([False, True]),
        step_count=jnp.array([5.0, 20.0]),
        max_episode_steps=256,
    )
    reward, terms = task_rewards.visfly_reward_terms(**values)
    expected = np.array(
        [
            0.01 * 18 / (9 + 1e-6) - 0.002 * 2 - 0.01 / 0.7 - 0.005 * 0.5 * 1 / (0.5 + 1e-6),
            0.01 * 0.15 / (0.3 + 1e-6)
            - 0.002 * 0.5
            - 0.01 / 2.2
            + (256 - 20) * 0.1 * (0.2 + 0.8 / 1.5),
        ]
    )
    np.testing.assert_allclose(reward, expected, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(sum(terms.values()), reward, rtol=2e-6, atol=2e-6)
    assert "failure_cost" not in terms


def test_fan_progress_and_approaching_dynamic_obstacle_equations():
    """Equations 8, 12 and 14 reward progress and penalize approaching objects."""
    assert hasattr(task_rewards, "fan_reward_terms"), "Fan equations 5-16 are missing"
    common = dict(
        previous_distance=jnp.array([10.0, 10.0]),
        distance=jnp.array([9.9, 10.1]),
        velocity=jnp.array([[2.0, 0.0], [2.0, 0.0]]),
        acceleration=jnp.zeros((2, 2)),
        previous_acceleration=jnp.zeros((2, 2)),
        static_clearance=jnp.array([5.0, 5.0]),
        dynamic_clearance=jnp.array([[1.5], [1.5]]),
        obstacle_to_drone=jnp.array([[[-2.0, 0.0]], [[-2.0, 0.0]]]),
        obstacle_velocity=jnp.array([[[-1.0, 0.0]], [[1.0, 0.0]]]),
        dynamic_mask=jnp.array([[True], [True]]),
        collision=jnp.array([False, False]),
        weights=dict(
            alive=0.0,
            acceleration=0.0,
            velocity=0.0,
            distance=0.0,
            progress=1.0,
            jerk=0.0,
            obstacle=1.0,
            hover=0.0,
            collision=0.0,
        ),
        min_speed=0.0,
        max_speed=3.0,
        safe_distance=2.0,
        hover_distance=0.5,
    )
    reward, terms = task_rewards.fan_reward_terms(**common)
    np.testing.assert_allclose(terms["progress"], -np.expm1([-0.1, 0.1]), atol=2e-6)
    dilation = 1 + math.e
    np.testing.assert_allclose(
        terms["obstacle"], -np.expm1([2 - 1.5 / dilation, 2 - 1.5]), rtol=2e-6
    )
    assert reward[0] < reward[1]


def test_paper_reward_terms_are_finite_with_zero_speed_and_empty_dynamic_set():
    """Zero-velocity/empty scans do not produce undefined angles or gradients."""
    assert hasattr(task_rewards, "fan_reward_terms")
    kwargs = dict(
        previous_distance=jnp.array([1.0]),
        distance=jnp.array([1.0]),
        velocity=jnp.zeros((1, 2)),
        acceleration=jnp.zeros((1, 2)),
        previous_acceleration=jnp.zeros((1, 2)),
        static_clearance=jnp.array([jnp.inf]),
        dynamic_clearance=jnp.full((1, 1), jnp.inf),
        obstacle_to_drone=jnp.zeros((1, 1, 2)),
        obstacle_velocity=jnp.zeros((1, 1, 2)),
        dynamic_mask=jnp.array([[False]]),
        collision=jnp.array([False]),
        weights=dict(
            alive=0.1,
            acceleration=0.01,
            velocity=0.1,
            distance=0.0,
            progress=1.0,
            jerk=0.01,
            obstacle=0.1,
            hover=1.0,
            collision=2.0,
        ),
        min_speed=0.0,
        max_speed=3.0,
        safe_distance=1.2,
        hover_distance=0.5,
    )
    reward, terms = jax.jit(task_rewards.fan_reward_terms)(**kwargs)
    assert np.isfinite(reward).all()
    assert all(np.isfinite(value).all() for value in terms.values())
    gradient = jax.grad(
        lambda acceleration: task_rewards.fan_reward_terms(
            **{**kwargs, "acceleration": acceleration}
        )[0].sum()
    )(kwargs["acceleration"])
    assert np.isfinite(gradient).all()


def test_visfly_source_initialization_keeps_long_course_actions_unsaturated():
    """PyTorch non-orthogonal initialization uses bounded weights and nonzero biases."""
    assert "initialization" in Actor.__dataclass_fields__, "Initialization recipe missing"
    actor = Actor(kind="visfly_depth", action_size=4, initialization="pytorch")
    obs = {
        "state": jnp.zeros((2, 10)),
        "depth": jnp.full((2, 64, 64, 1), 10.0),
        "visfly_state": jnp.tile(
            jnp.array([2.0, 0.0, 3.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), (2, 1)
        ),
        "visfly_target": jnp.tile(jnp.array([98.0, 0.0, 3.0]), (2, 1)),
    }
    params = actor.init(jax.random.key(19), obs, actor.initialize_memory(2))
    raw = actor.apply(params, obs, actor.initialize_memory(2))[0]
    assert float(jnp.max(jnp.abs(jnp.tanh(raw)))) < 0.95
    for name, subtree in params["params"]["visfly_policy"]["VisFlyFeatures_0"].items():
        if "kernel" in subtree:
            fan_in = math.prod(subtree["kernel"].shape[:-1])
            assert float(jnp.max(jnp.abs(subtree["kernel"]))) <= 1 / math.sqrt(fan_in)
            assert bool(jnp.any(subtree["bias"] != 0)), name


def test_visfly_source_action_range_is_three_g_and_two_radians_per_second():
    """Port max_acc=3g and max_rate=2 from upstream drone_state.json."""
    action = Action(level="body_rate", thrust_to_weight_bounds=(0.0, 3.0), body_rate_limit=2.0)
    values = action.decode(jnp.array([[-1.0, -1.0, -1.0, -1.0], [1.0, 1.0, 1.0, 1.0]]))
    np.testing.assert_allclose(values[:, 0], [0.0, 3.0 * action.hover], rtol=1e-6)
    np.testing.assert_allclose(values[:, 1:], [[-2.0, -2.0, -2.0], [2.0, 2.0, 2.0]])


def test_raw_point_window_retains_missed_returns_and_expires_old_points():
    """The paper pools raw historical points before making each 36-bin scan."""
    assert hasattr(paper_fan_lidar, "push_point_window"), "Raw point sliding window missing"
    points = jnp.zeros((1, 3, 1, 3))
    mask = jnp.zeros((1, 3, 1), bool)
    points, mask = paper_fan_lidar.push_point_window(
        points, mask, jnp.array([[[2.0, 0.0, 3.0]]]), jnp.array([[True]])
    )
    points, mask = paper_fan_lidar.push_point_window(
        points, mask, jnp.zeros((1, 1, 3)), jnp.array([[False]])
    )
    # UAV moved a metre; historical hit is re-expressed at its current position.
    ring = paper_fan_lidar.encode_range_ring(
        points.reshape(1, -1, 3) - jnp.array([[[1.0, 0.0, 3.0]]]), mask.reshape(1, -1)
    )
    np.testing.assert_allclose(jnp.min(ring), 0.1, atol=1e-6)
    for _ in range(2):
        points, mask = paper_fan_lidar.push_point_window(
            points, mask, jnp.zeros((1, 1, 3)), jnp.array([[False]])
        )
    assert not bool(jnp.any(mask))
