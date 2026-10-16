"""World-frame actions require enough orientation in egocentric policy inputs."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.spatial.transform import Rotation

from drone_playground.simulation.environment import make_environment


def test_rotation_observation_disambiguates_world_goal_actions():
    """Rotated starts with equal egocentric goals must expose different world frames."""
    options = {"task": "navigation", "scene": "S01", "num_envs": 2}
    baseline = make_environment(**options, action={"level": "acceleration"})
    informed = make_environment(
        **options,
        action={"level": "acceleration"},
        observation_config={"navigation_world_rotation": True},
    )
    state = baseline.reset(jax.random.key(8))
    rotations = Rotation.from_euler("z", jnp.array([[0.0], [5.0]]), degrees=True)
    delta = rotations.apply(jnp.array([[90.0, 0.0, 0.0]] * 2))
    states = state.physics.states.replace(
        pos=(baseline.task.goal - delta)[:, None],
        vel=jnp.zeros_like(state.physics.states.vel),
        quat=rotations.as_quat()[:, None],
    )
    state = state.replace(physics=state.physics.replace(states=states))
    original = baseline.observe(state)["state"]
    actual = informed.observe(state)["state"]
    np.testing.assert_allclose(original[0], original[1], atol=1e-6)
    np.testing.assert_array_equal(actual[:, :10], original)
    assert actual.shape == (2, 19)
    assert not np.allclose(actual[0, 10:], actual[1, 10:])
    world_target = jnp.einsum("bij,bj->bi", actual[:, 10:].reshape(2, 3, 3), actual[:, 3:6])
    np.testing.assert_allclose(world_target, delta / 30.0, atol=1e-6)
    flipped = state.replace(physics=state.physics.replace(states=states.replace(quat=-states.quat)))
    np.testing.assert_array_equal(informed.observe(flipped)["state"], actual)
    assert informed.action.contract == baseline.action.contract
    assert informed.action.frame == "world"
    for env in (baseline, informed):
        env.sim.close()


@pytest.mark.parametrize("value", [None, 1, "true"])
def test_world_rotation_requires_a_boolean(value):
    """Reject truthy strings and numbers at the observation configuration boundary."""
    with pytest.raises(ValueError, match="navigation_world_rotation must be boolean"):
        make_environment(
            task="navigation", scene="S01", observation_config={"navigation_world_rotation": value}
        )


def test_world_rotation_is_navigation_only():
    """Do not silently accept an unused flag for another task's state encoding."""
    with pytest.raises(ValueError, match="requires Navigation"):
        make_environment(observation_config={"navigation_world_rotation": True})
