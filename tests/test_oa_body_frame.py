"""OA body-frame acceleration must agree with body-state observation and physics."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from jax.scipy.spatial.transform import Rotation

from drone_playground.simulation.environment import make_environment
from drone_playground.simulation.methods import Action


def test_body_acceleration_matches_world_setpoint_for_arbitrary_attitude():
    """Body commands must convert to the same physical setpoint as rotated world commands."""
    env = make_environment(task="navigation", scene="S01", num_envs=2)
    state = env.reset(jax.random.key(10))
    rotations = Rotation.from_euler(
        "xyz", jnp.array([[0.0, 0.0, 90.0], [20.0, -15.0, 40.0]]), degrees=True
    )
    physics = state.physics.replace(
        states=state.physics.states.replace(quat=rotations.as_quat()[:, None])
    )
    body = Action(level="acceleration", frame="body")
    world = Action(level="acceleration", frame="world")
    physical = jnp.array([[1.5, 0.0, 0.5], [-0.3, 1.2, 0.6]])
    expected_world = rotations.apply(physical)
    np.testing.assert_allclose(
        body.world_acceleration(physics, physical), expected_world, atol=1e-6
    )
    result = body.setpoint(physics, physical)
    expected = world.setpoint(physics, expected_world)
    assert result.level == expected.level == "attitude_thrust"
    assert result.frame == expected.frame == "world"
    np.testing.assert_allclose(result.value, expected.value, atol=1e-5, rtol=1e-5)
    assert body.contract["frame"] == "body" and world.contract["frame"] == "world"
    env.sim.close()


def test_body_acceleration_rotation_gradient_is_physical_world_rotation():
    """The acceleration Jacobian is the full current body-to-world rotation."""
    env = make_environment(num_envs=1)
    state = env.reset(jax.random.key(8))
    rotation = Rotation.from_euler("xyz", jnp.array([[10.0, -15.0, 90.0]]), degrees=True)
    physics = state.physics.replace(
        states=state.physics.states.replace(quat=rotation.as_quat()[:, None])
    )
    action = Action(level="acceleration", frame="body")
    jacobian = jax.jacfwd(lambda value: action.world_acceleration(physics, value[None])[0])(
        jnp.zeros(3)
    )
    np.testing.assert_allclose(jacobian, rotation.as_matrix()[0], atol=1e-6)
    env.sim.close()


@pytest.mark.parametrize(
    "level,frame",
    [("horizontal_acceleration", "body"), ("attitude_thrust", "body"), ("body_rate", "world")],
)
def test_unimplemented_frame_combinations_are_rejected(level, frame):
    """Do not silently reinterpret other action levels in a different frame."""
    with pytest.raises(ValueError, match="Unsupported action frame"):
        Action(level=level, frame=frame)


def test_body_policy_delivers_world_acceleration_to_point_mass_backward_model():
    """Delay buffers and the point-mass surrogate require issued world acceleration."""
    from drone_playground.learning.dynamics import training_step

    env = make_environment(
        task="navigation",
        scene="S01",
        num_envs=1,
        action={"level": "acceleration", "frame": "body"},
    )
    state = env.reset(jax.random.key(5))
    rotation = Rotation.from_euler("z", jnp.array([[90.0]]), degrees=True)
    state = state.replace(
        physics=state.physics.replace(
            states=state.physics.states.replace(quat=rotation.as_quat()[:, None])
        )
    )
    action = jnp.array([[0.2, 0.0, 0.0]])
    expected = jnp.array([[0.0, 1.2, 0.0]])
    native = env.step(state, action)
    model = training_step(env, model="point_mass_lag")
    surrogate = model(state, action)
    np.testing.assert_allclose(native.commands.applied_input, expected, atol=1e-6)
    np.testing.assert_allclose(surrogate.commands.applied_input, expected, atol=1e-6)
    np.testing.assert_allclose(surrogate.physics.states.pos, native.physics.states.pos, atol=1e-6)
    jac = jax.jacrev(lambda u: model(state, u).physics.states.vel[0, 0, :])(action)
    assert np.isfinite(jac).all()
    assert np.abs(np.asarray(jac)[1, 0, 0]) > np.abs(np.asarray(jac)[0, 0, 0])
    env.sim.close()


@pytest.mark.parametrize("region", ["static", "dynamic"])
@pytest.mark.parametrize("sensor", ["depth", "lidar"])
@pytest.mark.parametrize("algo", ["ppo", "bptt", "shac"])
def test_oa_body_frame_candidate_has_matching_observation_action(region, sensor, algo):
    """All OA native learners now use the same public body-action default."""
    root = Path(__file__).resolve().parents[1] / "src/drone_playground/configs"
    with initialize_config_dir(version_base=None, config_dir=str(root)):
        cfg = compose(
            config_name="config",
            overrides=[
                f"experiment=oa_{region}_{sensor}_{algo}",
                f"learning={algo}",
                "method.action.frame=body",
                "observation.navigation_world_rotation=false",
            ],
        )
    assert cfg.method.action.level == "acceleration"
    assert cfg.method.action.frame == "body"
    assert cfg.observation.get("navigation_world_rotation", False) is False


def test_lotf_action_frame_is_body_and_retains_native_rate_interface():
    """Do not recast the existing body-rate LOTF contract as acceleration."""
    root = Path(__file__).resolve().parents[1] / "src/drone_playground/configs"
    with initialize_config_dir(version_base=None, config_dir=str(root)):
        cfg = compose(
            config_name="config",
            overrides=[
                "experiment=oa_static_depth_shac",
                "learning=shac",
                "learning.backward_model=lotf",
                "method.action.level=body_rate",
                "method.action.frame=body",
            ],
        )
    assert cfg.method.action.frame == "body"
    action = Action(level=cfg.method.action.level, frame=cfg.method.action.frame)
    assert action.contract["frame"] == "body"
