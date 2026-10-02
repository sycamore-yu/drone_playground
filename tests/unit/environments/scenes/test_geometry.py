"""Control-course ray and collision geometry preserves gate orientation."""

import jax.numpy as jnp
import mujoco
import numpy as np


def test_oriented_gate_bar_ray_matches_mujoco():
    from drone_playground.environments.sensors.rays import cast_rays

    xml = '<mujoco><worldbody><geom type="box" pos="1 0 1" size=".1 .5 .1" euler="0 0 45"/></worldbody></mujoco>'
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    rotation = np.asarray(data.geom_xmat).reshape(1, 3, 3)
    origin = np.array([[0.0, 0.0, 1.0]])
    direction = np.array([[1.0, 0.0, 0.0]])
    expected = mujoco.mj_ray(
        model, data, origin[0], direction[0], None, 1, -1, np.zeros(1, np.int32)
    )
    actual = cast_rays(
        jnp.array([2]),
        jnp.array([[0.1, 0.5, 0.1]]),
        jnp.array([[1.0, 0.0, 1.0]]),
        jnp.array([True]),
        origin,
        direction,
        jnp.array([-2.0, -2.0, 0.0]),
        jnp.array([2.0, 2.0, 2.0]),
        False,
        rotations=rotation,
    )
    np.testing.assert_allclose(actual, [expected], atol=1e-6)


def test_capsule_ray_and_distance_match_rounded_obstacle():
    from drone_playground.environments.scenes.geometry import signed_distance
    from drone_playground.environments.sensors.rays import primitive_hit

    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><geom type="capsule" pos="1 0 1" size=".2 .5"/></worldbody></mujoco>'
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    starts = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.6], [1.0, 0.0, 2.0]])
    directions = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
    expected = [
        mujoco.mj_ray(model, data, p, d, None, 1, -1, np.zeros(1, np.int32))
        for p, d in zip(starts, directions, strict=True)
    ]
    actual = primitive_hit(
        jnp.int32(4),
        jnp.array([0.2, 1.0, 0.0]),
        jnp.array([1.0, 0.0, 1.0]),
        starts,
        directions,
        None,
    )
    np.testing.assert_allclose(actual, expected, atol=1e-6)
    distance = signed_distance(
        jnp.int32(4),
        jnp.array([0.2, 1.0, 0.0]),
        jnp.array([1.0, 0.0, 1.0]),
        jnp.array([1.0, 0.0, 1.8]),
    )
    np.testing.assert_allclose(distance, 0.1, atol=1e-6)
