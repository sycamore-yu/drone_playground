import jax
import jax.numpy as jnp
import numpy as np
import pytest

from drone_playground.networks.depth_flight import DepthFlightPolicy, inverse_depth_image


@pytest.mark.parametrize("distance,returned", [(0.1, False), (0.2, True), (9.0, True),
                                              (10.0, True), (10.1, False), (24.0, False)])
def test_d435i_axial_depth_cutoff_on_a_real_plane(distance, returned):
    from drone_playground.environments.scenes.navigation import KIND_BOX
    from drone_playground.environments.sensors.depth_flight import DepthFlightCamera
    from tests.test_depth_sensor import synthetic_bank

    camera = DepthFlightCamera(pitch_degrees=0)
    bank = synthetic_bank([
        dict(kind=KIND_BOX, size=(0.05, 50, 50), origin=(distance + 0.05, 0, 20))
    ])
    depth, valid = camera.sample(bank, 0, jnp.array([0., 0., 20.]), jnp.eye(3), 0.)
    assert np.all(np.asarray(valid) == returned)
    np.testing.assert_allclose(depth, distance if returned else 10.0, atol=1e-6)


def test_d435i_resized_intrinsics_and_recorded_capture_decimation():
    from drone_playground.environments.sensors.depth_flight import DepthFlightCamera

    camera = DepthFlightCamera()
    fx, fy = camera.focal_pixels
    np.testing.assert_allclose(np.rad2deg(2 * np.arctan([32 / fx, 24 / fy])), [87, 58])
    calibration = camera.calibration()
    assert calibration["frame_decimation"] == 3
    assert calibration["source_rate_hz"] == 10
    assert calibration["nominal_capture_rate_hz"] == 30


def test_depth_preprocessing_keeps_closest_obstacle_and_unknown_is_far():
    depth = jnp.full((1, 48, 64), 10.0).at[0, 1, 2].set(1.0)
    valid = jnp.ones_like(depth, bool)
    image = inverse_depth_image(depth, valid)
    assert image.shape == (1, 12, 16, 1)
    np.testing.assert_allclose(image[0, 0, 0, 0], 2.4)
    np.testing.assert_allclose(inverse_depth_image(depth, jnp.zeros_like(valid)), -0.3)
    np.testing.assert_allclose(
        inverse_depth_image(depth, jnp.zeros_like(valid), far_m=24), -0.475
    )


def test_depth_actor_is_recurrent_and_has_finite_action_and_auxiliary_gradients():
    network = DepthFlightPolicy()
    depth, valid = jnp.ones((2, 48, 64)), jnp.ones((2, 48, 64), bool)
    proprio, hidden = jnp.ones((2, 10)), jnp.zeros((2, 192))
    params = network.init(jax.random.key(0), depth, valid, proprio, hidden)
    action, memory = network.apply(params, depth, valid, proprio, hidden)
    second, _ = network.apply(params, depth, valid, proprio, memory)
    assert action.shape == (2, 3) and memory.shape == hidden.shape
    assert not np.allclose(action, second)

    def objective(parameters):
        prediction, _ = network.apply(
            parameters, depth, valid, proprio, hidden, method=network.predict
        )
        return jnp.mean((prediction[..., :3] - 1) ** 2 + (prediction[..., 3:] - 2) ** 2)

    gradient = jax.grad(objective)(params)
    assert all(np.isfinite(x).all() for x in jax.tree.leaves(gradient))
    assert np.linalg.norm(gradient["params"]["output_projection"]["kernel"]) > 0


def test_depth_method_runs_a_real_scene_rollout_with_finite_policy_gradient():
    from drone_playground.composition import build_environment, compose_method, validate_config
    from drone_playground.learning.algorithms.pointcloud_bptt import initialize
    from drone_playground.learning.algorithms.pointcloud_navigation_bptt import rollout_loss

    config = compose_method(
        "learning/depth_navigation", overrides=["env.scene.scene_ids=[S01]", "training.num_envs=2"]
    )
    validate_config(config)
    task = build_environment(config, "cpu")
    try:
        state, network, _ = initialize(task, config)

        def loss(parameters):
            return rollout_loss(task, network, parameters, jax.random.key(7), 2, 2)

        (value, metrics), gradient = jax.jit(jax.value_and_grad(loss, has_aux=True))(state.params)
        assert np.isfinite(value) and np.isfinite(metrics["velocity_prediction_loss"])
        assert all(np.isfinite(x).all() for x in jax.tree.leaves(gradient))
        assert sum(float(jnp.sum(x * x)) for x in jax.tree.leaves(gradient)) > 0
        assert task.sensor_calibration["width"] == 64
        assert task.sensor_calibration["far_m"] == network.far_m == 10
    finally:
        task.close()
