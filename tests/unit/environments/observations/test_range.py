"""A sensor-only nearest-range representation preserves obstacle direction and history."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def test_range_bins_preserve_nearest_return_and_ignore_invalid_finite_filler():
    from drone_playground.environments.observations.polar_range import polar_range_features

    # XYZ/range are normalized by the sensor's forty-meter range, as in MID360 input.
    frames = jnp.array(
        [
            [
                [0.025, 0.0, 0.0, 0.025, 1.0],
                [0.25, 0.0, 0.0, 0.25, 1.0],
                [0.0025, 0.0, 0.0, 0.0025, 0.0],
                [0.0, 0.05, 0.0, 0.05, 1.0],
            ]
        ]
    )
    encoded = polar_range_features(frames, azimuth_bins=24, elevation_bins=3, range_scale=40.0)
    assert encoded.shape == (1, 24 * 3 * 2)
    features = encoded.reshape(1, 24 * 3, 2)
    assert features[..., 1].sum() == 2
    assert features[..., 0].max() == pytest.approx(0.5, abs=1e-6)
    permutation = jnp.array([3, 1, 0, 2])
    np.testing.assert_array_equal(
        encoded, polar_range_features(frames[:, permutation], 24, 3, 40.0)
    )
    changed = frames.at[0, 0, :4].multiply(0.5)
    assert polar_range_features(changed, 24, 3, 40.0).max() >= encoded.max()
    assert (
        polar_range_features(changed, 24, 3, 40.0).reshape(1, 72, 2)[..., 0].max()
        > features[..., 0].max()
    )


def test_polar_encoder_keeps_empty_frames_zero_and_gradients_finite():
    from drone_playground.environments.observations.polar_range import polar_range_features

    zeros = jnp.zeros((2, 4, 20, 5))
    result = polar_range_features(zeros, 24, 3, 40.0)
    assert result.shape == (2, 4, 144)
    np.testing.assert_array_equal(result, 0.0)
    gradient = jax.grad(lambda x: polar_range_features(x, 24, 3, 40.0).sum())(zeros)
    assert np.isfinite(gradient).all()


def test_actor_and_critic_use_the_explicit_range_encoder_and_same_observation_contract():
    from drone_playground.networks.perception import SensorLayout, perception_network_factory

    layout = SensorLayout("lidar", 20, 4, 8, 5)
    config = dict(
        distribution_type="tanh_normal",
        sensor_encoder="polar_range",
        range_scale=40.0,
        hidden_sizes=[32, 32],
        critic_uses_sensor=True,
    )
    network = perception_network_factory(layout, config)(layout.total_size, 4)
    p = network.policy_network.init(jax.random.PRNGKey(1))
    v = network.value_network.init(jax.random.PRNGKey(2))
    # Direct sensor-only range features, with no hidden learned PointNet module.
    assert all(
        "SharedEncoder" not in str(path) for path, _ in jax.tree_util.tree_flatten_with_path(p)[0]
    )
    obs = jnp.zeros((1, layout.total_size))
    other = obs.at[:, 20:].set(jnp.tile(jnp.array([0.025, 0, 0, 0.025, 1.0]), (1, 32)))
    a, b = network.policy_network.apply(None, p, obs), network.policy_network.apply(None, p, other)
    assert not np.array_equal(a, b)
    a, b = network.value_network.apply(None, v, obs), network.value_network.apply(None, v, other)
    assert not np.array_equal(a, b)
    with pytest.raises(ValueError):
        perception_network_factory(layout, {**config, "sensor_encoder": "undeclared"})
