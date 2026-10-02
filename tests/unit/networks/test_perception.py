"""P5-04: matched perception encoder and PPO checkpoint contract."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from drone_playground.networks.perception import (
    PointFrameEncoder,
    SensorLayout,
    privileged_critic_fields,
)
from drone_playground.networks.factory import network_factory


def test_actor_and_critic_information_boundary_is_explicit():
    contract = privileged_critic_fields()
    assert contract["privileged_fields"] == []
    assert contract["actor_and_critic_share_observation_container"] is True
    assert contract["critic_uses_sensor"] is False
    assert contract["scene_manifest_visible_to_policy"] is False
    assert contract["future_obstacle_motion_visible_to_policy"] is False


def test_sensor_measurements_change_the_policy_output():
    layout = SensorLayout(
        kind="depth",
        proprioception_size=20,
        history=1,
        points_per_frame=4,
        channels=2,
        grid=(2, 2),
        embedding_size=8,
    )
    cfg = {
        "sensor_layout": layout.as_dict(),
        "hidden_sizes": [16],
        "distribution_type": "normal",
        "init_noise_std": 0.3,
    }
    networks = network_factory(cfg)((layout.total_size,), 4)
    params = networks.policy_network.init(jax.random.PRNGKey(0))
    base = jnp.zeros((1, layout.total_size), jnp.float32)
    changed = base.at[:, layout.proprioception_size :].set(1.0)
    a = networks.policy_network.apply(None, params, base)
    b = networks.policy_network.apply(None, params, changed)
    # NormalDistribution policy params are (mean, std). The std is shared;
    # the encoder must make the mean sensitive to the actual depth values.
    assert not np.array_equal(np.asarray(a[0]), np.asarray(b[0]))


def test_point_encoder_is_permutation_invariant():
    encoder = PointFrameEncoder(embedding_size=16)
    frames = jax.random.normal(jax.random.PRNGKey(5), (2, 1, 12, 5))
    params = encoder.init(jax.random.PRNGKey(6), frames)
    encoded = encoder.apply(params, frames)
    reversed_points = encoder.apply(params, frames[:, :, ::-1])
    np.testing.assert_allclose(encoded, reversed_points, atol=1e-6, rtol=1e-6)
