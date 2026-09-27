"""P5-04: matched perception encoder and PPO checkpoint contract."""

from __future__ import annotations

import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from brax.training.acme import running_statistics, specs

from drone_playground.composition import compose_config, native_training_config, sensor_layout
from drone_playground.learning.networks import network_factory
from drone_playground.learning.perception import SensorLayout, privileged_critic_fields
from drone_playground.runs.checkpoints import load_policy, save_policy


def native(name: str) -> dict:
    config = compose_config(name)
    return native_training_config(config)


def test_depth_and_lidar_share_policy_width_but_keep_sensor_identity():
    depth = native("p5_static_depth_ppo")
    lidar = native("p5_static_lidar_ppo")
    assert depth["observation_size"] == lidar["observation_size"] == 2420
    assert depth["sensor_layout"]["kind"] == "depth"
    assert depth["sensor_layout"]["grid"] == [20, 15]
    assert lidar["sensor_layout"]["kind"] == "lidar"
    assert lidar["sensor_layout"]["grid"] is None
    assert depth["hidden_sizes"] == lidar["hidden_sizes"] == [128, 128]
    assert depth["network"]["name"] if "network" in depth else True


def test_actor_and_critic_information_boundary_is_explicit():
    contract = privileged_critic_fields()
    assert contract["privileged_fields"] == []
    assert contract["actor_and_critic_share_observation_container"] is True
    assert contract["critic_uses_sensor"] is False
    assert contract["scene_manifest_visible_to_policy"] is False
    assert contract["future_obstacle_motion_visible_to_policy"] is False


def test_sensor_measurements_change_the_policy_output():
    cfg = native("p5_static_depth_ppo")
    layout = SensorLayout.from_dict(cfg["sensor_layout"])
    networks = network_factory(cfg)((layout.total_size,), 4)
    params = networks.policy_network.init(jax.random.PRNGKey(0))
    base = jnp.zeros((1, layout.total_size), jnp.float32)
    changed = base.at[:, layout.proprioception_size :].set(1.0)
    a = networks.policy_network.apply(None, params, base)
    b = networks.policy_network.apply(None, params, changed)
    # NormalDistribution policy params are (mean, std). The std is shared;
    # the encoder must make the mean sensitive to the actual depth values.
    assert not np.array_equal(np.asarray(a[0]), np.asarray(b[0]))


def test_perception_checkpoint_round_trip_rebuilds_the_same_network():
    resolved = compose_config("p5_static_lidar_ppo")
    cfg = native_training_config(resolved)
    layout = SensorLayout.from_dict(cfg["sensor_layout"])
    networks = network_factory(cfg)((layout.total_size,), 4)
    normalizer = running_statistics.init_state(specs.Array((layout.total_size,), jnp.float32))
    params = (
        normalizer,
        networks.policy_network.init(jax.random.PRNGKey(1)),
        networks.value_network.init(jax.random.PRNGKey(2)),
    )
    from brax.training.agents.ppo import networks as ppo_networks

    obs = jax.random.uniform(jax.random.PRNGKey(3), (2, layout.total_size))
    expected = ppo_networks.make_inference_fn(networks)(params, deterministic=True)(
        obs, jax.random.PRNGKey(4)
    )[0]
    with tempfile.TemporaryDirectory() as tmp:
        path = save_policy(Path(tmp), params, cfg, 17)
        maker, loaded, meta = load_policy(path)
        actual = maker(loaded, deterministic=True)(obs, jax.random.PRNGKey(4))[0]
    np.testing.assert_array_equal(actual, expected)
    assert meta["step"] == 17
    assert meta["config"]["observation"]["name"] == "navigation_lidar"


def test_dynamic_presets_keep_the_same_matched_perception_contract():
    for static, dynamic in (
        ("p5_static_depth_ppo", "p5_dynamic_depth_ppo"),
        ("p5_static_lidar_ppo", "p5_dynamic_lidar_ppo"),
    ):
        a, b = compose_config(static), compose_config(dynamic)
        assert sensor_layout(a) == sensor_layout(b)
        assert a["network"] == b["network"]
        assert a["objective"] == b["objective"]
        assert a["policy"] == b["policy"]
        assert a["dynamics"] == b["dynamics"]
