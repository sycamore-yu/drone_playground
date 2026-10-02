"""Perception contracts derived from current experiment composition."""

from drone_playground.composition import (
    compose_experiment,
    native_training_config,
    sensor_layout,
)


def test_depth_and_lidar_keep_sensor_identity_with_equal_input_width():
    depth = native_training_config(compose_experiment('papers/dva'))
    lidar = native_training_config(compose_experiment('navigation/ppo'))
    assert depth["observation_size"] == lidar["observation_size"] == 2420
    assert depth["sensor_layout"]["kind"] == "depth"
    assert depth["sensor_layout"]["grid"] == [20, 15]
    assert lidar["sensor_layout"]["kind"] == "lidar"
    assert lidar["sensor_layout"]["grid"] is None
    assert depth["sensor_layout"]["embedding_size"] == 128
    assert lidar["sensor_layout"]["embedding_size"] == 128


def test_dynamic_presets_keep_the_same_perception_contract():
    for experiment, static, dynamic in (
        ("papers/dva", "navigation/static", "navigation/dynamic"),
        ("navigation/ppo", "navigation/static", "navigation/dynamic"),
    ):
        static_config = compose_experiment(experiment, static)
        dynamic_config = compose_experiment(experiment, dynamic)
        assert sensor_layout(static_config) == sensor_layout(dynamic_config)
        assert static_config["network"] == dynamic_config["network"]
        assert static_config["objective"] == dynamic_config["objective"]
        assert static_config["method"] == dynamic_config["method"]
        assert static_config["env"]["action"] == dynamic_config["env"]["action"]
