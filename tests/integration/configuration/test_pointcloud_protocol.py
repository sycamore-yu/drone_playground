"""Reject configuration labels that misrepresent the executed paper protocol."""

import pytest

from drone_playground.configuration import compose_experiment, validate_config
from drone_playground.environments.factory import build_environment


@pytest.mark.parametrize("sensor_hz", [5.0, 20.0])
def test_synchronous_sensor_rate_must_match_the_executed_policy_tick(sensor_hz):
    config = compose_experiment("papers/differentiable_pointcloud")
    config["env"]["sensor"]["source_rate_hz"] = sensor_hz
    with pytest.raises(ValueError, match=r"sensor.*frequency"):
        build_environment(config, device="cpu")


@pytest.mark.parametrize(
    ("field", "value"), [("network_output_frame", "world"), ("command_units", "normalized")]
)
def test_policy_coordinate_and_unit_labels_match_the_command_transform(field, value):
    config = compose_experiment("papers/differentiable_pointcloud")
    config["method"][field] = value
    with pytest.raises(ValueError, match=r"policy.*(frame|units)"):
        build_environment(config, device="cpu")


@pytest.mark.parametrize(
    ("group", "field", "value"),
    [
        ("task", "body_radius", 0.035),
        ("task", "goal_radius", 1.0),
        ("env", "physics_freq", 250),
        ("task", "duration", 20.0),
        ("evaluation", "duration", 20.0),
        ("evaluation", "speeds", [21.0]),
    ],
)
def test_selected_navigation_protocol_owns_its_constraints(group, field, value):
    config = compose_experiment(
        "navigation/differentiable_pointcloud_acceleration_benchmark", overrides=["mode=eval"]
    )
    validate_config(config)
    (config["env"]["task"] if group == "task" else config[group])[field] = value
    with pytest.raises(ValueError, match=r"Benchmark protocol"):
        validate_config(config)


def test_generic_navigation_adapter_accepts_custom_conditions_without_a_benchmark_label():
    config = compose_experiment("navigation/differentiable_pointcloud", overrides=["mode=eval"])
    config["evaluation"]["protocol"] = None
    config["env"]["task"].update(goal_radius=0.3, duration=20.0)
    config["evaluation"].update(duration=20.0, episodes=2, speeds=[4.0])
    validate_config(config)
