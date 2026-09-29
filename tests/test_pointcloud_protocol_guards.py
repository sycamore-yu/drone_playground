"""Reject configuration labels that misrepresent the executed paper protocol."""

import pytest

from drone_playground.composition import validate_config
from tests.reference_configs import compose_reference as compose_config


@pytest.mark.parametrize("sensor_hz", [5.0, 20.0])
def test_synchronous_sensor_rate_must_match_the_executed_policy_tick(sensor_hz):
    config = compose_config("paper_pointcloud")
    config["env"]["sensor"]["source_rate_hz"] = sensor_hz
    with pytest.raises(ValueError, match="sensor.*frequency"):
        validate_config(config)


@pytest.mark.parametrize(
    ("field", "value"), [("network_output_frame", "world"), ("command_units", "normalized")]
)
def test_policy_coordinate_and_unit_labels_match_the_command_transform(field, value):
    config = compose_config("paper_pointcloud")
    config["method"][field] = value
    with pytest.raises(ValueError, match="Paper policy.*(frame|units)"):
        validate_config(config)


@pytest.mark.parametrize(
    ("group", "field", "value"),
    [
        ("task", "body_radius", 0.035),
        ("task", "goal_radius", 1.0),
        ("task", "physics_freq", 250),
        ("task", "duration", 20.0),
        ("evaluation", "duration", 20.0),
        ("evaluation", "speeds", [4.0]),
        ("evaluation", "scene_ids", ["S01", "D01"]),
        ("evaluation", "episodes", 2),
    ],
)
def test_navigation8_nominal_label_requires_the_frozen_protocol(group, field, value):
    config = compose_config("paper_pointcloud_navigation8")
    validate_config(config)
    (config["env"]["task"] if group == "task" else config[group])[field] = value
    with pytest.raises(ValueError, match="Selected navigation protocol"):
        validate_config(config)
