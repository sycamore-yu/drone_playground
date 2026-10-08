"""Composition-level sensor/observation contract checks."""

import copy

import pytest

from drone_playground.configuration import compose_experiment, validate_config


def test_perception_observation_requires_a_sensor_component():
    config = compose_experiment("papers/dva")
    validate_config(config)
    stripped = copy.deepcopy(config)
    del stripped["env"]["sensor"]
    with pytest.raises(ValueError, match=r"Missing environment component: sensor"):
        validate_config(stripped)
