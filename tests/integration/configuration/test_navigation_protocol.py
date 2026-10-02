"""Navigation protocol checks that require a composed experiment."""

from copy import deepcopy

import pytest

from drone_playground.composition import compose_experiment, validate_config


def test_composition_rejects_navigation_protocol_changes():
    config = compose_experiment('navigation/ppo', "navigation/static")
    validate_config(config)
    changes = (
        ("duration", 30.0),
        ("goal_radius", 1.0),
        ("freq", 100),
        ("dynamic", True),
    )
    for field, value in changes:
        changed = deepcopy(config)
        changed["env"]["task"][field] = value
        with pytest.raises(ValueError):
            validate_config(changed)
