"""Navigation protocol checks that require a composed experiment."""

from copy import deepcopy

import pytest

from drone_playground.configuration import compose_experiment, validate_config


def test_composition_rejects_navigation_protocol_changes():
    config = compose_experiment("navigation/ppo", "navigation/static")
    validate_config(config)
    changes = (
        ("task", "duration", 30.0),
        ("task", "goal_radius", 1.0),
        (None, "freq", 100),
        ("scene", "scene_ids", ["undeclared-scene"]),
    )
    for group, field, value in changes:
        changed = deepcopy(config)
        target = changed["env"] if group is None else changed["env"][group]
        target[field] = value
        with pytest.raises(ValueError):
            validate_config(changed)
