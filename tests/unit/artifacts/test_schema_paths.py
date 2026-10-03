"""Runtime loading accepts one current schema and never rewrites historical metadata."""

import copy

import pytest

from drone_playground.artifacts.schema import require_current
from drone_playground.configuration import load_config


def test_historical_schema_requires_explicit_migration():
    config = {"config_version": 3, "env": {"execution": {}}}
    original = copy.deepcopy(config)
    with pytest.raises(ValueError, match="migrate"):
        require_current(config)
    assert config == original


def test_current_config_remains_independent_after_validation():
    config = load_config("environment")
    result = require_current(config)
    assert result == config
    result["env"]["task"]["duration"] = 0.2
    assert config["env"]["task"]["duration"] != 0.2
