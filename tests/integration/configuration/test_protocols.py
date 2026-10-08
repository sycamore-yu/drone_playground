"""A named benchmark locks physical task conditions before any run is created."""

import copy

import pytest

from drone_playground.configuration import compose_experiment, validate_config


def test_named_navigation_protocol_is_checked_and_recorded_by_identity():
    from drone_playground.benchmarks import protocol_identity

    cfg = compose_experiment(
        "papers/super", "navigation/static", ["+evaluation.protocol=benchmarks/navigation.yaml"]
    )
    validate_config(cfg)
    identity = protocol_identity(cfg)
    assert identity["name"] == "navigation" and identity["version"] == 2
    assert identity["catalog"].endswith("catalog.xml")
    changed = copy.deepcopy(cfg)
    changed["env"]["task"]["goal_radius"] = 0.8
    with pytest.raises(ValueError, match=r"protocol|协议"):
        validate_config(changed)


def test_named_protocol_does_not_require_a_hash_for_a_local_asset(tmp_path):
    from drone_playground.environments.scenes.catalog import DEFAULT_CATALOG

    cfg = compose_experiment(
        "papers/super", "navigation/static", ["+evaluation.protocol=benchmarks/navigation.yaml"]
    )
    source = DEFAULT_CATALOG
    changed = tmp_path / "catalog.xml"
    changed.write_bytes(source.read_bytes() + b"\n")
    cfg["env"]["scene"]["catalog_path"] = str(changed)
    validate_config(cfg)
