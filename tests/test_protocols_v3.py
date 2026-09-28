"""A named benchmark locks physical task conditions before any run is created."""

import copy
from pathlib import Path

import pytest

from drone_playground.composition import compose_method, validate_config


def test_named_navigation_protocol_is_checked_and_recorded_by_identity():
    from drone_playground.evaluation.protocols import protocol_identity

    cfg = compose_method("paper/super", "navigation/static", ["evaluation=navigation_v1"])
    validate_config(cfg)
    identity = protocol_identity(cfg)
    assert identity["name"] == "navigation" and identity["version"] == 1
    assert len(identity["sha256"]) == 64
    changed = copy.deepcopy(cfg)
    changed["env"]["task"]["goal_radius"] = 0.8
    with pytest.raises(ValueError, match="protocol|协议"):
        validate_config(changed)


def test_named_protocol_rejects_geometry_changes_before_startup(tmp_path):
    cfg = compose_method("paper/super", "navigation/static", ["evaluation=navigation_v1"])
    source = Path(__file__).parents[1] / "assets/scenes/navigation/catalog.json"
    changed = tmp_path / "catalog.json"
    changed.write_bytes(source.read_bytes() + b"\n")
    cfg["env"]["scene"]["catalog_path"] = str(changed)
    with pytest.raises(ValueError, match="catalog|geometry|几何"):
        validate_config(cfg)
