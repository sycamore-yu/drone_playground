"""A named benchmark locks physical task conditions before any run is created."""

import copy

import pytest

from drone_playground.composition import compose_experiment, validate_config
from tests.helpers.paths import REPO_ROOT


def test_named_navigation_protocol_is_checked_and_recorded_by_identity():
    from drone_playground.benchmarks import protocol_identity

    cfg = compose_experiment('papers/super', "navigation/static", ["+evaluation.protocol=benchmarks/navigation.yaml"])
    validate_config(cfg)
    identity = protocol_identity(cfg)
    assert identity["name"] == "navigation" and identity["version"] == 2
    assert len(identity["sha256"]) == 64
    changed = copy.deepcopy(cfg)
    changed["env"]["task"]["goal_radius"] = 0.8
    with pytest.raises(ValueError, match=r"protocol|协议"):
        validate_config(changed)


def test_named_protocol_rejects_geometry_changes_before_startup(tmp_path):
    cfg = compose_experiment('papers/super', "navigation/static", ["+evaluation.protocol=benchmarks/navigation.yaml"])
    source = REPO_ROOT / "assets/scenes/navigation/catalog.json"
    changed = tmp_path / "catalog.json"
    changed.write_bytes(source.read_bytes() + b"\n")
    cfg["env"]["scene"]["catalog_path"] = str(changed)
    with pytest.raises(ValueError, match=r"catalog|geometry|几何"):
        validate_config(cfg)
