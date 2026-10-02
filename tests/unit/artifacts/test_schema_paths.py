"""Retained v3 snapshots follow moved source paths without rewriting their files."""

import copy

from drone_playground.artifacts.schema import require_current


def test_execution_layout_is_normalized_without_mutating_recorded_values():
    saved = {
        "config_version": 3,
        "env": {"execution": {"dynamics": {
            "_target_": "drone_playground.models.lotf.LOTFModel",
            "forward": "lotf_simplified", "drone": "example_quad",
        }, "command": "thrust_bodyrates", "controller": {
            "_target_": "drone_playground.execution.controllers.bodyrates.BodyRateControl",
            "name": "bodyrates",
        }, "tracker": None}},
    }
    original = copy.deepcopy(saved)
    normalized = require_current(saved)
    assert saved == original
    assert "execution" not in normalized["env"]
    assert normalized["env"]["dynamics"]["forward"] == "lotf_simplified"
    assert normalized["env"]["dynamics"]["_target_"] == "drone_playground.dynamics.lotf.LOTFModel"
    assert normalized["env"]["action"]["command"] == "thrust_bodyrates"
    assert require_current(normalized) == normalized


def test_old_sensor_targets_keep_their_distinct_ray_models():
    cfg = {"config_version": 3, "env": {"sensor": {
        "_target_": "drone_playground.environments.sensors.pointcloud.UniformMid360Lidar",
        "azimuth_count": 12, "elevation_count": 3,
    }}}
    target = require_current(cfg)["env"]["sensor"]
    assert target["_target_"] == "drone_playground.environments.sensors.lidar.UniformRayLidar"
    assert target["azimuth_count"] == 12
