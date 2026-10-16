"""Geometric reference paths must detour around contacts and report unreachable cases."""

import numpy as np

from drone_playground.benchmark.geometry import ReferencePath, traversability
from drone_playground.simulation.scene import Scene


def test_reference_path_avoids_obstacle_and_rejects_blocked_endpoint(tmp_path):
    """The reference has finite body clearance, not just a straight-line lower bound."""
    path = tmp_path / "obstacle.xml"
    path.write_text(
        '<mujoco><worldbody><geom type="box" pos="2 2 1" size="0.3 0.7 2"/></worldbody></mujoco>'
    )
    scene = Scene(path)
    reference = ReferencePath(scene, [0, 0, 0], [4, 4, 2], 0.1, resolution=0.25)
    length = reference.length([0.5, 2, 1], [3.5, 2, 1])
    assert length > 3.0
    assert reference.length([0.5, 2, 1], [2, 2, 1]) is None
    assert reference.length([0.5, 0.5, 1], [3.5, 0.5, 1]) == 3.0


def test_traversability_uses_repeatable_sampling_and_body_diameter():
    """Empty-space complexity remains bounded by the declared flight volume."""
    scene = Scene()
    args = (scene, [0, 0, 0], [4, 4, 2])
    small = traversability(*args, 0.1, times=(0.0,))
    large = traversability(*args, 0.2, times=(0.0,))
    assert small == traversability(*args, 0.1, times=(0.0,))
    assert small["rays"] == 20 * 8 * 3 * 16
    np.testing.assert_allclose(small["value"], 2 * large["value"])
