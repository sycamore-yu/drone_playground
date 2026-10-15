"""The stability gate requires both per-scene quality and original-case retention."""

import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "stability_window", Path(__file__).resolve().parents[1] / "tools/ppo_stability_window.py"
)
WINDOW = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WINDOW)


def cases():
    """Six scenes with the exact independent physical initialization identities."""
    return {
        (scene, "geometry", "1500000", str(world)): "SUCCESS"
        for scene in WINDOW.SCENES
        for world in range(25)
    }


def test_budget_rounding_covers_original_collapse_scale():
    """The actual rollout size never rounds the agreed budgets down."""
    assert WINDOW.window_steps(64 * 512) == (28, 4)
    with pytest.raises(ValueError, match="positive integer"):
        WINDOW.window_steps(0)


def test_per_scene_pass_is_insufficient_when_retention_fails():
    """Eight lost cases fail; seven lost cases meet the95percent retention gate."""
    original = cases()
    current = dict(original)
    for scene in WINDOW.SCENES[:4]:
        for world in (0, 1):
            current[(scene, "geometry", "1500000", str(world))] = "TIMEOUT"
    result = WINDOW.quality(current, original)
    assert min(result["per_scene_successes"].values()) == 23
    assert result["retained_initial_successes"] == 142
    assert not result["passed"]
    current[("D01", "geometry", "1500000", "1")] = "SUCCESS"
    assert WINDOW.quality(current, original)["passed"]


def test_retention_pass_is_insufficient_when_one_scene_fails():
    """A weak scene cannot hide behind high aggregate retention."""
    original = cases()
    current = dict(original)
    for world in range(3):
        current[("S03", "geometry", "1500000", str(world))] = "COLLISION"
    result = WINDOW.quality(current, original)
    assert result["retention_fraction"] == 0.98
    assert result["per_scene_successes"]["S03"] == 22
    assert not result["passed"]


def test_changed_case_identity_cannot_masquerade_as_retention():
    """An equally successful different cohort cannot demonstrate preservation."""
    original = cases()
    current = dict(original)
    event = current.pop(("S01", "geometry", "1500000", "0"))
    current[("S01", "geometry", "2000000", "0")] = event
    with pytest.raises(ValueError, match="exact original"):
        WINDOW.quality(current, original)
