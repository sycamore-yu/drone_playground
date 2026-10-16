"""A static-density intervention keeps dynamic scenes and sampled geometry matched."""

import importlib.util
from pathlib import Path

import jax
import numpy as np
import pytest
from hydra import compose, initialize_config_dir

from drone_playground.simulation.procedural_scene import ProceduralScene


def test_static_scale_reduces_only_static_masks_and_preserves_default_identity():
    """Match both layouts while reducing static counts by thirty percent."""
    original = ProceduralScene()
    explicit_default = ProceduralScene({"static_obstacle_scale": 1.0})
    reduced = ProceduralScene({"static_obstacle_scale": 0.7})
    assert original.geometry_hash == explicit_default.geometry_hash
    assert original.geometry_hash != reduced.geometry_hash
    key = jax.random.PRNGKey(1_100_000)
    before = jax.jit(lambda k: original.sample(k, 32))(key)
    after = jax.jit(lambda k: reduced.sample(k, 32))(key)
    dynamic = np.asarray(before.moving.any(-1))
    assert dynamic.any() and (~dynamic).any()
    counts = np.asarray(before.active.sum(-1))
    actual = np.asarray(after.active.sum(-1))
    np.testing.assert_array_equal(actual[dynamic], counts[dynamic])
    assert np.all(np.abs(actual[~dynamic] - counts[~dynamic] * 0.7) <= 1.00001)
    for field in ("centers", "sizes", "target", "speed", "phase", "cylinder", "moving", "goal"):
        np.testing.assert_array_equal(getattr(before, field), getattr(after, field))
    np.testing.assert_array_equal(after.active[dynamic], before.active[dynamic])


@pytest.mark.parametrize("scale", [0, -0.1, 1.1, True, float("nan"), float("inf")])
def test_invalid_static_scales_fail_before_sampling(scale):
    """Reject unsupported density before any geometry is generated."""
    with pytest.raises(ValueError, match="static_obstacle_scale"):
        ProceduralScene({"static_obstacle_scale": scale})


def test_static_density_queue_changes_only_static_scale_from_distance_baseline():
    """Resolved training configurations differ solely in static obstacle count."""
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "density_scheduler", root / "tools/optimize_paper_ppo.py"
    )
    scheduler = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scheduler)
    with initialize_config_dir(
        version_base=None, config_dir=str(root / "src/drone_playground/configs")
    ):
        for family in ("depth", "lidar"):
            scheduler.configure_distance_search()
            baseline = compose(
                config_name="config",
                overrides=scheduler.trial_options(f"{family}_d20", 0, Path("/tmp/static-control")),
            )
            scheduler.configure_static_density_search()
            assert len(scheduler.CASES) == 2
            lighter = compose(
                config_name="config",
                overrides=scheduler.trial_options(
                    f"{family}_static_light_d20", 0, Path("/tmp/static-control")
                ),
            )
            assert lighter.simulation.procedural.static_obstacle_scale == 0.7
            lighter.output = baseline.output
            del lighter.simulation.procedural.static_obstacle_scale
            assert lighter == baseline
