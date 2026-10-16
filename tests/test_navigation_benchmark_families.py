"""Native MJCF flight cases and Hydra domain-randomization presets."""

from pathlib import Path

import jax
import numpy as np
import pytest
from hydra import compose, initialize_config_dir

from drone_playground.simulation.factory import create_environment
from drone_playground.simulation.scene import Scene
from tools.build_navigation_scenes import FAMILIES, LEVELS, OUTPUT, build


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("level", LEVELS)
def test_native_flightbench_style_scene_compiles_and_has_safe_endpoints(family, level):
    """Each declared scene is baked, loadable, and preserves the flight endpoints."""
    name = f"{family}_{level}"
    assert (OUTPUT / f"{name}.xml").read_text() == build(family, level)
    scene = Scene(name)
    assert scene.model.ngeom >= 10
    assert float(scene.clearance(np.array([2.0, 0.0, 3.0]))) > 0.5
    assert float(scene.clearance(np.array([98.0, 0.0, 3.0]))) > 0.5
    assert len(scene.geometry_hash) == 64


def test_native_mazes_require_detours_and_dynamic_obstacles_move():
    """Maze walls block the centerline while mocap motion changes collision geometry."""
    maze = Scene("maze_hard")
    assert float(maze.clearance(np.array([15.0, 0.0, 3.0]))) < 0
    assert float(maze.clearance(np.array([15.0, 9.0, 3.0]))) > 0
    dynamic = Scene("dynamic_dense_easy")
    start = np.asarray(dynamic.positions(jax.numpy.asarray([0.0])))[0]
    moved = np.asarray(dynamic.positions(jax.numpy.asarray([1.0])))[0]
    assert np.linalg.norm(start - moved) > 0.01


@pytest.mark.parametrize("preset", ["nominal", "dynamics", "sim2real"])
def test_randomization_presets_compose_and_build_physical_environments(preset):
    """Hydra is the only source of preset values and the environment uses them."""
    with initialize_config_dir(
        version_base=None,
        config_dir=str(Path(__file__).parents[1] / "src/drone_playground/configs"),
    ):
        config = compose(
            config_name="config",
            overrides=[
                "experiment=oa_static_depth",
                f"randomization={preset}",
                "simulation.device=cpu",
                "simulation.num_envs=2",
            ],
        )
    env = create_environment(config)
    state = env.reset(jax.random.PRNGKey(27))
    if preset == "nominal":
        assert not config.simulation.randomization
        assert env.sensor.latency_bounds == (0.0, 0.0)
    elif preset == "dynamics":
        assert config.simulation.randomization.mass == [0.90, 1.10]
        assert env.sensor.latency_bounds == (0.0, 0.0)
    else:
        assert config.simulation.disturbance.gust_period_s == 0.2
        assert env.sensor.latency_bounds == (0.02, 0.04)
        assert config.simulation.action_delay_s == [0.025, 0.05]
    assert np.isfinite(np.asarray(state.physics.states.pos)).all()
