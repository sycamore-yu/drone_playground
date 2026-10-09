"""Build and install a real wheel, then exercise it outside the source checkout."""

import os
import subprocess
import sys
import venv
from pathlib import Path
from zipfile import ZipFile

import pytest

WHEEL_SMOKE = """
import importlib
import sys
from pathlib import Path

import drone_playground
import jax
import jax.numpy as jnp
import numpy as np
from drone_playground.simulation.networks import Actor
from drone_playground.simulation.scene import Scene

package = Path(drone_playground.__file__).resolve().parent
assert package.is_relative_to(Path(sys.argv[1]).resolve()), package
assert all(device.platform == 'cpu' for device in jax.devices())
for module in (
    'simulation.environment', 'simulation.methods', 'simulation.observation',
    'simulation.sensors', 'simulation.tasks', 'simulation.policy',
    'simulation.runner', 'simulation.replay', 'simulation.evaluation',
    'simulation.ros_planner', 'simulation.ros_planner.client',
    'simulation.ros_planner.planner_pb2', 'simulation.ros_planner.planner_pb2_grpc',
    'learning.trainer', 'learning.checkpoint', 'learning.ppo', 'learning.apg',
    'learning.shac', 'cli',
):
    imported = importlib.import_module('drone_playground.' + module)
    assert Path(imported.__file__).resolve().is_relative_to(package)
for name in ('empty', 'S01', 'S02', 'S03', 'S06', 'D01', 'D02', 'D03', 'D06', 'racing'):
    scene = Scene(name)
    assert len(scene.geometry_hash) == 64
    if name != 'empty':
        assert scene.xml_path.resolve().is_relative_to(package / 'assets' / 'scenes')
        assert scene.model.ngeom > 0
    assert scene.positions(jnp.array([0., 2.5])).shape == (2, scene.model.ngeom, 3)
    print('scene', name, scene.geometry_hash, flush=True)
obs = {
    'state': jnp.zeros((1, 18)), 'depth': jnp.ones((1, 12, 16, 1)),
    'points': jnp.ones((1, 8, 3)), 'mask': jnp.ones((1, 8), dtype=bool),
}
memory = jnp.zeros((1, 192))
for kind in ('state', 'depth', 'lidar'):
    actor = Actor(kind)
    parameters = actor.init(jax.random.key(0), obs, memory)
    action, carry, velocity = actor.apply(parameters, obs, memory)
    assert action.shape == (1, 4 if kind == 'state' else 3)
    assert carry.shape == (1, 192) and velocity.shape == (1, 3)
    assert all(np.isfinite(value).all() for value in (action, carry, velocity))
    print('actor', kind, action.shape, flush=True)
print('wheel imports, 10 scenes and 3 actors passed', flush=True)
"""


@pytest.fixture(scope="module")
def installed_wheel(tmp_path_factory):
    """Build and install the package in a venv outside the source checkout."""
    root = Path(__file__).resolve().parents[1]
    outside = tmp_path_factory.mktemp("installed-wheel")
    assert not outside.is_relative_to(root)
    env = dict(os.environ, JAX_PLATFORMS="cpu", PYTHONDONTWRITEBYTECODE="1")
    env.pop("PYTHONPATH", None)

    def run(command):
        result = subprocess.run(
            command, cwd=outside, env=env, text=True, capture_output=True, timeout=300
        )
        assert result.returncode == 0, result.stdout + result.stderr
        return result.stdout

    run([sys.executable, "-m", "build", "--wheel", "--outdir", str(outside), str(root)])
    (wheel,) = outside.glob("*.whl")
    with ZipFile(wheel) as archive:
        members = set(archive.namelist())
    for source, target in (
        (root / "assets/scenes", "drone_playground/assets/scenes"),
        (root / "src/drone_playground/configs", "drone_playground/configs"),
    ):
        for file in source.rglob("*"):
            if file.is_file() and file.suffix in {".xml", ".png", ".yaml"}:
                assert f"{target}/{file.relative_to(source).as_posix()}" in members
    prefix = outside / "venv"
    venv.EnvBuilder(system_site_packages=True, with_pip=False).create(prefix)
    python = prefix / "bin/python"
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "--python",
            str(python),
            "install",
            "--no-deps",
            str(wheel),
        ]
    )
    return run, prefix, python


def test_wheel_imports_scenes_and_actors(installed_wheel):
    """Load all required scenes and actors from the installed wheel."""
    run, prefix, python = installed_wheel
    output = run([str(python), "-I", "-c", WHEEL_SMOKE, str(prefix)])
    assert "wheel imports, 10 scenes and 3 actors passed" in output


@pytest.mark.parametrize(
    ("experiment", "task", "sensor"),
    [
        ("tracking", "tracking", "state"),
        ("racing", "racing", "state"),
        ("navigation_depth", "navigation", "depth"),
        ("navigation_lidar", "navigation", "lidar"),
    ],
)
def test_wheel_hydra_entrypoint(installed_wheel, experiment, task, sensor):
    """Resolve task, sensor and learning recipes through the installed CLI."""
    from omegaconf import OmegaConf

    run, prefix, _ = installed_wheel
    for algorithm in ("ppo", "apg", "shac"):
        output = run(
            [
                str(prefix / "bin/drone-playground"),
                "--cfg",
                "job",
                "--resolve",
                f"experiment={experiment}",
                f"learning={algorithm}",
                "simulation.device=cpu",
            ]
        )
        config = OmegaConf.create(output)
        assert config.task.name == task
        assert config.sensor.name == sensor
        assert config.learning.algorithm == algorithm
        assert config.simulation.device == "cpu"
