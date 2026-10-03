"""Exercise the composed recipes and moved responsibilities through real entry points."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hydra.utils import get_object

from drone_playground.composition import compose_experiment, validate_config
from drone_playground.configuration import CONFIG_ROOT


def _targets(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key in ("_target_", "trainer", "policy_evaluator", "entrypoint"):
                yield item
            yield from _targets(item)
    elif isinstance(value, list):
        for item in value:
            yield from _targets(item)


RECIPES = sorted(
    path.relative_to(CONFIG_ROOT / "experiment").with_suffix("").as_posix()
    for path in (CONFIG_ROOT / "experiment").rglob("*.yaml")
)


@pytest.mark.parametrize("recipe", RECIPES)
def test_every_recipe_resolves_to_real_implementations(recipe):
    overrides = ["runtime.device=cpu"]
    if recipe == "navigation/native":
        overrides += [
            "method.algorithm=configuration_test",
            "method.deployment.address=127.0.0.1:50051",
        ]
    cfg = compose_experiment(recipe, overrides=overrides)
    validate_config(cfg)
    for target in _targets(cfg):
        assert callable(get_object(target)), target


def test_benchmark_reads_real_locked_files_outside_repository_cwd(tmp_path, monkeypatch):
    from drone_playground.benchmarks import load_protocol, protocol_identity, protocol_scenes

    monkeypatch.chdir(tmp_path)
    cfg = compose_experiment("navigation/ppo")
    identity = protocol_identity(cfg)
    protocol = load_protocol("benchmarks/navigation.yaml")
    assert protocol["timeout"]["duration_s"] == 300.0
    assert protocol_scenes(protocol) == ["S01", "S02", "S03", "S06", "D01", "D02", "D03", "D06"]
    from drone_playground.environments.scenes.catalog import DEFAULT_CATALOG, verified_geometry

    _, actual = verified_geometry(DEFAULT_CATALOG)
    assert actual == identity["mjcf_sha256"]


@pytest.mark.parametrize(
    "forward",
    [
        "so_rpy",
        "so_rpy_rotor",
        "so_rpy_rotor_drag",
        "first_principles",
        "lotf_high_fidelity",
        "lotf_simplified",
    ],
)
def test_dynamics_randomization_and_actual_container_step(forward):
    from drone_playground.dynamics.crazyflow import CrazyflowModel
    from drone_playground.dynamics.lotf import LOTFModel

    constructor = LOTFModel if forward.startswith("lotf_") else CrazyflowModel
    backend = constructor(
        forward=forward,
        backward="direct",
        domain_randomization={
            "enabled": True,
            "dynamics": {"mass": [1.1, 1.1]},
        },
    )
    from drone_playground.control.setpoints import AttitudeSetpoint, RateSetpoint
    from drone_playground.environments.initialization import initialize_simulation

    reference = initialize_simulation(backend, 0.04, 50, "cpu", [0.0, 0.0, 1.5])
    try:
        original = reference.sim.default_data
        sampled = backend.randomize(original, jax.random.PRNGKey(7))
        np.testing.assert_allclose(sampled.params.mass, original.params.mass * 1.1)
        control = (
            RateSetpoint(thrust=1.0, body_rates=jnp.zeros(3))
            if forward.startswith("lotf_")
            else AttitudeSetpoint(rpy=jnp.zeros(3), thrust=0.3)
        )
        following = jax.jit(lambda data: backend.step(data, control, 0.002))(sampled)
        assert np.isfinite(following.states.pos).all()
        assert np.isfinite(following.states.vel).all()
    finally:
        reference.close()


def test_point_mass_native_step_keeps_its_own_state_type():
    from drone_playground.dynamics.point_mass import PointMassLag, PointMassState

    backend = PointMassLag()
    state = PointMassState.create(jnp.array([[0.0, 0.0, 1.5]]))
    from drone_playground.control.setpoints import StateSetpoint

    following = backend.step(state, StateSetpoint(acceleration=jnp.zeros((1, 3))), 0.02)
    assert isinstance(following, PointMassState)
    assert np.isfinite(following.pos).all()


def test_reference_adapter_does_not_compose_a_learning_experiment(monkeypatch):
    from drone_playground import composition

    cfg = compose_experiment(
        "control/differentiable_pointcloud_hovering", overrides=["runtime.device=cpu"]
    )

    def unexpected(*args, **kwargs):
        raise AssertionError("Environment construction called experiment composition")

    monkeypatch.setattr(composition, "compose_experiment", unexpected)
    task = composition.build_environment(cfg, "cpu", "eval", 1)
    try:
        assert task.task.name == "hovering"
        assert task.action_size == 3
    finally:
        task.close()
