"""Validate the public v3 assembly rather than the names of paper algorithms."""

import importlib
from pathlib import Path

import pytest


def test_runtime_modules_have_single_responsibility_locations():
    root = Path(__file__).resolve().parents[1] / "src/drone_playground"
    required = (
        "methods/neural.py",
        "methods/optimal_control/sampling.py",
        "networks/policies.py",
        "environments/tasks/navigation.py",
        "environments/scenes/navigation.py",
        "environments/sensors/lidar.py",
        "execution/transition.py",
        "models/point_mass.py",
        "learning/algorithms/pointcloud_bptt.py",
        "runtime/jax_runner.py",
        "runtime/host_runner.py",
        "visualization/rscope_io.py",
    )
    assert all((root / path).is_file() for path in required)


@pytest.mark.parametrize(
    "environment", ["hovering", "tracking", "racing", "navigation/static", "navigation/dynamic"]
)
def test_ppo_composes_for_each_core_environment(environment):
    module = importlib.import_module("drone_playground.composition")
    assert hasattr(module, "compose_method"), "A method/environment assembly entry is required"
    cfg = module.compose_method("learning/ppo", environment)
    module.validate_config(cfg)
    assert cfg["config_version"] == 3
    assert cfg["method"]["trainable"]
    assert cfg["algorithm"]["name"] == "ppo"
    assert cfg["env"]["name"] == environment
    assert set(("scene", "task", "sensor", "observation", "execution")) <= set(cfg["env"])
    assert "backward" not in cfg["env"]["execution"]["dynamics"]
    assert "gradient" in cfg["algorithm"]


@pytest.mark.parametrize(
    "method",
    ["paper/super", "paper/ego_planner", "optimization/attitude_mpc", "optimization/sampling_mpc"],
)
def test_optimization_recipe_exposes_modes_before_any_run_is_created(method, tmp_path):
    module = importlib.import_module("drone_playground.composition")
    assert hasattr(module, "compose_method")
    cfg = module.compose_method(method)
    assert not cfg["method"]["trainable"]
    cfg["mode"] = "train"
    with pytest.raises(ValueError, match="训练阶段|trainable|training stage"):
        module.run_experiment(cfg, tmp_path, "must-not-exist")
    assert not (tmp_path / "experiments/must-not-exist").exists()


def test_sensor_and_forward_model_have_one_config_owner():
    module = importlib.import_module("drone_playground.composition")
    assert hasattr(module, "compose_method")
    cfg = module.compose_method("paper/super", "navigation/dynamic")
    assert cfg["env"]["sensor"]["name"] == "mid360"
    assert "sensor" not in cfg["env"]["observation"]
    assert "dynamics" not in cfg
    assert cfg["method"]["output"] == "trajectory"
    assert cfg["env"]["execution"]["tracker"]["name"] == "trajectory_tracking"


def test_pointcloud_and_lotf_have_separate_source_and_training_identities():
    module = importlib.import_module("drone_playground.composition")
    assert hasattr(module, "compose_method")
    paper = module.compose_method("paper/pointcloud_flight")
    lotf = module.compose_method("paper/lotf")
    assert paper["method"]["name"] != lotf["method"]["name"]
    assert paper["algorithm"]["name"] == "pointcloud_bptt"
    assert lotf["algorithm"]["name"] == "lotf_bptt"
    assert paper["env"]["execution"]["dynamics"]["forward"] == "point_mass_lag"
    assert paper["algorithm"]["gradient"]["transition"] == "exponential"


def test_aero_mppi_identity_does_not_claim_a_sampling_mpc_implementation():
    root = Path(__file__).resolve().parents[1]
    path = root / "third_party/sources.yaml"
    assert path.is_file(), "Source identities must be independent from executable implementations"
    import yaml

    sources = yaml.safe_load(path.read_text())
    assert sources["aero_mppi"]["title"].startswith("AERO-MPPI")
    assert sources["loong"]["title"].startswith("LOONG")


def test_component_group_reselection_is_not_shadowed_by_a_recipe_copy():
    from drone_playground.composition import compose_method, validate_config

    cfg = compose_method(
        "learning/ppo",
        "tracking",
        ["algorithm=apg", "network=brax_apg", "training.num_timesteps=null"],
    )
    assert cfg["algorithm"]["name"] == "apg"
    validate_config(cfg)


def test_mpc_implementation_and_problem_adapter_must_match():
    from drone_playground.composition import compose_method, validate_config

    cfg = compose_method("optimization/attitude_mpc")
    cfg["method"]["decision"]["name"] = "sampling_mpc"
    with pytest.raises(ValueError, match="adapter|适配"):
        validate_config(cfg)


@pytest.mark.parametrize(
    "old,method,environment",
    [
        ("figure8_ppo", "learning/ppo", "tracking"),
        ("random_ppo", "learning/ppo", "tracking/random"),
        ("racing_ppo", "learning/ppo", "racing"),
        ("racing_attitude_mpc", "optimization/attitude_mpc", "racing"),
        ("racing_sampling_mpc", "optimization/sampling_mpc", "racing"),
        ("p5_static_super", "paper/super", "navigation/static"),
        ("p5_static_ego", "paper/ego_planner", "navigation/static"),
        ("lotf_hybrid_hover", "paper/lotf", "paper/lotf_hover"),
        ("lotf_hybrid_tracking", "paper/lotf", "paper/lotf_tracking"),
        ("paper_pointcloud", "paper/pointcloud_flight", "paper/pointcloud_flight"),
    ],
)
def test_public_environment_preserves_the_qualified_source_model_and_protocol(
    old, method, environment
):
    from drone_playground.composition import compose_method
    from tests.reference_configs import compose_reference

    assert compose_method(method, environment)["env"] == compose_reference(old)["env"]
