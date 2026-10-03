"""Validate the public v3 assembly rather than the names of paper algorithms."""

import importlib

import pytest

from tests.helpers.paths import REPO_ROOT


def test_runtime_modules_have_single_responsibility_locations():
    root = REPO_ROOT / "src/drone_playground"
    required = (
        "control/wrappers.py",
        "control/controllers/mpc/sampling.py",
        "networks/policies.py",
        "environments/tasks/navigation/task.py",
        "environments/scenes/catalog.py",
        "environments/sensors/lidar.py",
        "control/transition.py",
        "dynamics/point_mass.py",
        "learning/algorithms/recurrent_bptt.py",
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
    assert hasattr(module, "compose_experiment"), "A method/environment assembly entry is required"
    cfg = module.compose_experiment("control/ppo", environment)
    module.validate_config(cfg)
    assert cfg["config_version"] == 4
    assert cfg["method"]["trainable"]
    assert cfg["algorithm"]["name"] == "ppo"
    assert cfg["env"]["name"] == environment
    assert set(("scene", "task", "sensor", "reference", "controller", "dynamics")) <= set(
        cfg["env"]
    )
    assert "backward" not in cfg["env"]["dynamics"]
    assert "gradient" in cfg["algorithm"]


@pytest.mark.parametrize(
    "method",
    ["papers/super", "papers/ego_planner", "control/attitude_mpc", "control/sampling_mpc"],
)
def test_optimization_recipe_exposes_modes_before_any_run_is_created(method, tmp_path):
    module = importlib.import_module("drone_playground.composition")
    assert hasattr(module, "compose_experiment")
    cfg = module.compose_experiment(method)
    assert not cfg["method"]["trainable"]
    cfg["mode"] = "train"
    with pytest.raises(ValueError, match=r"training"):
        module.run_experiment(cfg, tmp_path, "must-not-exist")
    from drone_playground.artifacts.layout import find_experiment

    assert find_experiment(tmp_path, "must-not-exist") is None


def test_sensor_and_forward_model_have_one_config_owner():
    module = importlib.import_module("drone_playground.composition")
    assert hasattr(module, "compose_experiment")
    cfg = module.compose_experiment("papers/super", "navigation/dynamic")
    assert cfg["env"]["sensor"]["name"] == "mid360"
    assert "sensor" not in cfg["env"]["task"]["observation"]
    assert "dynamics" not in cfg
    assert cfg["method"]["output"] == "trajectory"
    assert cfg["env"]["controller"]["_target_"].endswith(".TrajectoryTracking")


def test_source_dynamics_does_not_create_a_private_method():
    from drone_playground.composition import compose_experiment
    from tests.helpers.configs import bodyrates_config

    reconstruction = compose_experiment("papers/differentiable_pointcloud")
    cfg = bodyrates_config()
    assert reconstruction["env"]["dynamics"]["forward"] == "point_mass_lag"
    assert cfg["algorithm"]["name"] == "bptt"
    assert cfg["method"]["implementation"] == "neural"
    assert cfg["env"]["task"]["name"] == "hovering"


def test_aero_mppi_identity_does_not_claim_a_sampling_mpc_implementation():
    root = REPO_ROOT
    path = root / "patches/sources.json"
    assert path.is_file(), "Source identities must be independent from executable implementations"
    import yaml

    sources = yaml.safe_load(path.read_text())
    assert sources["aero_mppi"]["title"].startswith("AERO-MPPI")
    assert sources["loong"]["title"].startswith("LOONG")


def test_component_group_reselection_is_not_shadowed_by_a_recipe_copy():
    from drone_playground.composition import compose_experiment, validate_config

    cfg = compose_experiment(
        "control/ppo",
        "tracking",
        ["algorithm=apg", "network=mlp_32", "training.num_timesteps=null"],
    )
    assert cfg["algorithm"]["name"] == "apg"
    validate_config(cfg)


def test_mpc_constructor_is_explicit_and_does_not_dispatch_by_display_name():
    from drone_playground.composition import compose_experiment, validate_config

    cfg = compose_experiment("control/attitude_mpc")
    target = cfg["method"]["decision"]["_target_"]
    cfg["method"]["decision"]["name"] = "sampling_mpc"
    validate_config(cfg)
    assert cfg["method"]["decision"]["_target_"] == target
    assert target.endswith(".build_attitude")
