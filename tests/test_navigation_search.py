"""Four-architecture search must preserve real evidence and bounded budgets."""

import importlib.util
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


@pytest.fixture
def search():
    """Load the scheduler without starting any worker."""
    path = Path(__file__).parents[1] / "tools/optimize_paper_ppo.py"
    spec = importlib.util.spec_from_file_location("navigation_search", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_four_architectures_are_explicit_without_restarting_external_baselines(search):
    """The new preset includes old CNN/PointNet and both paper representations."""
    assert hasattr(search, "configure_navigation_search"), "Unified search preset missing"
    search.configure_navigation_search(Path("/results"))
    assert {value[0] for value in search.CASES.values()} == {"depth", "lidar", "fan", "visfly"}
    assert len(search.CASES) == 12
    assert search.ADOPTED_RUNS["fan_base_s0"].name == "fan_margin2_s0"
    assert search.ADOPTED_RUNS["visfly_base_s0"].name == "visfly_shared_s0"
    records = {f"{case}_s{i}": {"status": "missing"} for case in search.CASES for i in range(3)}
    records["fan_base_s0"]["status"] = "running"
    records["visfly_base_s0"]["status"] = "early_stopped"
    assert search.next_trial(records) == ("depth_base", 0)
    records["depth_base_s0"]["status"] = "running"
    assert search.next_trial(records) == ("lidar_base", 0)


def test_search_confirms_real_acceptance_before_other_candidates(search):
    """An accepted seed gets replication; failures and probe scores do not."""
    assert hasattr(search, "configure_navigation_search"), "Unified search preset missing"
    search.configure_navigation_search(Path("/results"))
    records = {f"{case}_s{i}": {"status": "missing"} for case in search.CASES for i in range(3)}
    records["depth_base_s0"]["status"] = "accepted"
    assert search.next_trial(records) == ("depth_base", 1)
    records["depth_base_s1"]["status"] = "early_stopped"
    assert search.next_trial(records) != ("depth_base", 2)


def test_controls_use_environment_interactions_not_the_same_update_count(search):
    """Different horizons receive explicit time and sample budgets."""
    assert hasattr(search, "interaction_controls"), "Interaction budget missing"
    old = search.interaction_controls("depth", num_envs=16, horizon=32)
    fan = search.interaction_controls("fan", num_envs=16, horizon=128)
    visfly = search.interaction_controls("visfly", num_envs=16, horizon=256)
    assert old["evaluation_interval"] * 512 == fan["evaluation_interval"] * 2048
    assert old["evaluation_interval"] * 512 == visfly["evaluation_interval"] * 4096
    assert old["max_interactions"] > old["min_updates"] * 512
    assert fan["min_updates"] * 2048 >= 4_000_000
    assert visfly["min_updates"] * 4096 >= 1_000_000


def test_unobserved_candidate_scores_stay_unknown(search, tmp_path):
    """No report must not be substituted with a measured zero-success score."""
    assert search.full_screen_evidence(tmp_path) is None


def test_parameter_changes_are_isolated_per_candidate(search):
    """A candidate changes one recipe field, not an undisclosed parameter bundle."""
    assert hasattr(search, "configure_navigation_search"), "Unified search preset missing"
    search.configure_navigation_search(Path("/results"))
    for case, (_, _, parent) in search.CASES.items():
        if parent is not None:
            assert len(search.CASE_OPTIONS[case]) == 1, case


@pytest.mark.parametrize(
    "case",
    [
        "depth_lowlr",
        "lidar_lowlr",
        "fan_lowlr",
        "visfly_lowstd",
        "depth_goal",
        "lidar_goal",
        "fan_exploration",
        "visfly_hoverthrust",
    ],
)
def test_hydra_resolves_one_treatment_with_unchanged_acceptance(search, case):
    """Check resolved configs rather than only trusting override strings."""
    search.configure_navigation_search(Path("/results"))
    parent = search.CASES[case][2]
    config_dir = str(Path(__file__).parents[1] / "src/drone_playground/configs")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        control = compose(config_name="config", overrides=search.trial_options(parent, 0, "/same"))
        candidate = compose(config_name="config", overrides=search.trial_options(case, 0, "/same"))
    control = OmegaConf.to_container(control, resolve=True)
    candidate = OmegaConf.to_container(candidate, resolve=True)
    setting = search.CASE_OPTIONS[case][0].split("=", 1)[0].split(".")
    old, new = control, candidate
    for part in setting[:-1]:
        old, new = old[part], new[part]
    assert old[setting[-1]] != new[setting[-1]]
    new[setting[-1]] = old[setting[-1]]
    assert control == candidate
    assert candidate["simulation"]["procedural"]["min_start_goal_distance_m"] == 80.0
    assert candidate["benchmark"]["fixed_checkpoint_seeds"] is True
    assert candidate["results"]["keep_best_screening"] is True
    assert candidate["initial_checkpoint"] is None and candidate["resume"] is None


def test_stable_acceptance_requires_three_seeds_of_the_same_recipe(search):
    """Do not combine unrelated lucky seeds into a supposedly stable policy."""
    search.configure_navigation_search(Path("/results"))
    records = {f"{case}_s{i}": {"status": "missing"} for case in search.CASES for i in range(3)}
    for key in ("depth_base_s0", "depth_lowlr_s1", "visfly_lowstd_s2"):
        records[key]["status"] = "accepted"
    assert search.stable_modalities(records) == set()
    for seed in range(3):
        records[f"depth_base_s{seed}"]["status"] = "accepted"
    assert search.stable_modalities(records) == {"depth"}
