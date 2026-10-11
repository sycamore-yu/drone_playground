"""Tests for repeatable, isolated reward-ablation job admission and contracts."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


@pytest.fixture
def scheduler():
    """Load the scheduler module without starting a manager or GPU process."""
    source = Path(__file__).parents[1] / "tools/train_navigation_reward_ablations.py"
    spec = importlib.util.spec_from_file_location("_navigation_reward_ablation_scheduler", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_reference_proposal_reuses_trial9_recipe_without_weights(tmp_path, scheduler):
    """Remove only original experiment, seed and output, never training coefficients."""
    target = tmp_path / ".scheduler/proposals/depth_ppo_s0_trial9.json"
    target.parent.mkdir(parents=True)
    target.write_text(
        json.dumps(
            {
                "from_scratch": True,
                "initial_checkpoint": None,
                "command": [
                    "python",
                    "-m",
                    "drone_playground.cli",
                    "mode=train",
                    "experiment=navigation_procedural_depth_ppo_validation",
                    "learning=ppo",
                    "learning.options.lr=0.0002",
                    "initial_checkpoint=null",
                    "resume=null",
                    "seed=0",
                    "output=/old/dir",
                ],
            }
        )
    )
    options, digest = scheduler.reference_options(tmp_path, "depth")
    assert options == [
        "learning=ppo",
        "learning.options.lr=0.0002",
        "initial_checkpoint=null",
        "resume=null",
    ]
    assert len(digest) == 64
    args = scheduler.overrides("depth", "b", 0, "/new/location", options)
    assert args[1] == "experiment=navigation_procedural_depth_ppo_ablation_b"
    assert "initial_checkpoint=null" in args
    assert args[-1] == "output=/new/location"


@pytest.mark.parametrize(
    "sensor,letter",
    [
        ("depth", "a"),
        ("depth", "b"),
        ("depth", "d"),
        ("lidar", "a"),
        ("lidar", "c"),
        ("lidar", "d"),
    ],
)
def test_exactly_one_variable_can_change_in_hydra_pair(scheduler, sensor, letter):
    """Use the actual Hydra config rather than comparing authored YAML fragments."""
    config_dir = str((Path(__file__).parents[1] / "src/drone_playground/configs").resolve())
    source_options = [
        "learning=ppo",
        "initial_checkpoint=null",
        "resume=null",
        "++learning.options.progress_reward_scale=0.5",
        "++learning.options.height_boundary_weight=1.0",
    ]
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        control = OmegaConf.to_container(
            compose(
                config_name="config",
                overrides=scheduler.overrides(
                    sensor, letter, 0, "/same/output", source_options, baseline=True
                ),
            ),
            resolve=True,
        )
        treatment = OmegaConf.to_container(
            compose(
                config_name="config",
                overrides=scheduler.overrides(sensor, letter, 0, "/same/output", source_options),
            ),
            resolve=True,
        )
    scheduler.validate_pair(control, treatment, letter)
    assert treatment["simulation"]["procedural"]["start_sampling"] == "collision_free"
    assert treatment["benchmark"]["fixed_checkpoint_seeds"] is True
    assert treatment["results"]["keep_best_screening"] is True


def test_scheduler_admits_one_pilot_per_arm_before_repeating_seed(tmp_path, scheduler):
    """A failed seed-zero trial does not consume two extra training seeds."""
    state = scheduler.cases_status(tmp_path)
    assert scheduler.chosen_candidate(state) == ("depth", "b", 0)
    trial = tmp_path / "depth_b_s0"
    trial.mkdir()
    (trial / "run.json").write_text(json.dumps({"status": "early_stopped"}))
    state = scheduler.cases_status(tmp_path)
    assert scheduler.chosen_candidate(state) == ("lidar", "c", 0)
    assert state["depth_b"][0]["status"] == "early_stopped"


def test_scheduler_requires_real_benchmark_acceptance_for_repeat_seeds(tmp_path, scheduler):
    """Only an accepted full benchmark authorizes seed-one and seed-two pilots."""
    for sensor, letter in scheduler.CASES:
        first = tmp_path / f"{sensor}_{letter}_s0"
        first.mkdir()
        (first / "run.json").write_text(json.dumps({"status": "early_stopped"}))
    state = scheduler.cases_status(tmp_path)
    assert scheduler.chosen_candidate(state) is None

    accepted = tmp_path / "depth_b_s0/run.json"
    accepted.write_text(json.dumps({"status": "accepted"}))
    state = scheduler.cases_status(tmp_path)
    assert scheduler.chosen_candidate(state) == ("depth", "b", 1)


def test_scheduler_prioritizes_acceptance_replication_over_new_reward_pilots(tmp_path, scheduler):
    """A formally passed seed-zero candidate gets the next free GPU slot."""
    passed = tmp_path / "lidar_c_s0"
    passed.mkdir()
    (passed / "run.json").write_text(json.dumps({"status": "accepted"}))
    state = scheduler.cases_status(tmp_path)
    assert scheduler.chosen_candidate(state) == ("lidar", "c", 1)
