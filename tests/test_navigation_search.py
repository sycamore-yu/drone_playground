"""Four-architecture search must preserve real evidence and bounded budgets."""

import importlib.util
import json
import subprocess
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


def test_four_architectures_require_fresh_matched_baselines(search):
    """Historical Fan/VisFly runs with different controls cannot replace fresh controls."""
    assert hasattr(search, "configure_navigation_search"), "Unified search preset missing"
    search.configure_navigation_search()
    assert {value[0] for value in search.CASES.values()} == {"depth", "lidar", "fan", "visfly"}
    assert len(search.CASES) == 28
    assert search.BASE_UNITS == {}
    records = {f"{case}_s{i}": {"status": "missing"} for case in search.CASES for i in range(3)}
    records["fan_base_s0"]["status"] = "running"
    records["visfly_base_s0"]["status"] = "early_stopped"
    assert search.next_trial(records) == ("depth_base", 0)
    records["depth_base_s0"]["status"] = "running"
    assert search.next_trial(records) == ("lidar_base", 0)


def test_accepted_modality_stops_competing_recipe_replication(search):
    """Three accepted Depth seeds prevent another visual recipe from consuming seeds."""
    search.configure_navigation_search()
    records = {f"{case}_s{i}": {"status": "missing"} for case in search.CASES for i in range(3)}
    for seed in range(3):
        records[f"depth_base_s{seed}"]["status"] = "accepted"
    records["visfly_base_s0"]["status"] = "accepted"
    assert search.next_trial(records) == ("lidar_base", 0)


def test_interrupted_family_does_not_prevent_independent_baseline(search):
    """Preserve the interrupted run while allowing unrelated sensor work."""
    search.configure_navigation_search()
    records = {f"{case}_s{i}": {"status": "missing"} for case in search.CASES for i in range(3)}
    records["depth_base_s0"]["status"] = "interrupted"
    assert search.next_trial(records) == ("lidar_base", 0)


def test_search_confirms_real_acceptance_before_other_candidates(search):
    """An accepted seed gets replication; failures and probe scores do not."""
    assert hasattr(search, "configure_navigation_search"), "Unified search preset missing"
    search.configure_navigation_search()
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
    search.configure_navigation_search()
    for case, (_, _, parent) in search.CASES.items():
        if parent is not None:
            if case in {
                "lidar_privileged",
                "depth_mixed_privileged",
                "depth_mixed_goal_privileged",
            }:
                # Mutually exclusive flags encode one choice of Critic input.
                assert search.CASE_OPTIONS[case] == [
                    "learning.options.critic_uses_sensor=false",
                    "learning.options.critic_uses_privileged=true",
                ]
            else:
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
        "depth_mixed",
        "depth_mixed_lowlr",
        "depth_mixed_goal",
        "depth_mixed_privileged",
        "depth_mixed_goal_privileged",
        "lidar_mixed",
        "depth_world_rotation",
        "lidar_world_rotation",
        "lidar_batch64",
        "lidar_privileged",
        "lidar_goal_privileged",
        "lidar_velocity1",
        "lidar_horizon128",
        "lidar_task_only",
        "depth_fixed_heading",
        "lidar_fixed_heading",
        "fan_exploration",
        "visfly_hoverthrust",
    ],
)
def test_hydra_resolves_one_treatment_with_unchanged_acceptance(search, case):
    """Check resolved configs rather than only trusting override strings."""
    search.configure_navigation_search()
    parent = search.CASES[case][2]
    config_dir = str(Path(__file__).parents[1] / "src/drone_playground/configs")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        control = compose(config_name="config", overrides=search.trial_options(parent, 0, "/same"))
        candidate = compose(config_name="config", overrides=search.trial_options(case, 0, "/same"))
    control = OmegaConf.to_container(control, resolve=True)
    candidate = OmegaConf.to_container(candidate, resolve=True)
    control_samples = control["simulation"]["num_envs"] * control["learning"]["options"]["horizon"]
    candidate_samples = (
        candidate["simulation"]["num_envs"] * candidate["learning"]["options"]["horizon"]
    )
    for field in ("evaluation_interval", "checkpoint_interval"):
        assert (
            control["learning"][field] * control_samples
            == candidate["learning"][field] * candidate_samples
        )
        candidate["learning"][field] = control["learning"][field]
    assert (
        candidate["early_stopping"]["min_updates"] * candidate_samples
        == control["early_stopping"]["min_updates"] * control_samples
    )
    candidate["early_stopping"]["min_updates"] = control["early_stopping"]["min_updates"]
    for option in search.CASE_OPTIONS[case]:
        setting = option.split("=", 1)[0].lstrip("+").split(".")
        old, new = control, candidate
        for part in setting[:-1]:
            old, new = old[part], new[part]
        assert old.get(setting[-1]) != new[setting[-1]]
        if setting[-1] in old:
            new[setting[-1]] = old[setting[-1]]
        else:
            del new[setting[-1]]
    assert control == candidate
    assert candidate["simulation"]["procedural"]["min_start_goal_distance_m"] == (
        0.0 if case.startswith("depth_mixed_") else 80.0
    )
    assert candidate["benchmark"]["fixed_checkpoint_seeds"] is True
    assert candidate["results"]["keep_best_screening"] is True
    assert candidate["initial_checkpoint"] is None and candidate["resume"] is None


def test_idle_depth_slot_precedes_more_lidar_trials(search):
    """A busy LiDAR queue cannot starve an eligible Depth treatment."""
    search.configure_navigation_search()
    records = {f"{case}_s{i}": {"status": "missing"} for case in search.CASES for i in range(3)}
    records["depth_base_s0"] = {
        "status": "early_stopped",
        "best": {"value": 0.1},
        "latest": {"value": 0.0, "outcomes": {"OUT_OF_BOUNDS": 31, "COLLISION": 17}},
    }
    records["depth_lowlr_s0"]["status"] = "early_stopped"
    records["lidar_base_s0"]["status"] = "running"
    assert search.next_trial(records) == ("depth_goal", 0)


def test_measured_parent_and_failure_mode_change_candidate_order(search):
    """Changing evidence, not the candidate list, changes the next proposal."""
    search.configure_navigation_search()
    records = {f"{case}_s{i}": {"status": "missing"} for case in search.CASES for i in range(3)}
    for case in ("depth_base", "lidar_base"):
        records[f"{case}_s0"] = {
            "status": "early_stopped",
            "best": {"value": 0.0},
            "latest": {"value": 0.0, "outcomes": {"COLLISION": 48}},
        }
    records["depth_mixed_s0"] = {"status": "early_stopped", "best": {"value": 0.13}}
    assert search.next_trial(records) == ("depth_mixed_lowlr", 0)
    records["depth_mixed_lowlr_s0"]["status"] = "running"
    assert search.next_trial(records) == ("lidar_goal", 0)


def test_regression_controls_precede_more_paper_trials_and_wait_for_both_factors(search):
    """Measured regression promotes its controls; combinations wait for main effects."""
    search.configure_navigation_search()
    records = {f"{case}_s{i}": {"status": "missing"} for case in search.CASES for i in range(3)}
    for case in ("depth_base", "lidar_base", "lidar_privileged"):
        records[f"{case}_s0"]["status"] = "early_stopped"
    records["depth_mixed_s0"] = {
        "status": "budget_exhausted",
        "best": {"value": 0.1354166667},
        "latest": {"value": 0.0, "outcomes": {"OUT_OF_BOUNDS": 35, "COLLISION": 13}},
    }
    assert search.next_trial(records) == ("depth_mixed_goal", 0)
    records["depth_mixed_goal_s0"]["status"] = "running"
    assert search.next_trial(records) == ("depth_mixed_privileged", 0)
    records["depth_mixed_goal_s0"]["status"] = "early_stopped"
    records["depth_mixed_goal_s0"]["latest"] = {"value": 0.0}
    records["depth_mixed_privileged_s0"]["status"] = "running"
    assert search.next_trial(records) != ("lidar_goal_privileged", 0)
    records["lidar_goal_s0"] = {"status": "pruned", "latest": {"value": 0.0}}
    records["lidar_privileged_s0"]["latest"] = {"value": 0.0}
    assert search.next_trial(records) == ("lidar_goal_privileged", 0)
    records["lidar_goal_privileged_s0"]["status"] = "running"
    records["depth_mixed_privileged_s0"]["status"] = "early_stopped"
    assert search.next_trial(records) != ("depth_mixed_goal_privileged", 0)
    records["depth_mixed_privileged_s0"]["latest"] = {"value": 0.0}
    assert search.next_trial(records) == ("depth_mixed_goal_privileged", 0)


def test_lidar_budget_uses_full_screens_and_preserves_late_depth_learning(search):
    """Two stagnant screens stop LiDAR at equal samples, never a late-learning Depth run."""
    search.configure_navigation_search()
    for per_update in (512, 2048):
        record = {
            "status": "running",
            "updates": 1_105_920 // per_update,
            "interactions": 1_105_920,
            "screens": [
                {"checkpoint_update": steps // per_update, "value": 0.0}
                for steps in (368_640, 737_280, 1_105_920)
            ],
        }
        assert search.prune_lidar_candidate("lidar_base", record)
        assert not search.prune_lidar_candidate("depth_mixed", record)
        assert not search.prune_lidar_candidate("fan_base", record)
        record["screens"][-1]["value"] = 0.01
        assert not search.prune_lidar_candidate("lidar_base", record)
        record["screens"] = record["screens"][:2]
        assert not search.prune_lidar_candidate("lidar_base", record)


def test_optimizer_recovers_once_and_records_pruned_latest_not_peak(search, tmp_path):
    """Recovery must retain the failed tail and must not duplicate a completed trial."""
    oa = pytest.importorskip("optim_agent")
    search.configure_navigation_search()
    directory = tmp_path / "ledger"
    directory.mkdir()
    run = tmp_path / "lidar_base_s0"
    run.mkdir()
    (run / "trial.json").write_text(
        json.dumps(
            {
                "family": "lidar",
                "seed": 0,
                "argv": ["python", "-u", "-m", "drone_playground.cli"],
            }
        )
    )
    records = {
        "lidar_base_s0": {
            "directory": str(run),
            "status": "pruned",
            "best": {"value": 0.1},
            "latest": {"value": 0.0},
            "screens": [
                {"checkpoint_update": 1200, "value": 0.1},
                {"checkpoint_update": 2160, "value": 0.0},
            ],
        }
    }
    study = oa.create_study(direction="maximize", storage=directory / "study.json", seed=0)
    search.sync_optim_study(study, directory, records)
    resumed = oa.create_study(direction="maximize", storage=directory / "study.json", seed=0)
    search.sync_optim_study(resumed, directory, records)
    assert len(resumed.trials) == 1
    assert resumed.trials[0].state == "pruned"
    assert resumed.trials[0].value == 0.0
    assert resumed.trials[0].intermediate[-1] == (2160, 0.0)


def test_reallocation_before_first_screen_has_no_fabricated_optimizer_score(search, tmp_path):
    """An intentionally canceled pilot remains unmeasured, including after recovery."""
    oa = pytest.importorskip("optim_agent")
    directory = tmp_path / "ledger"
    directory.mkdir()
    run = tmp_path / "visfly_lowstd_s0"
    run.mkdir()
    (run / "trial.json").write_text(
        json.dumps(
            {"family": "visfly", "seed": 0, "argv": ["python", "-u", "-m", "drone_playground.cli"]}
        )
    )
    records = {
        "visfly_lowstd_s0": {
            "directory": str(run),
            "status": "pruned",
            "best": None,
            "latest": None,
            "screens": [],
        }
    }
    for _ in range(2):
        study = oa.create_study(direction="maximize", storage=directory / "study.json", seed=0)
        search.sync_optim_study(study, directory, records)
    assert len(study.trials) == 1
    assert study.trials[0].state == "pruned"
    assert study.trials[0].value is None
    assert study.trials[0].intermediate == []


def test_restart_retries_pending_prune_signal_once(search, tmp_path, monkeypatch):
    """A crash between writing the request and signaling cannot lose the stop request."""
    search.configure_navigation_search()
    marker = tmp_path / "study_pruned.json"
    marker.write_text(json.dumps({"policy": search.ALLOCATION_POLICY}))
    calls = []
    monkeypatch.setattr(search.subprocess, "run", lambda argv, **kwargs: calls.append(argv))
    records = {"lidar_base_s0": {"status": "running", "directory": str(tmp_path)}}
    search.apply_search_pruning(records)
    search.apply_search_pruning(records)
    assert len(calls) == 1 and "--signal=SIGINT" in calls[0]
    assert json.loads(marker.read_text())["signal_sent_at_utc"]


def test_launch_timeout_preserves_worker_identity(search, tmp_path, monkeypatch):
    """A timed-out systemd client is not proof that the trainer failed to start."""
    root = tmp_path / "results"
    root.mkdir()
    identity = root / "fan_margin2_s0/run.json"
    worker = {"status": "running", "config_sha256": "worker-config", "git_commit": "source"}

    def run(argv, **kwargs):
        if argv[0] == "systemd-run":
            identity.write_text(json.dumps(worker))
            raise subprocess.TimeoutExpired(argv, 30)
        return subprocess.CompletedProcess(argv, 0, stdout="resolved")

    monkeypatch.setattr(search.subprocess, "run", run)
    monkeypatch.setattr(search.subprocess, "check_output", lambda *a, **k: "source\n")
    search.launch(tmp_path, root, Path("/python"), "fan_margin2", 0)
    assert json.loads(identity.read_text()) == worker
    assert json.loads((identity.parent / "launch_error.json").read_text())["status"] == "unknown"


def test_reserved_proposal_reuses_exact_command_after_restart(search, tmp_path, monkeypatch):
    """An interrupted ask/launch window resumes its saved configuration without overwrite."""
    root = tmp_path / "results"
    root.mkdir()
    commands = []

    def run(argv, **kwargs):
        if argv[0] == "systemd-run":
            commands.append(argv)
            if len(commands) == 1:
                raise KeyboardInterrupt
        return subprocess.CompletedProcess(argv, 0, stdout="resolved")

    monkeypatch.setattr(search.subprocess, "run", run)
    monkeypatch.setattr(search.subprocess, "check_output", lambda *a, **k: "source\n")
    monkeypatch.setattr(search, "active_units", set)
    with pytest.raises(KeyboardInterrupt):
        search.launch(tmp_path, root, Path("/python"), "fan_margin2", 0)
    trial = root / "fan_margin2_s0/trial.json"
    original = trial.read_bytes()
    records = search.collect(root, {})
    assert records["fan_margin2_s0"]["status"] == "reserved"
    assert search.next_trial(records) == ("fan_margin2", 0)
    search.launch(tmp_path, root, Path("/different-python"), "fan_margin2", 0)
    assert trial.read_bytes() == original
    assert commands[0] == commands[1]


def test_stable_acceptance_requires_three_seeds_of_the_same_recipe(search):
    """Do not combine unrelated lucky seeds into a supposedly stable policy."""
    search.configure_navigation_search()
    records = {f"{case}_s{i}": {"status": "missing"} for case in search.CASES for i in range(3)}
    for key in ("depth_base_s0", "depth_lowlr_s1", "visfly_lowstd_s2"):
        records[key]["status"] = "accepted"
    assert search.stable_modalities(records) == set()
    for seed in range(3):
        records[f"depth_base_s{seed}"]["status"] = "accepted"
    assert search.stable_modalities(records) == {"depth"}
