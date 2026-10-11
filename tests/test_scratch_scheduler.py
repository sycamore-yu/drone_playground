"""Guard the independent frozen audit and terminal Patience handoff."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir


@pytest.fixture
def scheduler():
    """Load the script without running its CLI or GPU-admission loop."""
    path = Path(__file__).resolve().parents[1] / "tools/train_scratch_matrix.py"
    spec = importlib.util.spec_from_file_location("_scratch_scheduler_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write(path, contents):
    """Write an isolated synthetic evidence record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(contents) if not isinstance(contents, str) else contents)


def test_verified_first_checkpoint_frozen_audit_counts_without_rewriting_old_run(
    scheduler, tmp_path
):
    """Count only original scratch weights with an independent frozen pass."""
    run = tmp_path / "depth_shac_s1"
    write(run / "config.yaml", "initial_checkpoint: null\n")
    write(run / "run.json", {"status": "interrupted", "layout_version": 2})
    policy = run / "checkpoints/step-000320/policy.zip"
    write(policy, "original-network-weights")
    write(
        policy.parent / "report.json",
        {
            "mode": "checkpoint_eval",
            "complete": True,
            "passed": True,
            "seed_base": 1_000_000,
        },
    )
    audit = tmp_path / ".scheduler/frozen_audits/depth_shac_s1"
    write(
        audit / "audit.json",
        {
            "status": "accepted",
            "checkpoint": "checkpoints/step-000320/policy.zip",
        },
    )
    write(
        audit / "report.json",
        {
            "mode": "benchmark",
            "complete": True,
            "passed": True,
            "seed_base": 2_000_000,
        },
    )
    assert scheduler.trial_status(run, {}) == "accepted"
    write(
        audit / "report.json",
        {
            "mode": "checkpoint_eval",
            "complete": True,
            "passed": True,
            "seed_base": 1_000_000,
        },
    )
    assert scheduler.trial_status(run, {}) == "invalid"


def test_failed_frozen_audit_is_terminal_and_new_early_stop_is_not_resumed(scheduler, tmp_path):
    """Neither failed tests nor Patience should trigger same-run extension."""
    run = tmp_path / "lidar_apg_s0"
    write(run / "run.json", {"status": "interrupted"})
    write(
        tmp_path / ".scheduler/frozen_audits/lidar_apg_s0/audit.json",
        {"status": "failed", "checkpoint": "checkpoints/step-000480/policy.zip"},
    )
    assert scheduler.trial_status(run, {}) == "frozen_benchmark_failed"
    new = tmp_path / "lidar_ppo_s0_trial4"
    write(new / "run.json", {"status": "early_stopped"})
    assert scheduler.trial_status(new, {}) == "early_stopped"


def test_retire_interrupted_legacy_and_pre_patience_instead_of_auto_resume(scheduler, tmp_path):
    """Old three-pass ZIP and first-pass runs without Patience stay archived."""
    old = tmp_path / "depth_ppo_s0"
    write(
        old / "config.yaml",
        "initial_checkpoint: null\nlearning:\n"
        "  selection: last_of_first_three_consecutive_passes\n",
    )
    write(old / "run.json", {"status": "interrupted", "layout_version": 2})
    write(old / "checkpoints/latest.training.zip", "original saved training state")
    assert scheduler.trial_status(old, {}) == "legacy_retired"

    obsolete_v3 = tmp_path / "lidar_ppo_s0_trial3"
    write(
        obsolete_v3 / "config.yaml",
        "initial_checkpoint: null\nlearning:\n  selection: first_full_validation_pass\n",
    )
    write(obsolete_v3 / "run.json", {"status": "interrupted", "layout_version": 3})
    write(obsolete_v3 / "checkpoints/model_000800/training/commit_success.txt", "ok")
    write(obsolete_v3 / "checkpoints/model_000800/metadata.json", "{}")
    assert scheduler.trial_status(obsolete_v3, {}) == "pre_patience_retired"


def test_new_patience_trial_can_resume_only_from_own_orbax_state(scheduler, tmp_path):
    """A new run resumes on unexpected interruption, not after Patience."""
    run = tmp_path / "depth_ppo_s0_trial3"
    write(
        run / "config.yaml",
        "initial_checkpoint: null\nlearning:\n"
        "  selection: first_full_validation_pass\nearly_stopping:\n"
        "  min_updates: 800\n",
    )
    write(run / "run.json", {"status": "interrupted", "layout_version": 3})
    write(run / "checkpoints/model_000800/training/commit_success.txt", "ok")
    write(run / "checkpoints/model_000800/metadata.json", "{}")
    write(
        tmp_path / ".scheduler/proposals/depth_ppo_s0_trial3.json",
        {
            "from_scratch": True,
            "initial_checkpoint": None,
            "command": ["python", "-m", "drone_playground.cli", "initial_checkpoint=null"],
        },
    )
    assert scheduler.trial_status(run, {}) == "recoverable"
    write(run / "run.json", {"status": "early_stopped", "layout_version": 3})
    assert scheduler.trial_status(run, {}) == "early_stopped"


def test_procedural_regime_never_uses_fixed_navigation_training_recipes(scheduler, tmp_path):
    """New protocol is isolated in another root and fixed Benchmark is unchanged."""
    cases = {case.key: case for case in scheduler.cases("procedural")}
    assert len(cases) == 12
    for task in ("depth", "lidar"):
        for algorithm in ("ppo", "apg", "shac"):
            case = cases[f"{task}_{algorithm}"]
            overrides = scheduler.overrides(case, 0)
            assert f"experiment=navigation_procedural_{task}" in " ".join(overrides)
            assert "initial_checkpoint=null" in overrides
            assert "resume=null" in overrides
    status = scheduler.report(tmp_path, {}, 1, regime="procedural")
    assert status["regime"] == "procedural"
    assert status["matrix_cells"] == 18
    fixed = scheduler.report(tmp_path, {}, 1)
    assert fixed["regime"] == "fixed"
    assert fixed["matrix_cells"] == 18


def test_procedural_racing_uses_separate_stable_recipes(scheduler):
    """New racing candidates must not silently re-use archived failed settings."""
    jobs = {case.key: case for case in scheduler.cases("procedural")}
    assert jobs["racing_ppo"].experiment == "racing_ppo_stable"
    assert jobs["racing_ppo"].horizon == 32
    assert jobs["racing_shac"].experiment == "racing_shac_stable"
    assert jobs["racing_shac"].learning_rate == 0.0001


def test_adaptive_procedural_candidate_never_switches_to_fixed_scenes(scheduler, tmp_path):
    """After nine failures, train on procedural maps, not benchmark geometry."""
    cases = {case.key: case for case in scheduler.cases("procedural")}
    for name in ("lidar_ppo", "depth_ppo"):
        case = cases[name]
        options, parent = scheduler.adaptive_overrides(case, 9, tmp_path, 0)
        assert parent == 0
        assert f"experiment=navigation_procedural_{case.task}_ppo_validation" in options
        assert not any("navigation_lidar_scratch" in option for option in options)
        assert not any(option.startswith("learning.scene_updates=") for option in options)
        assert "initial_checkpoint=null" in options


def test_ppo_candidate_ten_changes_only_start_distribution(scheduler, tmp_path):
    """The two new scratch PPO trials have an identical optimizer and observation."""
    case = {case.key: case for case in scheduler.cases("procedural")}["depth_ppo"]
    control, parent = scheduler.adaptive_overrides(case, 9, tmp_path, 0)
    assert parent == 0
    write(
        tmp_path / ".scheduler/proposals/depth_ppo_s0_trial9.json",
        {
            "initial_checkpoint": None,
            "parent_variant": parent,
            "command": [
                "python",
                "-m",
                "drone_playground.cli",
                "mode=train",
                *control,
                "seed=0",
                "output=results/control",
            ],
        },
    )
    candidate, paired_parent = scheduler.adaptive_overrides(case, 10, tmp_path, 0)
    assert paired_parent == 9
    assert candidate[0] == "experiment=navigation_procedural_depth_ppo_stratified"
    assert control[1:] == candidate[1:]


@pytest.mark.parametrize("sensor", ["depth", "lidar"])
@pytest.mark.parametrize("variant", ["validation", "stratified"])
def test_ppo_paired_presets_accept_saved_critic_override(sensor, variant):
    """A former failed launch must compose with its literal saved Hydra override."""
    config_dir = Path(__file__).resolve().parents[1] / "src/drone_playground/configs"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(
            config_name="config",
            overrides=[
                f"experiment=navigation_procedural_{sensor}_ppo_{variant}",
                "learning=ppo",
                "learning.options.critic_uses_privileged=false",
                "++learning.options.critic_uses_sensor=true",
                "learning.options.lr=0.0001",
            ],
        )
    assert cfg.scene.name == "train/procedural"
    assert cfg.learning.options.critic_uses_privileged is False
    assert cfg.learning.options.critic_uses_sensor is True
    assert cfg.results.keep_best_screening is True
    assert cfg.benchmark.fixed_checkpoint_seeds is True
    assert cfg.simulation.procedural.start_sampling == (
        "collision_free" if variant == "validation" else "stratified"
    )


def test_best_screening_score_survives_checkpoint_pruning(scheduler, tmp_path):
    """Optimizing candidate selection must not forget deleted high-score screens."""
    run = tmp_path / "depth_ppo_s0_trial3"
    write(run / "run.json", {"status": "early_stopped", "best_screen_score": 0.0677083333})
    write(
        run / "checkpoints/model_002720/screening/report.json",
        {
            "complete": True,
            "reports": {
                scene: {"task": "navigation", "episodes": 8, "successes": 0, "success_rate": 0.0}
                for scene in ("S01", "S02", "S03", "D01", "D02", "D03")
            },
        },
    )
    assert scheduler.best_evaluation_score(run) == pytest.approx(0.0677083333)


def test_candidate_four_is_independent_old_motion_ppo_ablation(scheduler):
    """Do not change already-running trials or import any old network weight."""
    candidates = {case.key: case for case in scheduler.cases("procedural")}
    for name in ("depth", "lidar"):
        case = candidates[f"{name}_ppo"]
        original = scheduler.overrides(case, 3)
        ablation = scheduler.overrides(case, 4)
        assert f"experiment=navigation_procedural_{name}_ppo_velocity" in ablation
        assert "simulation.num_envs=64" in ablation
        assert "learning.options.lr=0.0005" in ablation
        assert "initial_checkpoint=null" in ablation
        assert "resume=null" in ablation
        assert ablation != original
        assert "velocity" not in " ".join(original)


def test_selected_procedural_queue_is_smaller_and_stable(scheduler, tmp_path):
    """Queue priority and comparison contracts are versioned separately."""
    selected = ("depth_ppo", "lidar_ppo", "racing_shac", "racing_ppo")
    cases = {case.key: case for case in scheduler.cases("procedural", selected)}
    assert set(cases) == set(selected)
    assert cases["racing_shac"].experiment == "racing_shac_stable"
    assert cases["racing_shac"].learning_rate == 0.0001
    a = scheduler.report(tmp_path, {}, 2, regime="procedural", only=selected)
    assert a["train_families"] == 4
    assert a["matrix_cells"] == 6
    assert a["selected_families"] == sorted(selected)
