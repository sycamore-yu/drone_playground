"""Archive-verified Racing SHAC uses the same declared training defaults."""

from pathlib import Path

from hydra import compose, initialize_config_dir

import drone_playground


def test_racing_shac_stable_restores_trial11_defaults():
    """Check stable training and independent benchmark settings together."""
    directory = Path(drone_playground.__file__).parent / "configs"
    with initialize_config_dir(version_base=None, config_dir=str(directory)):
        cfg = compose(
            config_name="config", overrides=["experiment=racing_shac_stable", "learning=shac"]
        )

    assert cfg.simulation.num_envs == 256
    expected = {
        "horizon": 128,
        "lr": 0.00008125,
        "critic_lr": 0.0003,
        "critic_epochs": 2,
        "target_alpha": 0.95,
        "progress_reward_scale": 2.0,
        "failure_cost": 40.0,
        "bootstrap_weight": 0.2,
    }
    for key, value in expected.items():
        assert cfg.learning.options[key] == value
    assert cfg.learning.max_interactions == 256 * 128 * 1300
    assert cfg.learning.best.source == "evaluation"
    assert cfg.learning.early_stopping.metric == "success_rate"
    assert cfg.learning.early_stopping.target == 1.0
    assert cfg.benchmark.episodes == 100
    assert cfg.benchmark.record_replay
