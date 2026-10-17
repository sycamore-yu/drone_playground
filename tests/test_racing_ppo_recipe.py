"""Racing PPO uses the archived reference-options training recipe."""

from pathlib import Path

from hydra import compose, initialize_config_dir

import drone_playground


def test_racing_ppo_defaults_match_reference_options():
    """The default preset must not add the older stable preset's regularizers."""
    directory = Path(drone_playground.__file__).parent / "configs"
    with initialize_config_dir(version_base=None, config_dir=str(directory)):
        cfg = compose(
            config_name="config", overrides=["experiment=racing_ppo_stable", "learning=ppo"]
        )
    expected = {
        "horizon": 32,
        "lr": 0.0003,
        "critic_lr": 0.001,
        "initial_log_std": -1.5,
        "entropy_weight": 0.0,
        "progress_reward_scale": 1.0,
        "failure_cost": 0.0,
        "ppo_target_kl": 0.0,
        "ppo_value_clip": 0.0,
    }
    for key, value in expected.items():
        assert cfg.learning.options[key] == value, key
    assert cfg.simulation.num_envs == 512
    assert cfg.learning.max_iterations == 4096
    assert cfg.task.options.reward_shaping.progress_reward_scale == 1.0
    assert cfg.initial_checkpoint is None
    assert cfg.resume is None
    assert cfg.checkpoint is None
