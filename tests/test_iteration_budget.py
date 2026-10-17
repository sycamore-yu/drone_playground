"""Isaac Lab-style budgets count training updates, independently of rollout batch."""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir


@pytest.mark.parametrize(
    ("experiment", "algorithm", "budget"),
    [
        ("tracking_bptt_native", "bptt", 1000),
        ("racing_ppo_stable", "ppo", 4096),
        ("racing_shac_stable", "shac", 1300),
        ("oa_static_depth", "bptt", 1000),
        ("oa_dynamic_lidar", "shac", 1000),
        ("oa_static_depth_ppo", "ppo", 1000),
        ("oa_dynamic_lidar_ppo", "ppo", 1000),
    ],
)
def test_budget_is_iterations_not_environment_samples(experiment, algorithm, budget):
    """Changing the rollout batch cannot silently rescale the iteration budget."""
    root = Path(__file__).resolve().parents[1] / "src/drone_playground/configs"
    with initialize_config_dir(version_base=None, config_dir=str(root)):
        config = compose(
            config_name="config", overrides=[f"experiment={experiment}", f"learning={algorithm}"]
        )
    assert config.learning.max_iterations == budget
    assert "max_interactions" not in config.learning
