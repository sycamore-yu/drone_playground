"""OA published train recipes reuse historically successful learning settings."""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir


@pytest.mark.parametrize("region", ["static", "dynamic"])
@pytest.mark.parametrize("sensor", ["depth", "lidar"])
@pytest.mark.parametrize("algorithm", ["bptt", "shac"])
def test_native_oa_adopts_historic_learning_objective_and_optimizer(region, sensor, algorithm):
    """The new procedural generator is allowed; optimizer/reward remain historical."""
    root = Path(__file__).resolve().parents[1] / "src/drone_playground/configs"
    with initialize_config_dir(version_base=None, config_dir=str(root)):
        config = compose(
            config_name="config",
            overrides=[f"experiment=oa_{region}_{sensor}_{algorithm}", f"learning={algorithm}"],
        )
    opts = config.learning.options
    assert config.scene.name == "train/procedural"
    assert config.simulation.procedural.obstacle_motion == region
    assert config.simulation.procedural.layout == "matched"
    assert config.simulation.num_envs == (128 if sensor == "depth" else 64)
    assert opts.horizon == 32
    assert opts.lr == 0.0003
    assert opts.critic_lr == 0.001
    assert opts.critic_epochs == 4
    assert opts.gamma == 0.99
    assert opts.gae_lambda == 0.95
    assert opts.target_alpha == 0.995
    assert opts.temporal_gradient_alpha == pytest.approx(0.916290731874155)
    assert opts.perception_weight == 1.0
    assert opts.altitude_weight == 1.0
    assert config.learning.loss == ("zhang" if sensor == "depth" else "liu")
    if sensor == "depth":
        assert opts.perception_loss == {"clearance_margin": 1.5, "clearance_weight": 8.0}
    else:
        assert opts.perception_loss == {
            "clearance_margin": 2.0,
            "collision_weight": 8.0,
            "minimum_approach_speed": 1.0,
        }
    assert opts.failure_cost == (0.0 if algorithm == "bptt" and sensor == "depth" else 160.0)
    assert opts.initial_log_std == (-1.5 if algorithm == "bptt" and sensor == "depth" else -3.0)
    assert config.initial_checkpoint is None
    assert config.resume is None
    assert config.learning.max_iterations >= 640


@pytest.mark.parametrize("region", ["static", "dynamic"])
@pytest.mark.parametrize("sensor", ["depth", "lidar"])
def test_oa_ppo_preserves_successful_goal_reaching_recipe(region, sensor):
    """PPO keeps the measured from-scratch recipe; no historical Actor is loaded."""
    root = Path(__file__).resolve().parents[1] / "src/drone_playground/configs"
    with initialize_config_dir(version_base=None, config_dir=str(root)):
        config = compose(
            config_name="config", overrides=[f"experiment=oa_{region}_{sensor}_ppo", "learning=ppo"]
        )
    o = config.learning.options
    assert config.task.options.reward_profile == "goal_reaching"
    assert config.task.options.finite_horizon is True
    assert o.horizon == 512
    assert o.gamma == 1.0 and o.gae_lambda == 1.0
    assert o.lr == 0.0002 and o.critic_lr == 0.0005
    assert o.critic_uses_privileged is True
    assert o.entropy_weight == 0.005
    assert o.ppo_target_kl == 0.025
    assert o.ppo_value_clip == 10.0
    assert config.simulation.num_envs == (256 if sensor == "depth" else 64)
    assert config.learning.max_iterations == 1000
    assert config.initial_checkpoint is None and config.resume is None


@pytest.mark.parametrize("region", ["static", "dynamic"])
@pytest.mark.parametrize("sensor", ["depth", "lidar"])
@pytest.mark.parametrize("algorithm", ["bptt", "shac"])
def test_lotf_keeps_bodyrate_interface_without_invalid_acceleration_loss(region, sensor, algorithm):
    """Unverified LOTF transfer does not pretend the native acceleration loss applies."""
    root = Path(__file__).resolve().parents[1] / "src/drone_playground/configs"
    with initialize_config_dir(version_base=None, config_dir=str(root)):
        c = compose(
            config_name="config",
            overrides=[
                f"experiment=oa_{region}_{sensor}_{algorithm}",
                f"learning={algorithm}",
                "learning.backward_model=lotf",
                "method.action.level=body_rate",
                "learning.loss=null",
                "learning.options.perception_loss=null",
            ],
        )
    assert c.learning.backward_model == "lotf"
    assert c.method.action.level == "body_rate"
    assert c.learning.loss is None and c.learning.options.perception_loss is None
    assert c.learning.options.lr == 0.0003
