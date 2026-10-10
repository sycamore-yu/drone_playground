"""Independent old-recipe velocity PPO comparison without pretrained weights."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from drone_playground.learning.navigation_reward import motion_navigation_reward
from drone_playground.learning.ppo import advantage_progress_correlation
from drone_playground.learning.trainer import Trainer
from drone_playground.simulation.environment import make_environment
from drone_playground.simulation.evaluation import create_environment, create_method
from drone_playground.simulation.methods import Action
from drone_playground.simulation.procedural_scene import ProceduralScene
from drone_playground.simulation.tasks import Event


def test_velocity_action_is_native_attitude_and_preserves_hover():
    """Zero velocity command yields hover thrust through full Crazyflow control."""
    env = make_environment(
        task="navigation",
        scene="train/procedural",
        num_envs=2,
        sensor="depth",
        device="cpu",
        procedural={"max_obstacles": 6},
        action={"level": "velocity_yaw", "heading": "fixed"},
    )
    state = env.reset(jax.random.PRNGKey(3))
    physics = state.physics.replace(
        states=state.physics.states.replace(vel=jnp.zeros_like(state.physics.states.vel))
    )
    action = Action(level="velocity_yaw")
    neutral = action.setpoint(physics, jnp.zeros((2, 4)))
    assert neutral.level == "attitude_thrust"
    assert neutral.frame == "world"
    np.testing.assert_allclose(neutral.value[:, :3], 0.0, atol=1e-5)
    np.testing.assert_allclose(neutral.value[:, 3], action.hover, rtol=1e-6)
    forward = action.setpoint(physics, jnp.array([[2.0, 0.0, 0.0, 0.0]] * 2))
    assert (np.asarray(forward.value[:, 1]) > 0).all()
    stepped = env.step(state, jnp.zeros((2, 4)))
    assert np.isfinite(np.asarray(stepped.physics.states.pos)).all()


def test_old_motion_reward_failure_and_success_scale_and_gradient():
    """Retain 0.01-scaled +100 arrival and -1000 failure with finite gradients."""
    env = make_environment(
        task="navigation",
        scene="train/procedural",
        num_envs=2,
        sensor="depth",
        device="cpu",
        procedural={"max_obstacles": 6},
        action={"level": "velocity_yaw"},
    )
    before = env.reset(jax.random.PRNGKey(2))
    neutral = jnp.zeros((2, 4))
    state = before.replace(
        task=before.task.replace(event=jnp.array([Event.SUCCESS, Event.COLLISION]))
    )
    reward, terms = motion_navigation_reward(env, before, state, neutral)
    assert float(reward[0] - reward[1]) > 10.9
    assert "motion_safety_cost" in terms
    assert np.isfinite(np.asarray(reward)).all()


def test_ppo_advantage_progress_correlation_is_informative():
    """Detect useful ordering of advantages, not simply critic loss magnitude."""
    x = jnp.array([-2.0, -1.0, 1.0, 2.0])
    y = jnp.array([-1.0, -0.5, 0.5, 1.0])
    np.testing.assert_allclose(advantage_progress_correlation(x, y), 1.0, atol=1e-6)
    np.testing.assert_allclose(advantage_progress_correlation(x, -y), -1.0, atol=1e-6)
    np.testing.assert_allclose(advantage_progress_correlation(x, jnp.zeros_like(y)), 0.0)


@pytest.mark.parametrize(
    "experiment",
    (
        "navigation_procedural_depth_ppo_velocity",
        "navigation_procedural_lidar_ppo_velocity",
    ),
)
def test_velocity_ppo_real_training_update_with_procedural_observations(experiment):
    """A trainable CNN/PointNet-GRU Actor emits four velocity/yaw commands."""
    root = Path(__file__).resolve().parents[1] / "src/drone_playground/configs"
    with initialize_config_dir(version_base=None, config_dir=str(root)):
        config = compose(
            config_name="config",
            overrides=[
                "mode=train",
                f"experiment={experiment}",
                "simulation.num_envs=2",
                "simulation.procedural.max_obstacles=6",
                "simulation.device=cpu",
                "learning.options.horizon=1",
                "learning.options.ppo_epochs=1",
                "learning.options.critic_epochs=1",
                "learning.options.minibatches=1",
            ],
        )
    env = create_environment(config)
    assert env.action.level == "velocity_yaw"
    assert env.control_level == "attitude_thrust"
    assert env.scene.distribution.start_sampling == "collision_free"
    trainer = Trainer(
        env,
        kind=config.method.actor.kind,
        algorithm="ppo",
        config=OmegaConf.to_container(config.learning.options, resolve=True),
        actor_config=OmegaConf.to_container(config.method.actor, resolve=True),
        method=create_method(config),
        loss=config.learning.loss,
    )
    state = trainer.initialize()
    assert np.all(
        np.asarray(
            env.scene_clearance(state.env_state.scene_state.start, 0.0, state.env_state.scene_state)
        )
        > 0.45
    )
    updated, metrics = trainer.update(state)
    assert int(updated.updates) == 1
    assert state.params.keys() == updated.params.keys()
    for key in (
        "approx_kl",
        "navigation_progress_metres",
        "advantage_progress_correlation",
        "critic_return_rmse_preupdate",
        "advantage_positive_fraction",
        "motion_safety_cost",
        "navigation_collision_fraction",
    ):
        assert key in metrics
        assert np.isfinite(np.asarray(metrics[key])).all(), key


def test_legacy_reward_requires_opt_in_velocity_action():
    """Never silently replace navigation reward or Action for existing runs."""
    scene = ProceduralScene({"max_obstacles": 6})
    env = make_environment(
        task="navigation",
        scene=scene.name,
        sensor="depth",
        num_envs=1,
        device="cpu",
        procedural={"max_obstacles": 6},
        action={"level": "acceleration", "heading": "target"},
    )
    with pytest.raises(ValueError, match="requires Navigation PPO"):
        Trainer(
            env,
            kind="depth",
            algorithm="ppo",
            config={"reward_profile": "legacy_motion_navigation_v2"},
        )
