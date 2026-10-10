"""Independent old-recipe velocity PPO comparison without pretrained weights."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from drone_playground.learning.ppo import advantage_progress_correlation
from drone_playground.learning.trainer import Trainer
from drone_playground.simulation.environment import make_environment
from drone_playground.simulation.evaluation import create_environment, create_method
from drone_playground.simulation.methods import Action
from drone_playground.simulation.procedural_scene import ProceduralScene
from drone_playground.simulation.task_rewards import motion_navigation_reward
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


def test_task_reward_profile_matches_environment_cost_and_learning_objective():
    """Task/Environment owns reward math; Learning adds no hidden task reward."""
    env = make_environment(
        task="navigation",
        scene="train/procedural",
        num_envs=2,
        device="cpu",
        procedural={"max_obstacles": 6},
        action={"level": "velocity_yaw"},
        task_config={"reward_profile": "legacy_motion_navigation_v2"},
    )
    initial = env.reset(jax.random.key(15))
    action = jnp.zeros((2, 4))
    after = env.step(initial, action)
    expected_reward, expected_terms = motion_navigation_reward(env, initial, after, action)
    cost, task_terms = env.task_reward_components(initial, after, action)
    np.testing.assert_allclose(cost, -expected_reward, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(env.cost(initial, after, action), cost, rtol=1e-6)
    np.testing.assert_allclose(
        task_terms["motion_safety_cost"], expected_terms["motion_safety_cost"]
    )
    trainer = Trainer(
        env,
        kind="state",
        algorithm="ppo",
        config={"horizon": 1, "perception_weight": 0.0},
    )
    objective_cost, _, objective_terms = trainer.objective.cost(
        initial, after, action, trainer.objective.initialize_state(2)
    )
    np.testing.assert_allclose(objective_cost, cost, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        objective_terms["motion_arrival_reward"], expected_terms["motion_arrival_reward"]
    )
    assert env.task.contract["reward_profile"] == "legacy_motion_navigation_v2"


@pytest.mark.parametrize("kind", ["depth", "lidar"])
def test_task_owned_configuration_and_archived_reward_override_are_equivalent(kind):
    """New task contract and archived trial-4 Learning selector agree numerically."""
    root = Path(__file__).resolve().parents[1] / "src/drone_playground/configs"
    with initialize_config_dir(version_base=None, config_dir=str(root)):
        old = compose(
            config_name="config",
            overrides=[f"experiment=navigation_procedural_{kind}_ppo_velocity"],
        )
        new = compose(
            config_name="config",
            overrides=[f"experiment=navigation_procedural_{kind}_ppo_velocity_task"],
        )
    assert old.learning.options.reward_profile == "legacy_motion_navigation_v2"
    assert old.task.get("options", {}).get("reward_profile") is None
    assert new.learning.options.reward_profile == "default"
    assert new.task.options.reward_profile == "legacy_motion_navigation_v2"
    assert new.task.options.reward_shaping.progress_reward_scale == 0.0
    for recipe in (old, new):
        recipe.simulation.num_envs = 1
        recipe.simulation.procedural.max_obstacles = 6
        recipe.simulation.device = "cpu"
    previous, updated = create_environment(old), create_environment(new)
    state = previous.reset(jax.random.key(7))
    other = updated.reset(jax.random.key(7))
    control = jnp.zeros((1, 4))
    np.testing.assert_allclose(
        previous.cost(state, state, control), updated.cost(other, other, control), rtol=1e-6
    )
    assert previous.task.reward_profile == updated.task.reward_profile
    assert "reward_profile" not in previous.task.contract or (
        previous.task.contract["reward_profile"] == "legacy_motion_navigation_v2"
    )


def test_task_reward_shaping_overrides_old_learning_parameters():
    """Task is the source of truth when it explicitly defines reward coefficients."""
    env = make_environment(
        task="navigation",
        scene="S01",
        num_envs=1,
        device="cpu",
        task_config={"reward_shaping": {"progress_reward_scale": 0.6}},
    )
    baseline = make_environment(task="navigation", scene="S01", num_envs=1, device="cpu")
    left = Trainer(env, config={"horizon": 1, "progress_reward_scale": 0})
    right = Trainer(baseline, config={"horizon": 1, "progress_reward_scale": 0})
    first = env.reset(jax.random.key(5))
    shifted = first.replace(
        physics=first.physics.replace(
            states=first.physics.states.replace(pos=first.physics.states.pos.at[:, 0, 0].add(0.3))
        )
    )
    action = jnp.zeros((1, env.action_size))
    task_cost, _, terms = left.objective.cost(
        first, shifted, action, left.objective.initialize_state(1)
    )
    configured_cost, configured_terms = env.task_reward_components(first, shifted, action)
    np.testing.assert_allclose(env.cost(first, shifted, action), task_cost, rtol=1e-6)
    np.testing.assert_allclose(configured_cost, task_cost, rtol=1e-6)
    np.testing.assert_allclose(configured_terms["progress_reward"], terms["progress_reward"])
    base_cost, _, _ = right.objective.cost(
        first, shifted, action, right.objective.initialize_state(1)
    )
    assert float(terms["progress_reward"][0]) > 0
    np.testing.assert_allclose(base_cost - task_cost, terms["progress_reward"], rtol=1e-6)


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
    assert env.task.reward_profile == "legacy_motion_navigation_v2"
    # The original in-flight trial-4 YAML is immutable; the environment
    # constructor projects its historical Learning selector to Task.
    assert config.task.get("options", {}).get("reward_profile") is None
    assert config.learning.options.reward_profile == "legacy_motion_navigation_v2"
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
