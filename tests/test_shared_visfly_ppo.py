"""Joint actor/value gradients for VisFly's official shared-feature design."""

from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import optax
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from drone_playground.learning import ppo
from drone_playground.learning.trainer import _DEFAULTS, Trainer, TrainingState
from drone_playground.simulation.evaluation import create_environment, create_method
from drone_playground.simulation.networks import Critic


def _state():
    actor = {
        "params": {
            "visfly_policy": {
                "VisFlyFeatures_0": {"kernel": jnp.array([2.0])},
                "Dense_0": {"kernel": jnp.array([3.0])},
            }
        }
    }
    critic = {
        "params": {
            "visfly_value": {
                "VisFlyFeatures_0": {"kernel": jnp.array([9.0])},
                "Dense_0": {"kernel": jnp.array([4.0])},
            }
        }
    }
    return TrainingState(
        params=actor,
        critic_params=critic,
        target_critic_params=critic,
        optimizer_state={},
        rng=jax.random.key(0),
        env_state=None,
        recurrent_memory=jnp.zeros((1, 0)),
        objective_state=None,
        updates=0,
        interactions=0,
        log_std=jnp.zeros(1),
    )


def test_shared_features_have_one_optimizer_copy_and_receive_both_gradients():
    """Value loss reaches the actor-owned encoder without a duplicate trainable copy."""
    assert "shared_visual_encoder" in _DEFAULTS, "Shared VisFly optimizer setting missing"
    state = _state()
    variables = ppo.policy_variables(state, shared_visual_encoder=True)
    assert "VisFlyFeatures_0" not in variables["critic_head"]["params"]["visfly_value"]

    def loss(v):
        shared = v["params"]["params"]["visfly_policy"]["VisFlyFeatures_0"]["kernel"]
        actor_head = v["params"]["params"]["visfly_policy"]["Dense_0"]["kernel"]
        c = ppo.shared_critic_parameters(v["params"], v["critic_head"])
        value_encoder = c["params"]["visfly_value"]["VisFlyFeatures_0"]["kernel"]
        critic_head = c["params"]["visfly_value"]["Dense_0"]["kernel"]
        return jnp.sum(shared * actor_head + 0.5 * value_encoder * critic_head)

    gradient = jax.grad(loss)(variables)
    encoder = gradient["params"]["params"]["visfly_policy"]["VisFlyFeatures_0"]["kernel"]
    np.testing.assert_allclose(encoder, [5.0])
    np.testing.assert_allclose(
        gradient["critic_head"]["params"]["visfly_value"]["Dense_0"]["kernel"], [1.0]
    )
    # Original records must stay unchanged when constructing the tied computation.
    np.testing.assert_allclose(
        state.critic_params["params"]["visfly_value"]["VisFlyFeatures_0"]["kernel"], [9.0]
    )


def test_joint_optimizer_updates_value_head_and_keeps_encoder_synchronized():
    """One real Optax step updates both heads, with shared features identical after update."""
    assert "shared_visual_encoder" in _DEFAULTS
    state = _state()
    variables = ppo.policy_variables(state, shared_visual_encoder=True)
    optimizer = optax.sgd(0.1)
    state = state.replace(optimizer_state={"actor": optimizer.init(variables), "critic": None})
    trainer = SimpleNamespace(
        algorithm="ppo", config={"shared_visual_encoder": True}, actor_optimizer=optimizer
    )
    gradient = jax.tree.map(jnp.ones_like, variables)
    updated = Trainer._apply_actor(trainer, state, gradient)
    actor_encoder = updated.params["params"]["visfly_policy"]["VisFlyFeatures_0"]
    critic_encoder = updated.critic_params["params"]["visfly_value"]["VisFlyFeatures_0"]
    np.testing.assert_allclose(actor_encoder["kernel"], [1.9])
    np.testing.assert_array_equal(actor_encoder["kernel"], critic_encoder["kernel"])
    np.testing.assert_allclose(
        updated.critic_params["params"]["visfly_value"]["Dense_0"]["kernel"], [3.9]
    )


def test_nonshared_ppo_keeps_original_optimizer_tree():
    """All previous actor/critic checkpoint structures retain their normal PPO variables."""
    assert hasattr(ppo, "policy_variables")
    state = _state()
    variables = ppo.policy_variables(state, shared_visual_encoder=False)
    assert set(variables) == {"params", "log_std"}
    assert variables["params"] is state.params


def test_real_shared_visfly_ppo_update_ties_visual_features_after_learning():
    """A complete CPU Crazyflow/PPO update shares the actual CNN and both heads."""
    root = str(Path(__file__).parents[1] / "src/drone_playground/configs")
    with initialize_config_dir(version_base=None, config_dir=root):
        config = compose(
            config_name="config",
            overrides=[
                "experiment=navigation_paper_visfly2025_shared",
                "simulation.device=cpu",
                "simulation.num_envs=2",
                "simulation.procedural.max_obstacles=8",
                "learning.options.horizon=1",
                "learning.options.minibatches=1",
                "learning.options.ppo_epochs=1",
            ],
        )
    env = create_environment(config)
    trainer = Trainer(
        env,
        kind="visfly_depth",
        algorithm="ppo",
        seed=12,
        config=OmegaConf.to_container(config.learning.options, resolve=True),
        actor_config=OmegaConf.to_container(config.method.actor, resolve=True),
        critic=Critic(kind="visfly_depth", initialization="pytorch"),
        method=create_method(config),
        loss=None,
    )
    try:
        initial = trainer.initialize()
        updated, metrics = trainer.update(initial)
        assert int(updated.updates) == 1
        assert updated.optimizer_state["critic"] is None
        assert float(metrics["critic_loss"]) >= 0
        assert all(np.isfinite(np.asarray(value)).all() for value in metrics.values())
        actor_encoder = updated.params["params"]["visfly_policy"]["VisFlyFeatures_0"]
        value_encoder = updated.critic_params["params"]["visfly_value"]["VisFlyFeatures_0"]
        for actor, value in zip(
            jax.tree.leaves(actor_encoder), jax.tree.leaves(value_encoder), strict=True
        ):
            np.testing.assert_array_equal(actor, value)
        difference = max(
            float(jnp.max(jnp.abs(a - b)))
            for a, b in zip(
                jax.tree.leaves(initial.params), jax.tree.leaves(updated.params), strict=True
            )
        )
        assert difference > 0
    finally:
        env.sim.close()
