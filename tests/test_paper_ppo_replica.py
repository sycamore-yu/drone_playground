"""Paper-policy contracts and safe long-distance training population."""

from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from drone_playground.simulation.environment import BaseEnv
from drone_playground.simulation.methods import Action
from drone_playground.simulation.networks import Actor, Critic
from drone_playground.simulation.paper_fan_lidar import (
    clear_done_histories,
    empty_history,
    encode_range_ring,
    push_history,
)
from drone_playground.simulation.procedural_scene import ProceduralScene


def test_fan_range_minimum_mask_history():
    """Invalid/outside-height points do not enter nearest-range bins."""
    points = jnp.array([[[2.0, 0, 0], [4.0, 0, 0], [1.0, 0, 2.0], [jnp.nan, 0, 0]]])
    masks = jnp.array([[True, True, True, False]])
    features = encode_range_ring(points, masks)
    assert features.shape == (1, 36)
    assert np.isfinite(np.asarray(features)).all()
    np.testing.assert_allclose(np.min(np.asarray(features)), 0.2, atol=1e-6)
    assert int(np.count_nonzero(np.asarray(features) < 1)) == 1
    history = push_history(empty_history(1), features)
    assert history.shape == (1, 36, 36)
    np.testing.assert_array_equal(history[0, 0], features[0])
    np.testing.assert_array_equal(clear_done_histories(history, jnp.array([True])), 1)


@pytest.mark.parametrize("configuration", [{"fan_lidar": {}}, {"visfly": True}])
def test_paper_observation_broadcasts_static_fixed_scene_goal(configuration):
    """Fixed scenes expose one [3] goal; a paper Actor still receives [B,*] inputs."""
    batch = 8
    orientation = jnp.tile(jnp.array([0.0, 0.0, 0.0, 1.0]), (batch, 1))
    physics = SimpleNamespace(
        states=SimpleNamespace(
            pos=jnp.zeros((batch, 1, 3)),
            vel=jnp.zeros((batch, 1, 3)),
            quat=orientation[:, None, :],
            ang_vel=jnp.zeros((batch, 1, 3)),
        )
    )
    state = SimpleNamespace(
        physics=physics,
        previous_action=jnp.zeros((batch, 2)),
    )
    env = SimpleNamespace(
        sensor=None,
        observation_config=configuration,
        _observe_state=lambda _: {"state": jnp.zeros((batch, 10))},
        goal_for=lambda _: jnp.array([98.0, 0.0, 3.0]),
    )
    observation = BaseEnv._observe(env, state)
    if "fan_lidar" in configuration:
        assert observation["fan_command"].shape == (batch, 2)
        np.testing.assert_allclose(observation["fan_command"][:, 0], 98.0)
        mock_controller = SimpleNamespace(acceleration=lambda _, acceleration, yaw: acceleration)
        action = SimpleNamespace(
            level="horizontal_acceleration",
            heading="fixed",
            _acceleration=mock_controller,
        )
        setpoint = Action.setpoint(
            action,
            physics,
            jnp.zeros((batch, 2)),
            jnp.array([98.0, 0.0, 3.0]),
        )
        assert setpoint.shape == (batch, 3)
        np.testing.assert_allclose(setpoint[:, 2], 3.0)
    else:
        assert observation["visfly_target"].shape == (batch, 3)
        np.testing.assert_allclose(observation["visfly_target"][:, 0], 98.0)


@pytest.mark.parametrize(("kind", "action_size"), [("fan_lidar", 2), ("visfly_depth", 4)])
def test_paper_actor_and_critic_parameter_gradients(kind, action_size):
    """Both new networks have correctly shaped and differentiable policy outputs."""
    b = 2
    obs = {"state": jnp.zeros((b, 10))}
    if kind == "fan_lidar":
        obs.update(
            fan_map=jnp.ones((b, 36, 36, 1)),
            fan_state=jnp.zeros((b, 4)),
            fan_command=jnp.ones((b, 2)),
        )
    else:
        obs.update(
            depth=jnp.ones((b, 64, 64, 1)),
            visfly_state=jnp.zeros((b, 13)),
            visfly_target=jnp.ones((b, 3)),
        )
    actor = Actor(kind=kind, action_size=action_size)
    memory = actor.initialize_memory(b)
    assert memory.shape == (b, 0)
    params = actor.init(jax.random.key(1), obs, memory)
    raw, updated, aux = jax.jit(actor.apply)(params, obs, memory)
    assert raw.shape == (b, action_size)
    assert updated.shape == memory.shape
    assert aux.shape == (b, 3)
    gradients = jax.jit(jax.grad(lambda p: jnp.sum(actor.apply(p, obs, memory)[0])))(params)
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(gradients))
    if kind == "visfly_depth":
        critic = Critic(kind="visfly_depth")
        critic_params = critic.init(jax.random.key(2), obs)
        assert critic.apply(critic_params, obs).shape == (b,)


def test_long_distance_sampling_is_opt_in_and_safely_separated():
    """All long-course starts are at least 80m away without changing old scenes."""
    original = ProceduralScene({"start_sampling": "collision_free"})
    long = ProceduralScene({"start_sampling": "collision_free", "min_start_goal_distance_m": 80.0})
    key = jax.random.key(5)
    sample = jax.jit(lambda x: long.sample(x, 32))(key)
    assert float(jnp.min(jnp.linalg.norm(sample.goal - sample.start, axis=-1))) >= 80
    assert float(jnp.min(long.clearance(sample.start, scene_state=sample))) >= 0.45
    assert original.geometry_identity["generator"] == "procedural-navigation-v1"
    assert long.geometry_identity["generator"] == "procedural-navigation-v2-long-start"


@pytest.mark.parametrize(
    ("experiment", "kind", "sensor", "action_size"),
    [
        ("navigation_paper_fan2025_ppo", "fan_lidar", "lidar", 2),
        ("navigation_paper_visfly2025_ppo", "visfly_depth", "depth", 4),
    ],
)
def test_paper_configs_have_valid_independent_contracts(experiment, kind, sensor, action_size):
    """Never silently inherit non-NULL perception losses or prior checkpoints."""
    with initialize_config_dir(
        version_base=None,
        config_dir=str(Path(__file__).parents[1] / "src/drone_playground/configs"),
    ):
        config = compose(config_name="config", overrides=[f"experiment={experiment}"])
    assert config.method.actor.kind == kind
    assert config.sensor.name == sensor
    assert config.simulation.procedural.min_start_goal_distance_m == 80.0
    assert config.initial_checkpoint is None
    assert config.resume is None
    assert config.learning.loss is None
    assert config.learning.options.perception_loss is None
    assert (2 if config.method.action.level == "horizontal_acceleration" else 4) == action_size


@pytest.mark.parametrize(
    "experiment",
    ["navigation_paper_fan2025_ppo", "navigation_paper_visfly2025_ppo"],
)
def test_staged_screening_changes_no_training_parameters(experiment):
    """Fast two-scene probes alter evaluation scheduling, not the PPO recipe."""
    with initialize_config_dir(
        version_base=None,
        config_dir=str(Path(__file__).parents[1] / "src/drone_playground/configs"),
    ):
        baseline = compose(config_name="config", overrides=[f"experiment={experiment}"])
        staged = compose(config_name="config", overrides=[f"experiment={experiment}_staged"])
    assert staged.benchmark.fixed_checkpoint_seeds
    assert staged.screening.probe_between_full is True
    assert staged.screening.full_every == 3
    old = OmegaConf.to_container(baseline, resolve=True)
    new = OmegaConf.to_container(staged, resolve=True)
    assert old.pop("screening", None) is None
    assert new.pop("screening") == {"probe_between_full": True, "full_every": 3}
    assert old == new
