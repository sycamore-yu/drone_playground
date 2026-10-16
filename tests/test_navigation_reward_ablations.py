"""Versioned single-variable Navigation reward experiments and gradient contracts."""

from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from drone_playground.simulation.task_rewards import shaped_task_cost
from drone_playground.simulation.tasks import Event, Task


class _Env:
    def __init__(self, *, empty=False):
        self.dt = 0.1
        self.task = SimpleNamespace(
            name="navigation",
            reward_profile="default",
            low=jnp.array([0.0, -20.0, 0.5]),
            high=jnp.array([100.0, 20.0, 6.0]),
            radius=0.0,
        )
        self.empty = empty

    def _base_task_reward_components(self, before, after, action):
        return jnp.zeros(action.shape[0]), {}

    def scene_clearance(self, position, time, scene_state):
        del time, scene_state
        if self.empty:
            return jnp.full(position.shape[0], jnp.inf)
        return 10.0 - position[:, 0]


def _state(positions, events):
    event = jnp.asarray(events, dtype=jnp.int32)
    return SimpleNamespace(
        done=event != Event.RUNNING,
        task=SimpleNamespace(event=event),
        physics=SimpleNamespace(states=SimpleNamespace(pos=jnp.asarray(positions)[:, None, :])),
        time=jnp.zeros(len(event)),
        scene_state=None,
    )


def test_failure_event_costs_replace_generic_160_without_changing_success():
    """Only the episode's actual first terminal event selects the surcharge."""
    env = _Env()
    before = _state([[5, 0, 3]] * 4, [Event.RUNNING] * 4)
    after = _state(
        [[5, 0, 3]] * 4,
        [Event.COLLISION, Event.OUT_OF_BOUNDS, Event.TIMEOUT, Event.SUCCESS],
    )
    action = jnp.zeros((4, 3))
    profile = {
        "failure_cost": 160.0,
        "failure_cost_by_event": {
            "collision": 160.0,
            "out_of_bounds": 240.0,
            "timeout": 40.0,
            "numerical_failure": 160.0,
        },
    }
    cost, terms = jax.jit(lambda: shaped_task_cost(env, before, after, action, profile))()
    np.testing.assert_allclose(cost, [160, 240, 40, 0])
    np.testing.assert_allclose(terms["failure_cost"], [160, 240, 40, 0])
    baseline, _ = shaped_task_cost(env, before, after, action, {"failure_cost": 160.0})
    np.testing.assert_allclose(baseline, [160, 160, 160, 0])


def test_boundary_shaping_covers_x_y_z_with_finite_actionable_gradient():
    """Penalize all Cartesian workspace faces and clear gradients after done."""
    env = _Env()
    before = _state([[50, 0, 3]] * 4, [Event.RUNNING] * 4)
    after = _state(
        [[0.2, 0, 3], [50, 19.7, 3], [50, 0, 5.8], [50, 0, 3]],
        [Event.RUNNING] * 4,
    )
    action = jnp.zeros((4, 3))
    shaping = {"boundary_proximity": {"weight": 2.0, "margin": 1.5}}
    _, terms = shaped_task_cost(env, before, after, action, shaping)
    np.testing.assert_allclose(
        terms["boundary_proximity_cost"],
        0.2 * np.array([1.3**2, 1.2**2, 1.3**2, 0]),
        atol=1e-5,
    )

    def cost(x):
        changed = _state([[x, 0, 3]], [Event.RUNNING])
        previous = _state([[50, 0, 3]], [Event.RUNNING])
        return shaped_task_cost(env, previous, changed, jnp.zeros((1, 3)), shaping)[0].sum()

    np.testing.assert_allclose(jax.grad(cost)(0.2), -0.52, atol=1e-5)
    finished = _state([[50, 0, 3]] * 4, [Event.COLLISION] * 4)
    _, terms = shaped_task_cost(env, finished, after, action, shaping)
    np.testing.assert_array_equal(terms["boundary_proximity_cost"], 0)


def test_approach_risk_penalizes_rapid_proximity_before_collision():
    """Detect pre-contact closing speed while masking empty geometry."""
    env = _Env()
    before = _state([[8, 0, 3]], [Event.RUNNING])
    after = _state([[9, 0, 3]], [Event.RUNNING])
    shaping = {"approach_risk": {"weight": 0.5, "margin": 2.0, "speed_reference": 2.0}}
    cost, terms = shaped_task_cost(env, before, after, jnp.zeros((1, 3)), shaping)
    np.testing.assert_allclose(cost, [0.3], atol=1e-6)
    np.testing.assert_allclose(terms["approach_risk_cost"], [0.3], atol=1e-6)

    def toward_obstacle(x):
        changed = _state([[x, 0, 3]], [Event.RUNNING])
        return shaped_task_cost(env, before, changed, jnp.zeros((1, 3)), shaping)[0].sum()

    assert float(jax.grad(toward_obstacle)(9.0)) > 0.0
    empty = _Env(empty=True)
    no_hit, _ = shaped_task_cost(empty, before, after, jnp.zeros((1, 3)), shaping)
    np.testing.assert_array_equal(no_hit, 0)


@pytest.mark.parametrize(
    "name,shaping",
    [
        ("failure_cost_by_event", {"collision": 1.0}),
        ("boundary_proximity", {"weight": 2.0, "margin": 0.0}),
        ("approach_risk", {"weight": 2.0, "margin": 1.0, "speed_reference": -1.0}),
    ],
)
def test_task_rejects_incomplete_or_nonphysical_safety_profile(name, shaping):
    """Reject misleading or physically impossible task-only reward settings."""
    with pytest.raises(ValueError, match="reward-shaping"):
        Task("navigation", None, reward_shaping={name: shaping})


@pytest.mark.parametrize("task", ["tracking", "navigation"])
def test_reward_profile_validation_does_not_change_task_name(task):
    """Inspecting nested safety settings must never overwrite the task name."""
    created = Task(task, None)
    assert created.name == task
    assert created.duration == (20.0 if task == "tracking" else 300.0)


@pytest.mark.parametrize(
    "sensor,letter",
    [
        ("depth", "a"),
        ("depth", "b"),
        ("depth", "d"),
        ("lidar", "a"),
        ("lidar", "c"),
        ("lidar", "d"),
    ],
)
def test_ablation_changes_one_logical_setting_and_retains_fixed_validation(sensor, letter):
    """Every arm isolates one reward contract or the PPO discount factor."""
    root = str((Path(__file__).parents[1] / "src/drone_playground/configs").resolve())
    with initialize_config_dir(version_base=None, config_dir=root):
        control = compose(
            config_name="config",
            overrides=[
                f"experiment=navigation_procedural_{sensor}_ppo_ablation_base",
                "++learning.options.progress_reward_scale=0.5",
                "++learning.options.height_boundary_weight=0.5",
            ],
        )
        candidate = compose(
            config_name="config",
            overrides=[
                f"experiment=navigation_procedural_{sensor}_ppo_ablation_{letter}",
                "++learning.options.progress_reward_scale=0.5",
                "++learning.options.height_boundary_weight=0.5",
            ],
        )
    before = OmegaConf.to_container(control, resolve=True)
    after = OmegaConf.to_container(candidate, resolve=True)
    assert before["scene"]["name"] == after["scene"]["name"] == "train/procedural"
    assert before["simulation"] == after["simulation"]
    assert before["task"]["name"] == after["task"]["name"] == "navigation"
    if letter == "d":
        assert after["learning"]["options"]["gamma"] == 0.995
        after["learning"]["options"]["gamma"] = before["learning"]["options"]["gamma"]
    else:
        before_keys = set(before["task"]["options"]["reward_shaping"])
        after_keys = set(after["task"]["options"]["reward_shaping"])
        assert after_keys - before_keys == {
            {"a": "failure_cost_by_event", "b": "boundary_proximity", "c": "approach_risk"}[letter]
        }
        for key in after_keys - before_keys:
            after["task"]["options"]["reward_shaping"].pop(key)
    assert before == after


@pytest.mark.parametrize("sensor,letter", [("depth", "b"), ("lidar", "c")])
def test_real_crazyflow_step_uses_task_owned_reward_and_learning_loss(sensor, letter):
    """Task and objective agree on new safety terms under actual Crazyflow physics."""
    from drone_playground.learning.trainer import Trainer
    from drone_playground.simulation.factory import create_environment

    config_dir = str((Path(__file__).parents[1] / "src/drone_playground/configs").resolve())
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        config = compose(
            config_name="config",
            overrides=[
                f"experiment=navigation_procedural_{sensor}_ppo_ablation_{letter}",
                "simulation.device=cpu",
                "simulation.num_envs=1",
                "simulation.procedural.max_obstacles=6",
                "learning.options.minibatches=1",
                "learning.options.horizon=1",
                "learning.options.ppo_epochs=1",
                "learning.options.critic_epochs=1",
                "++learning.options.progress_reward_scale=0.5",
                "++learning.options.height_boundary_weight=0.5",
            ],
        )
    environment = create_environment(config)
    try:
        before = environment.reset(jax.random.key(5))
        action = jnp.zeros((1, environment.action_size))
        after = environment.step(before, action)
        cost, task_terms = environment.task_reward_components(before, after, action)
        trainer = Trainer(
            environment,
            kind=sensor,
            algorithm="ppo",
            loss=config.learning.loss,
            config=OmegaConf.to_container(config.learning.options, resolve=True),
        )
        total, _, learning_terms = trainer.objective.cost(
            before, after, action, trainer.objective.initialize_state(1)
        )
        np.testing.assert_allclose(
            total,
            cost + trainer.config["perception_weight"] * learning_terms["perception_cost"],
            atol=1e-5,
        )
        term_name = "boundary_proximity_cost" if letter == "b" else "approach_risk_cost"
        np.testing.assert_allclose(task_terms[term_name], learning_terms[term_name])
        assert np.isfinite(np.asarray(total)).all()
        _, metrics = trainer.update(trainer.initialize())
        assert term_name in metrics
        assert np.isfinite(np.asarray(metrics[term_name])).all()
        assert np.isfinite(np.asarray(metrics["actor_loss"])).all()
    finally:
        environment.sim.close()
