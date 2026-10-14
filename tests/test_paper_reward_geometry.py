"""Paper reward integration on the actual fixed and procedural collision models."""

from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from drone_playground.simulation import task_rewards
from drone_playground.simulation.methods import Action
from drone_playground.simulation.procedural_scene import ProceduralScene
from drone_playground.simulation.scene import Scene
from drone_playground.simulation.tasks import Event, Task


@pytest.mark.parametrize("profile", ["fan2025", "fan2025_arrival", "visfly2025"])
@pytest.mark.parametrize("procedural", [True, False])
def test_paper_reward_uses_real_geometry_without_legacy_failure_surcharge(profile, procedural):
    """Check geometric consistency, no duplicate terminal cost and the done mask."""
    cfg_name = {
        "fan2025": "navigation_paper_fan2025_equations",
        "fan2025_arrival": "navigation_paper_fan2025_arrival",
        "visfly2025": "navigation_paper_visfly2025_source",
    }[profile]
    root = str(Path(__file__).parents[1] / "src/drone_playground/configs")
    with initialize_config_dir(version_base=None, config_dir=root):
        config = compose(config_name="config", overrides=[f"experiment={cfg_name}"])
    task = Task("navigation", None, **OmegaConf.to_container(config.task.options, resolve=True))
    scene = ProceduralScene({"max_obstacles": 8}) if procedural else Scene("S01")
    state = scene.sample(jax.random.key(4), 2) if procedural else None
    positions = state.start if procedural else jnp.array([[2.0, 0.0, 3.0], [2.0, 1.0, 3.0]])
    action = Action(
        level="horizontal_acceleration" if profile.startswith("fan2025") else "body_rate"
    )
    physics = SimpleNamespace(
        states=SimpleNamespace(
            pos=positions[:, None],
            vel=jnp.tile(jnp.array([[[1.0, 0.0, 0.0]]]), (2, 1, 1)),
            quat=jnp.tile(jnp.array([[[0.0, 0.0, 0.0, 1.0]]]), (2, 1, 1)),
            ang_vel=jnp.zeros((2, 1, 3)),
        )
    )
    before = SimpleNamespace(
        physics=physics, done=jnp.array([False, True]), previous_action=jnp.zeros((2, action.size))
    )
    after = SimpleNamespace(
        physics=physics,
        scene_state=state,
        time=jnp.array([0.1, 0.1]),
        task=SimpleNamespace(event=jnp.array([Event.COLLISION, Event.COLLISION])),
    )

    def clearance(position, time, scene_state):
        return (
            scene.clearance(position, time, scene_state=scene_state)
            if procedural
            else (scene.clearance(position, time))
        )

    env = SimpleNamespace(
        task=task,
        scene=scene,
        dt=0.04,
        action=action,
        scene_clearance=clearance,
        goal_for=lambda _: state.goal if procedural else task.goal,
    )
    _, _, distances, _, active = task_rewards.obstacle_reward_geometry(env, after)
    expected_distance = clearance(positions, after.time, state)
    np.testing.assert_allclose(
        jnp.min(jnp.where(active, distances, jnp.inf), -1), expected_distance, rtol=1e-5
    )
    actions = jnp.zeros((2, action.size))
    reward, direct_terms = task_rewards.paper_navigation_reward(env, before, after, actions)
    cost, terms = task_rewards.shaped_task_cost(
        env, before, after, actions, {"failure_cost": 160.0, "altitude_weight": 1.0}
    )
    np.testing.assert_allclose(cost, -reward, rtol=1e-6)
    assert np.isfinite(cost).all()
    assert float(cost[1]) == 0.0
    assert set(terms) == set(direct_terms)
    if profile == "fan2025_arrival":
        after.task.event = jnp.array([Event.SUCCESS, Event.SUCCESS])
        physics.states.vel = jnp.full((2, 1, 3), 1_000.0)
        reward, terms = task_rewards.paper_navigation_reward(env, before, after, actions)
        assert np.isfinite(reward).all()
        np.testing.assert_array_equal(terms["paper_velocity_reward"], 0.0)
        np.testing.assert_array_equal(terms["paper_hover_reward"], 0.0)
        np.testing.assert_allclose(terms["paper_alive_reward"], [-0.1, 0.0])
        np.testing.assert_array_equal(terms["paper_arrival_reward"], [5.0, 0.0])
