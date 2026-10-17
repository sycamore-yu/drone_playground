"""Release training covers full-flight distances and confirms candidate success."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from drone_playground.learning.selection import EarlyStopping
from drone_playground.simulation.procedural_scene import ProceduralScene


@pytest.mark.parametrize("algorithm", ["ppo", "bptt", "shac"])
@pytest.mark.parametrize("sensor", ["depth", "lidar"])
def test_matched_generator_executes_real_policy_update(algorithm, sensor):
    """All three learners use the matched state and real sensor observations."""
    from pathlib import Path

    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    from drone_playground.learning.trainer import Trainer
    from drone_playground.simulation.factory import create_environment

    directory = Path(__file__).resolve().parents[1] / "src/drone_playground/configs"
    suffix = "_ppo" if algorithm == "ppo" else ""
    with initialize_config_dir(version_base=None, config_dir=str(directory)):
        config = compose(
            config_name="config",
            overrides=[
                f"experiment=oa_static_{sensor}{suffix}",
                f"learning={algorithm}",
                "simulation.device=cpu",
                "simulation.num_envs=2",
                "simulation.procedural.max_obstacles=8",
                "learning.options.horizon=2",
                "learning.options.minibatches=1",
                "learning.options.ppo_epochs=1",
                "learning.options.critic_epochs=1",
            ],
        )
    if sensor == "depth":
        config.sensor.config.width, config.sensor.config.height = 16, 12
        config.observation.depth = {"pool": 1}
    else:
        config.observation.lidar.point_count = 32
    env = create_environment(config)
    trainer = Trainer(
        env,
        kind=sensor,
        algorithm=algorithm,
        config=OmegaConf.to_container(config.learning.options, resolve=True),
        actor_config=OmegaConf.to_container(config.method.actor, resolve=True),
        loss=config.learning.loss,
        backward_model=config.learning.backward_model,
        backward_options=OmegaConf.to_container(config.learning.backward_options, resolve=True),
    )
    state, metrics = trainer.update(trainer.initialize(), sampling_progress=0.0)
    jax.block_until_ready((state, metrics))
    assert np.isfinite(float(metrics["loss"]))
    assert state.updates == 1
    assert "reset_distance_long_count" in metrics
    env.sim.close()


def test_confirmation_is_lazy_and_small_checks_cannot_replace_it(monkeypatch):
    """A larger check is invoked only after all quick cases pass."""
    from types import SimpleNamespace

    from omegaconf import OmegaConf

    from drone_playground.learning.selection import RewardSelection

    calls = []
    quick_success = iter([False, True])

    def factory(config, trainer):
        count = config.learning.best.episodes

        def evaluate(params):
            calls.append(count)
            passed = next(quick_success) if count == 24 else True
            return [{"return": 1.0, "success": passed}] * count

        return evaluate

    monkeypatch.setattr("drone_playground.learning.selection.fixed_evaluator", factory)
    config = OmegaConf.create(
        {
            "learning": {
                "best": {
                    "source": "evaluation",
                    "episodes": 24,
                    "confirmation_episodes": 96,
                    "interval": 1,
                    "seed": 1000000,
                    "metric": "success_rate",
                }
            }
        }
    )
    selector = RewardSelection(config, SimpleNamespace())
    first = selector.measure(SimpleNamespace(interactions=1, params=None))
    assert calls == [24] and first["confirmed"] is False
    second = selector.measure(SimpleNamespace(interactions=2, params=None))
    assert calls == [24, 24, 96] and second["confirmed"] is True
    assert second["episodes"] == 96


def test_matched_check_batch_balances_distance_and_geometry():
    """The 24-case check includes eight genuine long flights, not near-goal resets."""
    scene = ProceduralScene({"layout": "matched", "start_sampling": "distance_stratified"})
    state = jax.jit(lambda: scene.sample(jax.random.key(17), 24))()
    distance = np.linalg.norm(np.asarray(state.goal - state.start), axis=-1)
    assert np.sum((distance >= 10 - 1e-3) & (distance <= 30 + 1e-3)) == 8
    assert np.sum((distance >= 30) & (distance < 70 + 1e-3)) == 8
    assert np.sum(distance >= 80 - 1e-3) == 8
    assert np.max(distance) > 95
    assert np.all(np.asarray(scene.clearance(state.start, scene_state=state)) > 0.45)
    assert np.all(np.asarray(scene.clearance(state.goal, scene_state=state)) > 0.45)
    np.testing.assert_allclose(
        state.centers, scene.sample(jax.random.key(17), 24).centers, atol=1e-5
    )


def test_matched_curriculum_keeps_long_flights_and_shifts_toward_them():
    """Both early and late batches contain the full intended task distance."""
    scene = ProceduralScene({"layout": "matched", "start_sampling": "distance_curriculum"})
    early = scene.sample(jax.random.key(3), 120, sampling_progress=0.0)
    late = scene.sample(jax.random.key(3), 120, sampling_progress=1.0)
    first = np.linalg.norm(np.asarray(early.goal - early.start), axis=-1)
    last = np.linalg.norm(np.asarray(late.goal - late.start), axis=-1)
    assert np.mean(first >= 80) > 0.15
    assert np.mean(last >= 80) > 0.55
    assert last.mean() > first.mean()


def test_matched_static_routes_are_clear_and_not_one_outside_lane():
    """Each layout has its own bent traversable route, including narrow passages."""
    scene = ProceduralScene(
        {"layout": "matched", "start_sampling": "distance_stratified", "obstacle_motion": "static"}
    )
    state = scene.sample(jax.random.key(12), 24)
    route = state.route
    segments = route[:, :-1, None] + jnp.linspace(0, 1, 40)[None, None, :, None] * (
        route[:, 1:, None] - route[:, :-1, None]
    )
    points = segments.reshape(24, -1, 3)
    assert np.min(np.asarray(scene.clearance(points, scene_state=state))) > 0.25
    assert not np.allclose(route[0], route[1])
    assert np.max(np.abs(np.asarray(route[:, 1:-1, 1]))) > 5
    assert np.any(np.abs(np.asarray(state.centers[..., 1])) > 16)
    lane = jnp.stack([jnp.linspace(5, 95, 181), jnp.full(181, -17.5), jnp.full(181, 3)], -1)
    clearance = scene.clearance(jnp.broadcast_to(lane, (24, 181, 3)), scene_state=state)
    assert np.any(np.min(np.asarray(clearance), axis=1) < 0)


def test_matched_dynamic_motion_covers_both_benchmark_motion_families():
    """Moving obstacles vary phase and motion family and share sensor geometry."""
    scene = ProceduralScene(
        {"layout": "matched", "start_sampling": "distance_stratified", "obstacle_motion": "dynamic"}
    )
    state = scene.sample(jax.random.key(2), 24)
    kinds = np.asarray(state.motion[..., 0])[np.asarray(state.active)]
    assert 1 in kinds and 2 in kinds
    at_zero = scene._centers_at(state, jnp.zeros(24))
    at_one = scene._centers_at(state, jnp.ones(24))
    assert not np.allclose(at_zero, at_one)
    assert np.all(np.asarray(scene.clearance(state.start, scene_state=state)) > 0.45)
    assert scene.geometry_hash != ProceduralScene().geometry_hash


def test_quick_success_cannot_stop_before_confirmation():
    """Repeated small checks never replace the configured larger confirmation."""
    stopping = EarlyStopping(
        {"metric": "success_rate", "target": 1.0, "patience": 0, "min_delta": 0.0}, "evaluation"
    )
    quick = {"mean_reward": 1.0, "episode_records": [{"success": True}] * 24, "confirmed": False}
    stopping.observe(quick)
    assert stopping.reason() is None
    stopping.observe({**quick, "episode_records": [{"success": True}] * 96, "confirmed": True})
    assert stopping.reason()["reason"] == "target_reached"


def test_matched_preview_motion_equals_training_motion():
    """Preview records move obstacles with the same law as collision and sensing."""
    import mujoco

    from drone_playground.simulation.replay import _motion
    from drone_playground.simulation.training_visuals import _visual_model

    scene = ProceduralScene({"layout": "matched", "max_obstacles": 8, "obstacle_motion": "dynamic"})
    state = scene.sample(jax.random.key(8), 1)
    sample = {
        name: np.asarray(getattr(state, name)[0])
        for name in (
            "centers",
            "sizes",
            "target",
            "speed",
            "phase",
            "active",
            "cylinder",
            "moving",
            "motion",
        )
    }
    sample["scene_name"] = np.asarray("train/procedural")
    model, indices, _, _ = _visual_model(sample)
    data = mujoco.MjData(model)
    mocap = np.broadcast_to(data.mocap_pos, (2, 1, *data.mocap_pos.shape)).copy()
    _motion(model, np.array([0.0, 3.0]), mocap)
    for t_index, t in enumerate((0.0, 3.0)):
        expected = np.asarray(scene._centers_at(state, jnp.array([t])))[0]
        for index in indices:
            body = model.body(f"obstacle_{index}")
            np.testing.assert_allclose(
                mocap[t_index, 0, int(body.mocapid[0])], expected[index], atol=1e-5
            )
