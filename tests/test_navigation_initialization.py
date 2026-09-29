"""Training covers free course states while nominal evaluation starts remain frozen."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def make_env(split):
    from drone_playground.composition import build_environment, compose_method

    config = compose_method(
        "learning/navigation_bptt",
        "navigation/dynamic_velocity",
        [
            "+training.navigation_initialization=free_course_v1",
        ],
    )
    return build_environment(config, "cpu", split, 2)


def test_course_initialization_is_safe_seeded_and_covers_departure_and_goal():
    from drone_playground.environments.scenes.navigation import (
        body_centre_from_state,
        clearance_and_collision,
    )

    env = make_env("train")
    try:
        keys = jax.random.split(jax.random.PRNGKey(17), 64)
        reset = jax.jit(jax.vmap(env.reset))
        first, again = reset(keys), reset(keys)
        a, b = first.pipeline_state, again.pipeline_state
        np.testing.assert_array_equal(a.sim_data.states.pos, b.sim_data.states.pos)
        pos = np.asarray(a.sim_data.states.pos[:, 0, 0])
        assert np.ptp(pos[:, 0]) > 80
        assert (pos[:, 0] == 2.0).any()
        assert (pos[:, 0] > 94.0).any()
        assert (np.asarray(a.scene_time_offset) > 1).any()
        centres = jax.vmap(body_centre_from_state)(
            a.sim_data.states.pos[:, 0, 0], a.sim_data.states.quat[:, 0, 0]
        )
        gaps = jax.vmap(
            lambda i, t, p: clearance_and_collision(env.bank, i, t, p, env.body_radius)[0]
        )(a.scenario_id, a.scene_time_offset, centres)
        assert (np.asarray(gaps) >= 0.149).all()
        assert np.isfinite(first.obs).all()
        np.testing.assert_allclose(a.sensor_time[:, -1], a.scene_time_offset, atol=1e-6)
    finally:
        env.close()


def test_nominal_development_and_heldout_start_are_preserved():
    for split in ("dev", "heldout"):
        env = make_env(split)
        try:
            state = jax.jit(jax.vmap(env.reset))(jax.random.split(jax.random.PRNGKey(19), 8))
            data = state.pipeline_state
            np.testing.assert_array_equal(
                data.sim_data.states.pos[:, 0, 0], env.bank.start[data.scenario_id]
            )
            np.testing.assert_array_equal(data.sim_data.states.vel, 0.0)
            np.testing.assert_array_equal(data.scene_time_offset, 0.0)
        finally:
            env.close()


@pytest.mark.parametrize("algorithm", ["bptt", "shac"])
def test_resampled_windows_still_update_and_resume_exactly(tmp_path, algorithm):
    from importlib import import_module

    from tests.test_bptt_training import SmoothTask

    trainer = import_module("drone_playground.learning.algorithms." + algorithm)
    train = trainer.train

    class RandomTask(SmoothTask):
        def reset(self, key):
            s = super().reset(key)
            x = jax.random.uniform(key, (1,))
            return s.replace(
                pipeline_state=x, obs=jnp.r_[x, 1.0 - x], info={**s.info, "initial_position": x}
            )

    cfg = dict(
        algorithm=algorithm,
        task="toy",
        num_envs=2,
        horizon_length=3,
        policy_updates=4,
        learning_rate=0.001,
        hidden_sizes=[8],
        num_evals=3,
        seed=0,
        use_schedule=False,
        resample_window_initials=True,
    )
    _, params, metrics = train(RandomTask(), cfg, state_directory=tmp_path)
    assert metrics["window_resets"] == 8
    assert metrics["actual_steps"] == 24 and metrics["actor_parameter_delta_l2"] > 0
    load = (
        (lambda path: trainer.load_state(path, cfg))
        if algorithm == "bptt"
        else (lambda path: trainer.load_training_state(path)[0])
    )
    middle = load(tmp_path / "update-0000002.pkl")
    end = load(tmp_path / "update-0000004.pkl")
    assert not np.array_equal(
        middle.environment.info["initial_position"], end.environment.info["initial_position"]
    )
    _, restored, _ = train(RandomTask(), cfg, restore_state=tmp_path / "update-0000002.pkl")
    for a, b in zip(jax.tree.leaves(params), jax.tree.leaves(restored), strict=True):
        np.testing.assert_array_equal(a, b)
    with pytest.raises(ValueError):
        train(
            RandomTask(),
            {**cfg, "resample_window_initials": False},
            restore_state=tmp_path / "update-0000002.pkl",
        )
