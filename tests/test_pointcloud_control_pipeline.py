"""Real training/reload and sub-policy-tick termination for the qualified transfer."""

import copy
import json
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from drone_playground.composition import compose_method, run_experiment
from drone_playground.models.point_mass import PointMassLag, PointMassState


def test_control_collision_stops_at_physics_tick_and_retains_terminal_pose():
    from drone_playground.evaluation.pointcloud_control import advance_checked

    task = SimpleNamespace(
        task="hovering",
        model=PointMassLag(),
        physics_dt=0.002,
        substeps=50,
        bounds_low=jnp.full(3, -10.0),
        bounds_high=jnp.full(3, 10.0),
        clearance=lambda pos: 0.019 - pos[:, 0],
    )
    state = PointMassState.create(jnp.array([[0.0, 0.0, 1.0]])).replace(
        vel=jnp.array([[1.0, 0.0, 0.0]])
    )
    zeros = jnp.zeros((1, 3))
    result = advance_checked(
        task,
        state,
        zeros,
        zeros,
        jnp.zeros(1, jnp.int32),
        jnp.zeros(1),
        jnp.zeros(1, jnp.int32),
        jnp.zeros(1, jnp.int32),
    )
    terminal, timestamp, outcome, gates, _ = result
    assert int(outcome[0]) == 2
    np.testing.assert_allclose(timestamp, [0.02], atol=1e-6)
    again = advance_checked(
        task, terminal, zeros, zeros, jnp.zeros(1, jnp.int32), timestamp, outcome, gates
    )
    np.testing.assert_array_equal(again[0].pos, terminal.pos)
    np.testing.assert_array_equal(again[1], timestamp)


@pytest.mark.parametrize("task", ["hovering", "tracking", "racing"])
def test_control_public_train_and_frozen_eval_save_nonzero_updates(tmp_path, task):
    config = compose_method(
        "paper/pointcloud_flight",
        "paper/control/" + task,
        [
            "runtime.device=cpu",
            "training.num_envs=2",
            "training.policy_updates=2",
            "training.num_evals=2",
            "training.development_episodes=2",
            "algorithm.horizon_length=2",
            "env.sensor.azimuth_count=6",
            "env.sensor.elevation_count=2",
            "env.task.duration=0.2",
        ],
    )
    trained = run_experiment(config, tmp_path, "control-train")
    assert trained["actual_steps"] == 8
    assert trained["actual_updates"] == 2
    assert trained["actor_parameter_delta_l2"] > 0
    assert trained["full_budget_completed"]
    evaluation = copy.deepcopy(config)
    evaluation.update(mode="eval", checkpoint=trained["selected"]["checkpoint"])
    evaluation["evaluation"].update(episodes=2, split="heldout")
    report = run_experiment(evaluation, tmp_path, "control-eval")
    assert report["parameters_frozen"]
    assert report["num_trials"] == len(report["episodes"]) == 2
    assert report["requested_delay_ms"][0] >= 25.0
    assert max(report["effective_delay_ms"]) <= 50.0
    from drone_playground.runs.layout import find_experiment

    index = json.loads((find_experiment(tmp_path, 'control-eval') / 'rollouts/index.json').read_text())
    assert len(index["replays"]) == 2
    for case, row in enumerate(index["replays"]):
        assert row["transitions"] == report["episodes"][case]["steps"]
        assert row["frames"] == row["transitions"] + 1
        assert row["readback_verified"]
