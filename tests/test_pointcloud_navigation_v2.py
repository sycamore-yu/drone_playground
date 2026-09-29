"""Explicit new navigation protocol; historical 40s paper transfer remains reproducible."""

import copy

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from drone_playground.composition import compose_method, validate_config
from tests.test_pointcloud_evaluation import task_and_bank


def test_v2_pointcloud_preset_resolves_300_seconds_without_relabelling_v1():
    old = compose_method("paper/pointcloud_flight", "paper/pointcloud_navigation", ["mode=eval"])
    assert old["env"]["task"]["duration"] == old["evaluation"]["duration"] == 40.0
    current = compose_method(
        "paper/pointcloud_flight", "paper/pointcloud_navigation_v2", ["mode=eval"]
    )
    assert current["env"]["task"]["duration"] == current["evaluation"]["duration"] == 300.0
    assert current["evaluation"]["speeds"] == [4.0, 6.0, 8.0, 20.0]
    validate_config(current)
    changed = copy.deepcopy(current)
    changed["evaluation"]["duration"] = 301.0
    with pytest.raises(ValueError):
        validate_config(changed)
    changed = copy.deepcopy(current)
    changed["evaluation"]["speeds"] = [21.0]
    with pytest.raises(ValueError):
        validate_config(changed)


def test_v2_arrival_stops_within_policy_tick_and_cannot_leave_goal_again():
    from drone_playground.evaluation.pointcloud import advance_checked

    _, task, bank = task_and_bank()
    task.arrival_sampling = "physics"
    bank = bank.replace(active=jnp.zeros_like(bank.active), goal=jnp.array([[1.1, 0.0, 3.0]]))
    state = task.initial_state(bank).replace(vel=jnp.array([[40.0, 0.0, 0.0]]))
    args = (task, bank, state, jnp.zeros((1, 3)), jnp.zeros(1), jnp.zeros(1, jnp.int32))
    terminal, timestamp, outcome, _ = advance_checked(*args)
    assert int(outcome[0]) == 1
    assert 0 < float(timestamp[0]) < 0.02
    assert float(jnp.linalg.norm(terminal.pos - bank.goal)) <= task.goal_radius
    again = advance_checked(task, bank, terminal, jnp.ones((1, 3)), timestamp, outcome)
    np.testing.assert_array_equal(again[0].pos, terminal.pos)
    np.testing.assert_array_equal(again[1], timestamp)


def test_finished_pointcloud_batch_skips_sensor_and_policy_execution():
    from drone_playground.evaluation.pointcloud import make_rollout

    _, task, bank = task_and_bank()
    bank = bank.replace(active=jnp.zeros_like(bank.active), goal=bank.start)
    calls = []

    class CountedPolicy:
        hidden_size = 1

        def apply(self, params, points, valid, proprio, memory):
            del params, points, valid, proprio
            jax.debug.callback(lambda x: calls.append(int(x)), jnp.int32(1), ordered=True)
            return jnp.zeros((1, 3)), memory

    trace = make_rollout(task, CountedPolicy(), bank)({}, jnp.array([4.0]))
    jax.block_until_ready(trace)
    jax.effects_barrier()
    assert np.asarray(trace["active"]).sum() == 1
    assert len(calls) == 1
    np.testing.assert_array_equal(
        trace["pos"][1:], jnp.broadcast_to(trace["pos"][0], trace["pos"][1:].shape)
    )
