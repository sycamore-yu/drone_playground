"""Comparable initial flight cases for vectorized and serial benchmark methods."""

import jax
import numpy as np
import pytest

from drone_playground.simulation.controllers import MellingerController
from drone_playground.simulation.environment import make_environment
from drone_playground.simulation.methods import PlannerController
from drone_playground.simulation.runner import rollout, rollout_host, slice_initial_world


@pytest.mark.parametrize("task,scene", [("tracking", "empty"), ("navigation", "S01")])
def test_initial_world_is_identical_in_batched_and_serial_execution(task, scene):
    """Every physical parameter, sensor buffer, and RNG follows the same case."""
    many = make_environment(task=task, scene=scene, num_envs=3, sensor="state", device="cpu")
    one = make_environment(task=task, scene=scene, num_envs=1, sensor="state", device="cpu")
    initial = many.reset(jax.random.PRNGKey(2026))
    template = one.reset(jax.random.PRNGKey(99))
    for index in range(3):
        selected = slice_initial_world(initial, template, index, 3)
        assert jax.tree.structure(selected) == jax.tree.structure(template)
        for actual, expected in zip(
            jax.tree.leaves(selected), jax.tree.leaves(template), strict=True
        ):
            assert np.shape(actual) == np.shape(expected)
        np.testing.assert_array_equal(
            selected.physics.states.pos[0, 0], initial.physics.states.pos[index, 0]
        )
        np.testing.assert_array_equal(
            selected.physics.states.vel[0, 0], initial.physics.states.vel[index, 0]
        )
    with pytest.raises(ValueError, match="outside"):
        slice_initial_world(initial, template, 3, 3)


def test_host_and_vectorized_rollouts_start_from_the_same_case():
    """Serial host execution must not resample a policy's initial conditions."""

    class HostMellinger(MellingerController):
        execution = "host"

    many = make_environment(task="tracking", scene="empty", num_envs=2, duration=0.04)
    one = make_environment(task="tracking", scene="empty", num_envs=1, duration=0.04)
    initial = many.reset(jax.random.PRNGKey(123))
    template = one.reset(jax.random.PRNGKey(123))
    vectorized = PlannerController(MellingerController())
    serial = PlannerController(HostMellinger())
    _, batch_trace = rollout(many, seed=123, method=vectorized, initial_state=initial)
    for index in range(2):
        single_initial = slice_initial_world(initial, template, index, 2)
        records, single_trace, _, _ = rollout_host(
            one, serial, seed=123 + 1000 * index, initial_state=single_initial
        )
        np.testing.assert_allclose(
            single_trace["position"][0, 0], batch_trace["position"][0, index]
        )
        np.testing.assert_allclose(
            single_trace["velocity"][0, 0], batch_trace["velocity"][0, index]
        )
        assert len(records) == 1
