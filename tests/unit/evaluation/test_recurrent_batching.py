"""Changing evaluation batch size must not change cases, seeds or outcomes."""

import jax
import jax.numpy as jnp
import numpy as np

from drone_playground.configuration import compose_experiment
from drone_playground.environments.factory import build_environment
from drone_playground.evaluation.navigation.recurrent import RecurrentNavigationEvaluator
from drone_playground.networks.factory import build_network


def test_recurrent_batches_preserve_every_case_seed_and_transition():
    config = compose_experiment(
        "navigation/differentiable_pointcloud",
        overrides=[
            "runtime.device=cpu",
            "evaluation.protocol=null",
            "env.task.duration=0.2",
            "env.sensor.azimuth_count=8",
            "env.sensor.elevation_count=3",
            "network.hidden_size=8",
            "network.encoder.point_channels=[8,8]",
        ],
    )
    env = build_environment(config, "cpu", role="eval")
    try:
        task = env.task
        net = build_network(config["network"])
        bank = task.select_bank(jnp.array([0]))
        physical = task.initial_state(bank)
        points, valid, obs, _ = task.measure(bank, physical, jnp.zeros(1), jnp.full((1,), 4.0))
        params = net.init(jax.random.key(8), points, valid, obs, jnp.zeros((1, net.hidden_size)))
        evaluator = RecurrentNavigationEvaluator(task, net, seed_start=80000)
        full_report, full = evaluator.run(params)
        batched_report, batched = evaluator.run(params, batch_size=3)
        assert full_report["reset_seeds"] == batched_report["reset_seeds"]
        assert full_report["scene_success_rates"] == batched_report["scene_success_rates"]
        assert full_report["delay_ticks"] == batched_report["delay_ticks"]
        for expected, actual in zip(jax.tree.leaves(full), jax.tree.leaves(batched), strict=True):
            np.testing.assert_allclose(expected, actual, atol=1e-6, rtol=1e-6)
    finally:
        env.close()
