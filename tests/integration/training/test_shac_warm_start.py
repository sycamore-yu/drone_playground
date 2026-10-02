"""A requested actor warm start must be loaded, not silently ignored."""

import jax
import jax.numpy as jnp
import numpy as np
from brax.training.acme import running_statistics, specs

from drone_playground.artifacts.checkpoints import save_policy
from drone_playground.composition import compose_experiment, native_training_config
from drone_playground.environments.tasks.tracking.rigid_body import TrackingEnv
from drone_playground.learning.algorithms.shac import train
from drone_playground.networks.factory import network_factory


def test_shac_initial_snapshot_matches_requested_actor_and_then_updates(tmp_path):
    source = compose_experiment('control/bptt', "tracking")
    source["network"].update(hidden_sizes=[8], normalize_observations=False)
    native = native_training_config(source)
    net = network_factory(native)(43, 4)
    expected = (
        running_statistics.init_state(specs.Array((43,), jnp.float32)),
        net.policy_network.init(jax.random.PRNGKey(123)),
    )
    checkpoint = save_policy(tmp_path, expected, native, 100)
    config = dict(
        native,
        algorithm="shac",
        warm_start=str(checkpoint),
        num_envs=2,
        horizon_length=2,
        policy_updates=1,
        num_evals=2,
        critic_updates=1,
        learning_rate=0.0001,
        use_schedule=False,
    )
    snapshots = []
    env = TrackingEnv(device="cpu")
    try:
        _, _, metrics = train(
            env, config, policy_params_fn=lambda step, maker, params: snapshots.append(params)
        )
    finally:
        env.close()
    for actual, wanted in zip(
        jax.tree.leaves(snapshots[0]), jax.tree.leaves(expected), strict=True
    ):
        np.testing.assert_array_equal(actual, wanted)
    assert metrics["actor_parameter_delta_l2"] > 0
    assert metrics["critic_parameter_delta_l2"] > 0
    assert metrics["actual_steps"] == 4
