"""Real trajectory gradients and explicit checkpoint restoration for BPTT."""

import jax
import jax.numpy as jnp
import numpy as np
from brax.envs.base import State


class SmoothTask:
    observation_size = 2
    action_size = 1
    episode_length = 20

    def reset(self, key):
        return State(
            jnp.zeros(1),
            jnp.zeros(2),
            jnp.float32(0),
            jnp.float32(0),
            {},
            {"terminated": jnp.float32(0)},
        )

    def step(self, state, action):
        position = state.pipeline_state + 0.1 * action
        return state.replace(
            pipeline_state=position,
            obs=jnp.r_[position, 1.0 - position],
            reward=-jnp.square(1.0 - position[0]),
        )


def test_bptt_updates_real_policy_and_restores_all_state(tmp_path):
    from drone_playground.learning.algorithms.bptt import train

    cfg = dict(
        algorithm="bptt",
        task="toy",
        num_envs=4,
        horizon_length=5,
        policy_updates=4,
        learning_rate=0.01,
        hidden_sizes=[8],
        normalize_observations=False,
        num_evals=3,
        use_schedule=False,
        seed=0,
    )
    snapshots = []
    maker, params, metrics = train(
        SmoothTask(),
        cfg,
        policy_params_fn=lambda step, fn, p: snapshots.append((step, p)),
        state_directory=tmp_path,
    )
    assert metrics["actual_steps"] == 80
    assert metrics["actor_parameter_delta_l2"] > 0
    assert [s[0] for s in snapshots] == [0, 40, 80]
    _, restored, continuation = train(
        SmoothTask(), cfg, restore_state=tmp_path / "update-0000002.pkl"
    )
    for expected, actual in zip(jax.tree.leaves(params), jax.tree.leaves(restored)):
        np.testing.assert_allclose(actual, expected, rtol=0, atol=0)
    assert continuation["actual_steps"] == 80


def test_flat_p4_checkpoint_migration_preserves_native_noise_and_model():
    from drone_playground.runs.migration import migrate_v2

    old = dict(
        algorithm="ppo",
        task="racing",
        dynamics="first_principles",
        drone="cf21B_500",
        hidden_sizes=[64, 64],
        distribution_type="normal",
        num_timesteps=4194304,
        num_envs=1024,
        freq=50,
        normalize_observations=False,
        init_noise_std=0.367879,
        discounting=0.94,
        gae_lambda=0.97,
        clipping_epsilon=0.26,
        max_grad_norm=1.5,
    )
    current = migrate_v2(old)
    assert current["network"]["distribution_type"] == "normal"
    assert current["env"]["name"] == "racing"
    assert current["env"]["execution"]["dynamics"]["forward"] == "first_principles"
    assert current["runtime"]["action_delay_ms"] is None
    assert current["algorithm"]["gae_lambda"] == 0.97


def test_shac_sensor_training_uses_the_same_network_as_frozen_inference():
    from brax.training.agents.apg.networks import make_inference_fn

    from drone_playground.learning.algorithms.shac import train
    from drone_playground.networks.policies import network_factory

    class SensorToy(SmoothTask):
        observation_size = 32
        action_size = 4

        def reset(self, key):
            state = super().reset(key)
            return state.replace(obs=jnp.zeros(32))

        def step(self, state, action):
            position = state.pipeline_state + 0.1 * action[:1]
            return state.replace(
                pipeline_state=position,
                obs=jnp.full(32, position[0]),
                reward=-jnp.square(1.0 - position[0]),
            )

    cfg = dict(
        algorithm="shac",
        task="toy",
        num_envs=2,
        horizon_length=2,
        policy_updates=1,
        hidden_sizes=[8],
        num_evals=2,
        seed=0,
        normalize_observations=False,
        learning_rate=0.001,
        critic_updates=1,
        sensor_layout=dict(
            kind="lidar",
            proprioception_size=22,
            history=1,
            points_per_frame=2,
            channels=5,
            grid=None,
            embedding_size=8,
        ),
    )
    maker, params, result = train(SensorToy(), cfg)
    expected = make_inference_fn(network_factory(cfg)(32, 4))(params, deterministic=True)
    actual = maker(params, deterministic=True)
    np.testing.assert_allclose(expected(jnp.zeros(32), None)[0], actual(jnp.zeros(32), None)[0])
