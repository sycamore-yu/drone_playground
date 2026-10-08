"""Physical composition gradients, actuator delays, and public training contracts."""

import crazyflow  # noqa: F401
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from drone_playground.configuration import compose_experiment, validate_config
from drone_playground.control.controllers.trajectory import TrajectoryTracking
from drone_playground.control.controllers.trajectory_jax import JaxTrajectoryTracking
from drone_playground.environments.factory import build_environment


def config_for(kind):
    config = compose_experiment(
        "control/geometric",
        "hovering",
        [
            "env.task.duration=0.2",
            "training.num_envs=2",
            "training.policy_updates=2",
            "algorithm.horizon_length=8",
            "training.num_evals=2",
            "network.hidden_sizes=[8,8]",
            "training.checkpoint_eval_episodes=2",
            "runtime.device=cpu",
        ],
    )
    if kind == "waypoint":
        config["method"]["output"] = config["method"]["physical_decoder"]["kind"] = "waypoint"
        config["env"]["controller"].update(input_kind="waypoint", lead_seconds=0.0)
    return config


def test_jax_pd_preserves_host_force_geometry_and_has_finite_hover_derivatives():
    low, high = np.array([-1.0, -1.0, -3.2, 0.0]), np.array([1.0, 1.0, 3.2, 1.0])
    original = TrajectoryTracking().bind(low, high)
    differentiable = JaxTrajectoryTracking().bind(low, high)
    rng = np.random.default_rng(2)
    for _ in range(8):
        body = dict(
            pos=rng.normal(0, 0.2, 3),
            vel=rng.normal(0, 0.1, 3),
            quat=Rotation.from_euler("xyz", rng.normal(0, 0.2, 3)).as_quat(),
        )
        reference = dict(
            position=rng.normal(0, 0.3, 3),
            velocity=rng.normal(0, 0.2, 3),
            acceleration=rng.normal(0, 0.1, 3),
            yaw=rng.normal(0, 0.2),
        )
        np.testing.assert_allclose(
            differentiable.command(body, reference, 0.03),
            original.command(body, reference, 0.03),
            atol=2e-6,
            rtol=2e-5,
        )
    body = dict(pos=jnp.zeros(3), vel=jnp.zeros(3), quat=jnp.array([0.0, 0.0, 0.0, 1.0]))

    def command(target):
        return differentiable.command(
            body,
            dict(position=target, velocity=jnp.zeros(3), acceleration=jnp.zeros(3), yaw=0.0),
            0.03,
        )

    gradient = jax.jacfwd(command)(jnp.zeros(3))
    assert np.isfinite(gradient).all() and np.linalg.norm(gradient) > 0.1


@pytest.mark.parametrize("kind", ["waypoint", "trajectory"])
def test_geometry_gradient_reaches_delayed_physics_and_queue_stays_actuator_sized(kind):
    config = config_for(kind)
    env = build_environment(config, "cpu")
    try:
        initial = env.reset(jax.random.key(21))
        assert env.action_size == (3 if kind == "waypoint" else 9)
        assert initial.info["delay_command_queue"].shape[-1] == 4
        action = jnp.linspace(-0.08, 0.12, env.action_size)

        def objective(action):
            def step(state, _):
                state = env.step(state, action)
                return state, state.reward

            return jax.lax.scan(step, initial, None, 8)[1].mean()

        value, gradient = jax.jit(jax.value_and_grad(objective))(action)
        assert np.isfinite(gradient).all() and np.linalg.norm(gradient) > 1e-7
        direction = gradient / jnp.linalg.norm(gradient)
        eps = 0.005
        finite = (
            jax.jit(objective)(action + eps * direction)
            - jax.jit(objective)(action - eps * direction)
        ) / (2 * eps)
        np.testing.assert_allclose(jnp.dot(gradient, direction), finite, rtol=0.04, atol=2e-5)
        assert jax.jit(objective)(action + 0.01 * gradient) > value
        batch = jax.jit(jax.vmap(env.step))(
            jax.vmap(env.reset)(jax.random.split(jax.random.key(3), 2)), jnp.stack([action, action])
        )
        assert batch.info["applied_action"].shape == (2, 4)
        assert batch.info["requested_geometric_action"].shape == (2, env.action_size)
    finally:
        env.close()


def test_geometric_training_rejects_opaque_solver_and_invalid_gradient_paths():
    config = config_for("trajectory")
    validate_config(config)
    config["env"]["controller"] = dict(
        _target_="drone_playground.control.controllers.mpc.trajectory.AttitudeMPCTracking"
    )
    with pytest.raises(TypeError, match=r"differentiable"):
        build_environment(config, "cpu")
    config = config_for("trajectory")
    config["env"]["controller"]["lead_seconds"] = 0.0
    with pytest.raises(ValueError, match=r"lead"):
        build_environment(config, "cpu")
    config = config_for("waypoint")
    config["method"]["physical_decoder"]["count"] = 2
    with pytest.raises(ValueError, match=r"one target"):
        build_environment(config, "cpu")


def test_frozen_geometric_policy_keeps_observed_goal_and_jax_tracker_semantics(tmp_path):
    from brax.training.acme import running_statistics, specs

    from drone_playground.learning.brax_configuration import native_training_config
    from drone_playground.learning.checkpointing import save_policy
    from drone_playground.learning.inference import FrozenNeuralCommand
    from drone_playground.networks.factory import network_factory
    from drone_playground.runtime.tracking import ExternalTracking

    config = config_for("trajectory")
    native = native_training_config(config)
    network = network_factory(native)(43, 9)
    params = (
        running_statistics.init_state(specs.Array((43,), jnp.float32)),
        network.policy_network.init(jax.random.key(3)),
    )
    checkpoint = save_policy(tmp_path / "weights", params, native, 0)
    env = build_environment(config, "cpu")
    try:
        state = env.reset(jax.random.key(13))
        body = env.controller_observation(state)
        module = FrozenNeuralCommand(checkpoint).bind(env.env, tmp_path)
        module.start({}, [100, 100, 100], None, {})
        reply = module.step(
            dict(
                time=0.0,
                position=body["pos"],
                velocity=body["vel"],
                quaternion=body["quat"],
                policy_observation=state.obs,
                goal=[100, 100, 100],
            ),
            None,
        )
        action = module.policy.act(state.obs)
        curve = reply["output"]
        expected = env.task.observation.reference_goal(state.obs) + action[:3]
        np.testing.assert_allclose(curve.sample(curve.end_time)["position"], expected, atol=2e-5)
        tracker = ExternalTracking(env.env, state, tmp_path)
        command = tracker.command(dict(reply, trajectory=curve), state, 0)
        np.testing.assert_allclose(command, env.policy_command(state, action), atol=2e-6, rtol=2e-5)
        assert tracker.consumed == 1
        assert module.provenance["goal_source"] == "observation_reference"
    finally:
        env.close()


def test_geometric_warm_start_cannot_reinterpret_identically_sized_parameters():
    import copy

    from drone_playground.learning.checkpointing import require_matching_physical_decoder

    config = config_for("trajectory")
    metadata = dict(
        physical_decoder=copy.deepcopy(config["method"]["physical_decoder"]),
        config=copy.deepcopy(config),
    )
    require_matching_physical_decoder(metadata, config)
    config["method"]["physical_decoder"]["position_scale_m"][0] = 2.0
    with pytest.raises(ValueError, match=r"physical decoder"):
        require_matching_physical_decoder(metadata, config)
    config = config_for("trajectory")
    config["method"]["goal_source"] = "task_goal"
    with pytest.raises(ValueError, match=r"goal source"):
        require_matching_physical_decoder(metadata, config)
