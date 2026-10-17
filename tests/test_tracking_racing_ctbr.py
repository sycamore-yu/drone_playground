"""Public Tracking/Racing policies share physical thrust and body-rate commands."""

import json
import shlex
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hydra import compose, initialize_config_dir

from drone_playground.learning.dynamics import training_step
from drone_playground.simulation.environment import make_environment
from drone_playground.simulation.methods import Action


def test_all_fourteen_builder_recipes_use_identical_rate_control():
    """Changing backward models must not silently change Actor control quantities."""
    root = Path(__file__).resolve().parents[1]
    js = (root / "docs/command-builder.html").read_text().split("<script>")[1].split("</script>")[0]
    cases = json.loads(
        subprocess.check_output(
            ["node"],
            input=js + "\nprocess.stdout.write(JSON.stringify(CommandBuilder.combinations()));",
            text=True,
        )
    )
    count = 0
    with initialize_config_dir(
        version_base=None, config_dir=str(root / "src/drone_playground/configs")
    ):
        for case in cases:
            if case["mode"] != "train" or case["task"] not in ("tracking", "racing"):
                continue
            cfg = compose(config_name="config", overrides=shlex.split(case["command"])[3:])
            assert cfg.method.action.level == "body_rate", case["command"]
            assert cfg.method.action.frame == "body", case["command"]
            assert cfg.initial_checkpoint is None and cfg.resume is None
            assert cfg.learning.loss is None
            assert cfg.learning.best.episodes == 24
            assert cfg.learning.best.confirmation_episodes == 96
            count += 1
    assert count == 14


@pytest.mark.parametrize("model", [None, "lotf", "point_mass_lag"])
@pytest.mark.parametrize("delay", [0.0, 0.04])
def test_ctbr_backward_models_use_due_command_and_native_forward(model, delay):
    """Check real transitions, thrust order and body-rate gradients, including delay."""
    env = make_environment(
        task="tracking",
        num_envs=1,
        action={"level": "body_rate", "frame": "body"},
        action_delay_s=delay,
    )
    state = env.reset(jax.random.key(42))
    action = jnp.array([[0.0, 0.1, -0.15, 0.12]])
    physical = env.action.decode(action)
    np.testing.assert_allclose(
        env.action_setpoint(state, action).value, physical[:, [1, 2, 3, 0]], atol=1e-7
    )
    step = training_step(env, model)
    actual, expected = step(state, action), env.step(state, action)
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        if jax.dtypes.issubdtype(a.dtype, jax.dtypes.prng_key):
            a, b = jax.random.key_data(a), jax.random.key_data(b)
        np.testing.assert_array_equal(a, b)

    def pose(u):
        s = state
        for _ in range(4):
            s = step(s, u)
        return jnp.concatenate([s.physics.states.quat[0, 0], s.physics.states.vel[0, 0]])

    jac = jax.jacrev(pose)(action)[:, 0, :]
    assert np.isfinite(jac).all()
    assert np.linalg.norm(np.asarray(jac)[:, 0]) > 0
    assert all(np.linalg.norm(np.asarray(jac)[:, i]) > 0 for i in (1, 2, 3))
    if delay:
        initial_gradient = jax.jacrev(lambda u: step(state, u).physics.states.quat)(action)
        np.testing.assert_allclose(initial_gradient, 0, atol=1e-8)
    env.sim.close()


def test_retired_navigation_velocity_interface_is_not_public():
    """An abandoned experiment must not retain another controller and reward path."""
    root = Path(__file__).resolve().parents[1]
    assert not list((root / "src/drone_playground/configs/experiment").glob("*ppo_velocity*.yaml"))
    with pytest.raises(ValueError, match="Unknown action level"):
        Action(level="velocity_yaw")


@pytest.mark.parametrize("task,scene", [("tracking", "empty"), ("racing", "racing")])
def test_raw_reference_environment_has_same_default_as_public_recipe(task, scene):
    """Direct environment construction cannot silently reintroduce attitude actions."""
    env = make_environment(task=task, scene=scene)
    assert env.action.level == "body_rate" and env.action.frame == "body"
    env.sim.close()


def test_point_mass_ctbr_hover_and_pitch_have_physical_units():
    """The surrogate adds gravity once and integrates rad/s into attitude."""
    from drone_playground.simulation.dynamics.point_mass import point_mass_body_rate_step

    p = jnp.array([[0.0, 0.0, 2.0]])
    q = jnp.array([[0.0, 0.0, 0.0, 1.0]])
    v = jnp.zeros_like(p)
    mass = 0.034
    u = jnp.array([[mass * 9.81, 0.0, 0.0, 0.0]])
    pp, qq, vv, aa = point_mass_body_rate_step(p, q, v, v, u, 0.01, mass=mass)
    np.testing.assert_allclose(pp, p, atol=1e-7)
    np.testing.assert_allclose(qq, q, atol=1e-7)
    np.testing.assert_allclose(vv, 0, atol=1e-7)
    np.testing.assert_allclose(aa, 0, atol=1e-6)
    u = u.at[0, 2].set(1.0)
    pp, qq, vv, aa = point_mass_body_rate_step(p, q, v, v, u, 0.01, mass=mass)
    assert float(qq[0, 1]) > 0
    pp, qq, vv, aa = point_mass_body_rate_step(pp, qq, vv, aa, u, 0.01, mass=mass)
    assert float(vv[0, 0]) > 0
