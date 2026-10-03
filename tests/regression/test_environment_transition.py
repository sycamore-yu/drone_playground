"""Compare injected components to the unchanged tracking transition fixture."""

import hashlib
import json

import crazyflow  # noqa: F401
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.helpers.paths import FIXTURES_ROOT

ROOT = FIXTURES_ROOT
META = json.loads((ROOT / "composition-baseline.json").read_text())


@pytest.mark.parametrize("case", META["cases"], ids=lambda c: f"{c['task']}-{c['dynamics']}")
def test_preserved_transition(case):
    from drone_playground.composition import compose_experiment
    from drone_playground.environments.environment import build_environment

    path = ROOT / "composition-baseline.npz"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == META["sha256"]
    fixture = np.load(path)
    environment = "tracking"
    task_overrides = []
    if case["task"] != "figure8":
        task_overrides = [
            "dynamics@env.dynamics=crazyflow_so_rpy_cf21b_500",
            "env.task.duration=15.0",
            "env.task.reference=random",
            "env.task.reference_generator.name=random",
        ]
    cfg = compose_experiment(
        "control/ppo",
        environment,
        [
            *task_overrides,
            f"env.dynamics.forward={case['dynamics']}",
            "runtime.action_delay_ms=null",
            "runtime.action_delay_steps=0",
        ],
    )
    env = build_environment(cfg, device="cpu", role="eval", count=2)
    try:
        name = case["task"] + "__" + case["dynamics"]
        initial = env.reset(jax.random.PRNGKey(case["key"]))
        actions = jnp.asarray(fixture[name + "__actions"])

        def step(state, action):
            nxt = env.step(state, action)
            return nxt, (nxt.obs, nxt.reward, nxt.done)

        _, actual = jax.jit(lambda s, a: jax.lax.scan(step, s, a))(initial, actions)
        for field, value in zip(("obs", "reward", "done"), actual, strict=True):
            np.testing.assert_allclose(
                value,
                fixture[name + "__" + field],
                atol=3e-6,
                rtol=3e-6,
                err_msg=name + "/" + field,
            )
    finally:
        env.close()
