"""Compare injected components to the pre-migration 12-cell transition fixture."""

import hashlib
import json
from pathlib import Path

import crazyflow  # noqa: F401
import jax
import jax.numpy as jnp
import numpy as np
import pytest

ROOT = Path(__file__).parent / "fixtures"
META = json.loads((ROOT / "composition-baseline.json").read_text())


@pytest.mark.parametrize("case", META["cases"], ids=lambda c: f"{c['task']}-{c['dynamics']}")
def test_preserved_transition(case):
    from drone_playground.composition import build_environment
    from tests.reference_configs import compose_reference as compose_config

    path = ROOT / "composition-baseline.npz"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == META["sha256"]
    fixture = np.load(path)
    cfg = compose_config(case["task"] + "_ppo", [f"dynamics.forward={case['dynamics']}"])
    env = build_environment(cfg, device="cpu", split="dev", count=2)
    try:
        name = case["task"] + "__" + case["dynamics"]
        initial = env.reset(jax.random.PRNGKey(case["key"]))
        actions = jnp.asarray(fixture[name + "__actions"])

        def step(state, action):
            nxt = env.step(state, action)
            return nxt, (nxt.obs, nxt.reward, nxt.done)

        _, actual = jax.jit(lambda s, a: jax.lax.scan(step, s, a))(initial, actions)
        for field, value in zip(("obs", "reward", "done"), actual):
            np.testing.assert_allclose(
                value,
                fixture[name + "__" + field],
                atol=3e-6,
                rtol=3e-6,
                err_msg=name + "/" + field,
            )
    finally:
        env.close()
