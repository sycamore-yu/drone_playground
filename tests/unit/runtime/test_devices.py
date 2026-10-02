"""Execution and prediction own separate explicit device selections."""

import jax
import jax.numpy as jnp


def test_execution_scope_honors_requested_device():
    from drone_playground.runtime.devices import execution_scope

    with execution_scope("cpu"):
        value = jax.jit(lambda x: x + 1)(jnp.ones(3))
    assert {device.platform for device in value.devices()} == {"cpu"}
