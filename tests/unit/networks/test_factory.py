"""Configured networks must be real modules, not unconsumed configuration mappings."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize("encoder,shape", [("pointnet", (2, 7, 3)), ("depth_cnn", (2, 48, 64))])
def test_recurrent_builder_consumes_encoder_and_memory(encoder, shape):
    from drone_playground.networks.factory import build_network

    network = build_network({"encoder": {"type": encoder}, "memory": "gru", "hidden_size": 192})
    inputs = jnp.ones(shape)
    valid = jnp.ones(shape[:-1] if encoder == "pointnet" else shape, dtype=bool)
    state, memory = jnp.zeros((2, 10)), jnp.zeros((2, 192))
    params = network.init(jax.random.PRNGKey(7), inputs, valid, state, memory)
    action, next_memory = network.apply(params, inputs, valid, state, memory)
    assert action.shape == (2, 3)
    assert next_memory.shape == memory.shape
    assert np.isfinite(action).all()


def test_state_mlp_uses_requested_widths():
    from drone_playground.networks.factory import build_network

    network = build_network({"algorithm": "bptt", "hidden_sizes": [32, 16]}, 43, 4)
    params = network.policy_network.init(jax.random.PRNGKey(1))
    from brax.training.agents.apg.networks import make_inference_fn

    policy = make_inference_fn(network)((None, params), deterministic=True)
    action, _ = policy(jnp.ones((2, 43)), jax.random.PRNGKey(2))
    assert action.shape == (2, 4)
    assert np.isfinite(action).all()


def test_unimplemented_recurrent_fields_are_not_silently_ignored():
    from drone_playground.networks.factory import build_network

    with pytest.raises(ValueError, match="Unsupported"):
        build_network({"encoder": {"type": "pointnet"}, "memory": "transformer"})
