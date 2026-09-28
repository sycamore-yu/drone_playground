"""Brax network recipes, independent of task construction and execution logging."""

import functools

import jax
from brax.training.agents.apg import networks as apg_networks
from brax.training.agents.ppo import networks as ppo_networks
from flax import linen


def network_factory(config: dict):
    if config.get("sensor_layout"):
        from drone_playground.networks.encoders import (
            SensorLayout,
            perception_network_factory,
        )

        return perception_network_factory(SensorLayout.from_dict(config["sensor_layout"]), config)
    sizes = tuple(config.get("hidden_sizes", [64, 64]))
    if config["algorithm"] == "ppo":
        return functools.partial(
            ppo_networks.make_ppo_networks,
            policy_hidden_layer_sizes=sizes,
            value_hidden_layer_sizes=sizes,
            activation=linen.elu,
            init_noise_std=config.get("init_noise_std", 0.367879),
            distribution_type=config.get("distribution_type", "tanh_normal"),
            noise_std_type="log",
            state_dependent_std=False,
            mean_kernel_init_fn=jax.nn.initializers.orthogonal,
            mean_kernel_init_kwargs={"scale": 0.01},
        )
    if config["algorithm"] in ("apg", "shac"):
        return functools.partial(
            apg_networks.make_apg_networks,
            hidden_layer_sizes=sizes,
            activation=linen.elu,
            layer_norm=config.get("layer_norm", True),
        )
    raise ValueError(f"Unsupported Brax network recipe: {config['algorithm']}")


def make_lotf_network(observation_size, action_size, config, action_bias):
    from lotf.modules import MLP

    activations = {"relu": linen.relu, "elu": linen.elu, "tanh": linen.tanh}
    if config["activation"] not in activations:
        raise ValueError(f"Unknown LOTF network activation: {config['activation']}")
    return MLP(
        [observation_size, *config["hidden_sizes"], action_size],
        initial_scale=config["initial_scale"],
        action_bias=action_bias,
        nonlinearity=activations[config["activation"]],
    )


def lotf_inference_factory(network):
    def make_policy(params, deterministic=False):
        del deterministic

        def apply(observation, key):
            del key
            return network.apply(params[1], observation), {}

        return apply

    return make_policy
