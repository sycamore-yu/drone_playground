"""Construct configured networks for both training and frozen inference."""

import functools
from collections.abc import Mapping

import jax
from brax.training.agents.apg import networks as apg_networks
from brax.training.agents.ppo import networks as ppo_networks
from flax import linen


def build_network(config, observation_size=None, action_size=None, **kwargs):
    """Build a recurrent Flax module or a dimensioned Brax policy/value network.

    Recurrent modules own their input shape. Brax supplies observation/action
    dimensions through ``network_factory`` when its learner constructs a policy.
    Recorded pre-migration network targets are decoded here, without importing
    retired source modules or changing their parameter tree.
    """
    from drone_playground.networks.recurrent import DepthCnnGruPolicy, PointNetGruPolicy

    spec = dict(config.get("network", config))
    target = spec.get("_target_")
    recorded = {
        "drone_playground.networks.pointcloud.PointCloudPolicy": "pointnet",
        "drone_playground.networks.pointnet_gru.PointNetGruPolicy": "pointnet",
        "drone_playground.networks.depth_flight.DepthFlightPolicy": "depth_cnn",
        "drone_playground.networks.cnn_gru.DepthCnnGruPolicy": "depth_cnn",
    }
    encoder = spec.get("encoder", {})
    kind = encoder.get("type") if isinstance(encoder, Mapping) else encoder
    kind = kind or recorded.get(target)
    if kind in ("pointnet", "depth_cnn") or spec.get("memory"):
        if spec.get("memory", "gru") != "gru" or kind not in ("pointnet", "depth_cnn"):
            raise ValueError("Unsupported recurrent encoder or memory")
        if target is not None and recorded.get(target) != kind:
            raise ValueError(f"Unsupported recurrent network target: {target}")
        values = {key: value for key, value in spec.items() if key not in ("encoder", "memory", "name", "_target_")}
        if isinstance(encoder, Mapping):
            values.update({key: value for key, value in encoder.items() if key != "type"})
        allowed = {"hidden_size", "point_channels", "negative_slope", "point_input_scale", "output_init_scale"} if kind == "pointnet" else {"hidden_size", "input_shape", "far_m"}
        if set(values) - allowed:
            raise ValueError(f"Unsupported {kind} fields: {sorted(set(values) - allowed)}")
        for field in ("point_channels", "input_shape"):
            if field in values:
                values[field] = tuple(values[field])
        if int(values.get("hidden_size", 192)) <= 0:
            raise ValueError("Recurrent hidden_size must be positive")
        return (PointNetGruPolicy if kind == "pointnet" else DepthCnnGruPolicy)(**values)
    if target is not None:
        raise ValueError(f"Unsupported network target: {target}")
    if observation_size is None or action_size is None:
        raise ValueError("Brax networks require observation_size and action_size")
    if "network" in config:
        spec["algorithm"] = config["algorithm"]["name"]
    return _brax_network_factory(spec)(observation_size, action_size, **kwargs)


def network_factory(config):
    """Bind the shared constructor to Brax's dimension-supplying callback."""
    return functools.partial(build_network, config)


def _brax_network_factory(config: dict):
    if config.get("sensor_layout"):
        from drone_playground.networks.perception import SensorLayout, perception_network_factory

        return perception_network_factory(
            SensorLayout.from_dict(config["sensor_layout"]), config
        )
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
    if config["algorithm"] in ("apg", "bptt", "shac"):
        return functools.partial(
            apg_networks.make_apg_networks,
            hidden_layer_sizes=sizes,
            activation=linen.elu,
            layer_norm=config.get("layer_norm", True),
        )
    raise ValueError(f"Unsupported Brax network recipe: {config['algorithm']}")
