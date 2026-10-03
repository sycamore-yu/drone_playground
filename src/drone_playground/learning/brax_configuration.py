"""Explicit lowering from owned config namespaces to the Brax learner boundary."""

from drone_playground.environments.environment import build_observer, build_sensor


def sensor_layout(config):
    """Derive the existing encoder layout from the actual observation specification."""
    sensor = build_sensor(config)
    observer = build_observer(config, sensor)
    if sensor is None or not hasattr(observer, "sensor_size"):
        return None
    from drone_playground.networks.perception import SensorLayout

    grid = (
        (sensor.width // sensor.stride, sensor.height // sensor.stride)
        if observer.name == "navigation_depth"
        else None
    )
    return SensorLayout.from_observation(observer, grid=grid).as_dict()


def native_training_config(config):
    """Lower once; duplicate keys are errors rather than order-dependent overrides.

    Only the selected learner consumes this flat vocabulary. The saved canonical
    config retains algorithm, network and training as separate namespaces.
    """
    output, owners = {}, {}
    for namespace in ("algorithm", "network", "training"):
        for name, value in config[namespace].items():
            if name == "name":
                continue
            if name in owners:
                raise ValueError(f"Ambiguous learner field {name}: {owners[name]} and {namespace}")
            owners[name] = namespace
            output[name] = value
    task = config["env"]["task"]
    dynamics = config["env"]["dynamics"]
    output.update(
        algorithm=config["algorithm"]["name"],
        task=task["name"],
        dynamics=dynamics["forward"],
        drone=dynamics["drone"],
        freq=task["freq"],
        reference_count=task.get("reference_count", 1),
        numerical_guard=task.get("numerical_guard", False),
        device=config["runtime"]["device"],
        observation_size=build_observer(config, build_sensor(config)).size,
        sensor_layout=sensor_layout(config),
        components=config,
        config_version=4,
    )
    return output
