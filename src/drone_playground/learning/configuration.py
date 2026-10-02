"""Recurrent learner and physical-adaptation configuration contracts."""

def validate_reconstruction_config(config):
    """Validate capabilities independently of the historical P5 task constants."""
    if config["method"]["output"] != "world_acceleration":
        raise ValueError(
            "Paper policy requires the three-dimensional acceleration command contract"
        )
    if config["method"].get("network_output_frame") != "body":
        raise ValueError(
            "Paper policy network output frame must be body before R @ action"
        )
    if config["method"].get("command_units") != "m/s^2":
        raise ValueError("Paper policy command units must be m/s^2")
    if (
        config["env"]["action"]["controller"]["name"]
        != "acceleration_passthrough"
    ):
        raise ValueError("Select the point-mass acceleration execution preset")
    if config["env"]["dynamics"]["forward"] != "point_mass_lag":
        raise ValueError(
            "This method adapter qualifies the point-mass forward model"
        )
    if config["algorithm"]["gradient"]["transition"] not in (
        "direct",
        "exponential",
    ):
        raise ValueError("Unqualified point-mass derivative rule")
    if config["algorithm"]["name"] != "recurrent_bptt":
        raise ValueError("Select the recurrent paper BPTT trainer")
    frequency = config["env"]["task"]["freq"]
    if frequency <= 0 or config["env"]["task"]["physics_freq"] % frequency:
        raise ValueError(
            "Evaluation physics frequency must be divisible by the policy frequency"
        )
    sensor_frequency = config["env"]["sensor"]["source_rate_hz"]
    if sensor_frequency != frequency:
        raise ValueError(
            "The synchronous paper sensor frequency must equal the policy frequency; "
            "asynchronous cached observations require a separately qualified preset"
        )
    if (
        config["env"]["task"]["body_radius"] <= 0
        or config["env"]["task"]["duration"] <= 0
    ):
        raise ValueError("Body radius and duration must be positive")
    settings = config["training"]
    if (
        settings["num_envs"] < 1
        or settings["policy_updates"] < 1
        or settings["seed"] < 0
    ):
        raise ValueError(
            "Training counts must be positive and seed nonnegative"
        )
    horizon = config["algorithm"]["horizon_length"]
    if horizon < 1:
        raise ValueError("BPTT horizon must be positive")
    budget = settings["policy_updates"] * settings["num_envs"] * horizon
    if settings.get("num_timesteps") not in (None, budget):
        raise ValueError(
            "Declared budget differs from updates × environments × horizon"
        )
    if (
        config.get("mode", "train") == "train"
        and config["env"]["scene"]["name"] == "navigation"
    ):
        raise ValueError(
            "Navigation8 is held out for frozen-policy transfer evaluation"
        )
    if config["env"]["scene"]["name"] == "navigation":
        validate_navigation_protocol(config)


def validate_navigation_protocol(config):
    """Bind the nominal Navigation8 result label to its preregistered transfer protocol."""
    if config["evaluation"].get("protocol"):
        from drone_playground.benchmarks import protocol_identity

        protocol_identity(config)


def validate_navigation_adaptation(config):
    task, algo, settings = (
        config["env"]["task"],
        config["algorithm"],
        config["training"],
    )
    implementations = {
        "pointcloud_recurrent": "recurrent_navigation_bptt",
        "depth_recurrent": "recurrent_navigation_bptt",
    }
    implementation = config["method"]["implementation"]
    if implementation not in implementations:
        raise ValueError(
            "Navigation adaptation requires a qualified recurrent flight policy"
        )
    if (
        config["method"]["output"],
        config["method"].get("network_output_frame"),
        config["method"].get("command_units"),
    ) != ("world_acceleration", "body", "m/s^2"):
        raise ValueError(
            "The policy emits body-frame acceleration converted to world m/s^2"
        )
    if (
        config["env"]["action"]["controller"]["name"]
        != "acceleration_passthrough"
    ):
        raise ValueError(
            "Navigation adaptation cannot substitute an implicit controller"
        )
    # Named benchmarks validate their fixed conditions in protocols.py.
    # The reusable task only requires physically meaningful parameters.
    if (
        task["freq"] <= 0
        or task["physics_freq"] <= 0
        or task["physics_freq"] % task["freq"]
        or min(task["duration"], task["goal_radius"], task["body_radius"]) <= 0
    ):
        raise ValueError(
            "Navigation requires positive duration/radii and integral physics substeps"
        )
    if algo["name"] != implementations[implementation]:
        raise ValueError("Use the explicit navigation adaptation trainer")
    if config["env"]["dynamics"]["forward"] != "point_mass_lag":
        raise ValueError("The qualified adapter uses the point-mass lag model")
    if config["env"]["sensor"]["source_rate_hz"] != task["freq"]:
        raise ValueError(
            "Recurrent policy and exteroceptive measurements must have matching clocks"
        )
    delay = config["runtime"].get("action_delay_ms")
    if (
        delay is None
        or delay[1] > 1000 / task["freq"]
        or config["runtime"].get("action_delay_steps", 0)
    ):
        raise ValueError(
            "Choose millisecond transport delay within one policy interval"
        )
    if (
        min(
            settings["num_envs"],
            settings["policy_updates"],
            algo["horizon_length"],
        )
        < 1
    ):
        raise ValueError("Positive training counts are required")
    expected = (
        settings["num_envs"]
        * settings["policy_updates"]
        * algo["horizon_length"]
    )
    if settings.get("num_timesteps") not in (None, expected):
        raise ValueError("Declared interaction and update budgets disagree")
    if settings.get("resume") and settings.get("warm_start"):
        raise ValueError("Choose exact continuation or parameter warm start")
    lower, upper = task["command_distribution"]["speed_range_mps"]
    if not 0 < lower <= upper <= task["max_speed"]:
        raise ValueError(
            "Command range must fit within the nominal maximum speed"
        )
    if algo["gradient"]["transition"] not in ("direct", "exponential"):
        raise ValueError("Use the declared point-mass derivative contract")
