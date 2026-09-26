"""Read-only conversion for already-persisted P1–P4 checkpoint configurations.

New experiments use Hydra groups. This decoder preserves the externally stored
checkpoint contract; it does not maintain a second set of task/training code.
"""

import copy


def checkpoint_config(config: dict) -> dict:
    if isinstance(config.get("dynamics"), dict):
        return copy.deepcopy(config)
    if "components" in config:
        return copy.deepcopy(config["components"])
    from drone_playground.composition import compose_config

    task = config.get("task", "figure8")
    algorithm = config["algorithm"]
    cfg = compose_config(f"{task}_{algorithm}")
    cfg["dynamics"].update(
        forward=config.get("dynamics", cfg["dynamics"]["forward"]),
        drone=config.get("drone", cfg["dynamics"]["drone"]),
    )
    for group in ("algorithm", "network", "training", "task"):
        for key in cfg[group]:
            if key in config and key != "name":
                cfg[group][key] = copy.deepcopy(config[key])
    cfg["task"]["numerical_guard"] = config.get("numerical_guard", False)
    cfg["task"]["freq"] = config.get("freq", 50)
    cfg["task"]["reference_count"] = config.get("reference_count", 256)
    # Parameters absent in old records must use the historical native defaults,
    # not whichever later experiment happens to share their name.
    cfg["network"]["hidden_sizes"] = config.get("hidden_sizes", [64, 64])
    cfg["network"]["normalize_observations"] = config.get("normalize_observations", False)
    if algorithm == "ppo":
        cfg["network"]["distribution_type"] = config.get("distribution_type", "tanh_normal")
        cfg["network"]["init_noise_std"] = config.get("init_noise_std", 0.367879)
    else:
        cfg["network"]["layer_norm"] = config.get("layer_norm", True)
    cfg["source"] = "Read-only migrated P1–P4 checkpoint configuration"
    return cfg
