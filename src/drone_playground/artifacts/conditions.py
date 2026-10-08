"""Record physical experiment conditions without controlling construction."""

import copy

from drone_playground.environments.factory import build_controller


def experiment_conditions(config):
    """Return the actual declared conditions needed to interpret a comparison."""
    config = config.get("components", config)
    env, method = config["env"], config["method"]
    study = config["study"]
    if study["type"] not in ("method_reproduction", "controlled_comparison"):
        raise ValueError("Unknown experiment study type")
    if study["type"] == "controlled_comparison" and not study.get("conditions_id"):
        raise ValueError("Controlled comparison requires conditions_id")
    controller = build_controller(env["controller"])
    task = env["task"]
    objective = task.get("reward", task.get("loss", {}))
    return copy.deepcopy(
        dict(
            study=study,
            scene=env["scene"],
            dynamics=env["dynamics"],
            observation=task["observation"],
            sensor=env["sensor"],
            action_interface=controller.contract(),
            controller=env["controller"],
            reference=env["reference"],
            control_frequency_hz=env["freq"],
            physics_frequency_hz=env.get("physics_freq"),
            latency={
                name: config["runtime"].get(name)
                for name in ("action_delay_steps", "action_delay_ms")
            },
            safety_margin=dict(
                body_radius_m=task.get("body_radius"),
                clearance_margin_m=objective.get("clearance_margin"),
            ),
            method=method,
        )
    )
