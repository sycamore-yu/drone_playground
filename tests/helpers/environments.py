"""Small physical fixtures built through the public Hydra composition API."""

from drone_playground.configuration import load_config
from drone_playground.environments.environment import build_environment


def tracking_environment(
    *,
    task="tracking",
    reference="figure8",
    dynamics="so_rpy",
    device="cpu",
    reference_count=4,
    reference_seed=20000,
    numerical_guard=False,
    freq=50,
    duration=10.0,
    drone="cf2x_L250",
):
    """Select the reference and physical model without a separate environment class."""
    if task == "random":
        task, reference = "tracking", "random"
    config = load_config("environment", ["env=" + task])
    config["env"]["task"].update(
        freq=freq,
        duration=duration,
        reference_count=reference_count,
        numerical_guard=numerical_guard,
    )
    config["env"]["reference"]["name"] = reference
    config["env"]["dynamics"].update(forward=dynamics, drone=drone)
    config["runtime"].update(device=device, scene_seed_eval=reference_seed)
    return build_environment(config, role="eval", count=reference_count)
