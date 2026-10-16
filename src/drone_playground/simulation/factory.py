"""Construct environments and methods from the resolved experiment configuration."""

from hydra.utils import get_class, instantiate
from omegaconf import DictConfig, OmegaConf

from drone_playground.simulation.environment import BaseEnv, make_environment


def create_environment(config: DictConfig, *, num_envs=None, scene=None) -> BaseEnv:
    """Construct the physical input from the chosen method rather than its task name."""
    simulation = OmegaConf.to_container(config.simulation, resolve=True)
    simulation["num_envs"] = config.simulation.num_envs if num_envs is None else num_envs
    implementation = get_class(config.method.implementation._target_)
    action = OmegaConf.to_container(config.method, resolve=True).get("action", {})
    task_options = OmegaConf.to_container(config.task, resolve=True).get("options", {})
    level = None
    if not implementation.trainable:
        level = get_class(config.controller._target_).output_level
    return make_environment(
        task=config.task.name,
        scene=config.scene.name if scene is None else scene,
        reference=config.task.reference,
        duration=config.task.duration,
        reference_seed=config.task.reference_seed,
        task_config=task_options,
        sensor=config.sensor.name,
        sensor_config=OmegaConf.to_container(config.sensor.config, resolve=True),
        point_count=config.sensor.point_count,
        ray_sampling=config.sensor.get("ray_sampling", "full"),
        action=action,
        control_level=level,
        observation_config=OmegaConf.to_container(config.observation, resolve=True),
        **simulation,
    )


def create_method(config):
    """Use Hydra's recursive construction instead of a second method registry."""
    return instantiate(config.method.implementation, _convert_="all")
