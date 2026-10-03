"""The pinned LSY racing course, without training noise or randomization."""

from dataclasses import dataclass

import mujoco
from ml_collections import ConfigDict
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation

from drone_playground.resources import resource_path


def load_lsy_config():
    """Adapt actual MJCF geometry and explicit YAML values to the upstream API."""
    config = ConfigDict(
        OmegaConf.to_container(
            OmegaConf.load(resource_path("assets/scenes/racing/lsy_native.yaml")), resolve=True
        )
    )
    model = mujoco.MjModel.from_xml_path(str(resource_path("assets/scenes/racing/lsy_level0.xml")))
    track = config.env.track
    track.gates, track.obstacles = [], []
    for index in range(model.nbody):
        body = model.body(index)
        record = dict(
            pos=body.pos.tolist(),
            rpy=Rotation.from_quat(body.quat, scalar_first=True).as_euler("xyz").tolist(),
        )
        if body.name.startswith("gate:"):
            track.gates.append(ConfigDict(record))
        elif body.name.startswith("obstacle:"):
            track.obstacles.append(ConfigDict(record))
    start = model.site("drone_start")
    track.drones = [
        ConfigDict(
            dict(
                pos=start.pos.tolist(),
                rpy=Rotation.from_quat(start.quat, scalar_first=True).as_euler("xyz").tolist(),
                vel=model.numeric("initial_velocity").data.tolist(),
                ang_vel=model.numeric("initial_angular_velocity").data.tolist(),
            )
        )
    ]
    track.gate_order = model.numeric("gate_order").data.astype(int).tolist()
    track.safety_limits = ConfigDict(
        dict(
            pos_limit_low=model.numeric("world_low").data.tolist(),
            pos_limit_high=model.numeric("world_high").data.tolist(),
        )
    )
    return config


@dataclass
class RacingScene:
    name: str = "lsy_level0"

    def create_core(self, **kwargs):
        """Retain native gate rules while loading the course directly from MJCF."""
        from drone_playground.environments.tasks.lsy_upstream.race_core import RaceCoreEnv

        source = resource_path("assets/scenes/racing/lsy_level0.xml")

        class AssetRaceCore(RaceCoreEnv):
            gate_spec_path = source

            def _load_track_into_sim(self, track):
                del track
                spec = mujoco.MjSpec.from_file(str(source))
                spec.compiler.texturedir = str(source.parent)
                self.sim.spec.attach(
                    spec, prefix="", suffix="", frame=self.sim.spec.worldbody.add_frame()
                )
                self.sim.build_mjx()

        return AssetRaceCore(**kwargs)

    def config(self, model):
        cfg = load_lsy_config()
        cfg.sim.dynamics = getattr(model, "scene_reference_dynamics", model.forward)
        cfg.sim.drone = getattr(model, "scene_reference_drone", model.drone)
        cfg.env.disturbances = {}
        cfg.env.randomizations = {}
        return cfg
