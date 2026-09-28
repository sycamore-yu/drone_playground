"""Scene presets own geometry and native randomization, independently of reward."""

import tomllib
from dataclasses import dataclass
from pathlib import Path

from ml_collections import ConfigDict


@dataclass
class EmptyScene:
    name: str = "empty"
    takeoff: tuple = (-1.5, 1.0, 0.07)


def load_lsy_config():
    path = Path(__file__).parents[1] / "tasks/lsy_upstream/level0.toml"
    return ConfigDict(tomllib.loads(path.read_text()))


@dataclass
class LSYScene:
    name: str = "lsy_level0"
    disturbances: bool = True

    def config(self, model):
        cfg = load_lsy_config()
        cfg.sim.dynamics, cfg.sim.drone = model.forward, model.drone
        if not self.disturbances:
            cfg.env.disturbances = {}
        return cfg
