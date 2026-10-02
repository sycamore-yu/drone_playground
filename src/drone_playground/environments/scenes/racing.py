"""The pinned LSY racing course, without training noise or randomization."""

import sysconfig
import tomllib
from dataclasses import dataclass
from pathlib import Path

from ml_collections import ConfigDict


def load_lsy_config():
    relative = Path("assets/scenes/racing/lsy_level0.toml")
    path = Path(__file__).resolve().parents[4] / relative
    if not path.is_file():
        path = Path(sysconfig.get_path("data")) / "share/drone_playground" / relative
    return ConfigDict(tomllib.loads(path.read_text()))


@dataclass
class RacingScene:
    name: str = "lsy_level0"

    def config(self, model):
        cfg = load_lsy_config()
        cfg.sim.dynamics = getattr(
            model, "scene_reference_dynamics", model.forward
        )
        cfg.sim.drone = getattr(model, "scene_reference_drone", model.drone)
        cfg.env.disturbances = {}
        cfg.env.randomizations = {}
        return cfg
