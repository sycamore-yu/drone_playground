"""Load the project's Hydra files without constructing experiments or environments."""

from contextlib import nullcontext
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

CONFIG_ROOT = Path(__file__).resolve().parents[2] / "configs"


def load_config(name="config", overrides=()):
    """Resolve one configuration while preserving an already active CLI context."""
    current = GlobalHydra.instance()
    if current.is_initialized():
        roots = current.config_loader().get_search_path().get_path()
        if not any(
            entry.provider == "main"
            and Path(entry.path.removeprefix("file://")).resolve() == CONFIG_ROOT
            for entry in roots
        ):
            raise ValueError("Active Hydra belongs to a different configuration root")
        context = nullcontext()
    else:
        context = initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_ROOT))
    with context:
        return OmegaConf.to_container(
            compose(config_name=name, overrides=list(overrides)),
            resolve=True,
            throw_on_missing=True,
        )
