"""Installed resources and explicitly selected output locations."""

from importlib.resources import files
from pathlib import Path


def resource_path(relative: str) -> Path:
    """Return an installed package resource, independent of the source checkout."""
    path = Path(str(files("drone_playground").joinpath(relative)))
    if not path.exists():
        raise FileNotFoundError(f"Missing installed drone_playground resource: {relative}")
    return path
