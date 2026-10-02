"""An obstacle-free world for hovering and reference tracking."""

from dataclasses import dataclass


@dataclass
class EmptyScene:
    name: str = "empty"
    takeoff: tuple = (-1.5, 1.0, 0.07)
