"""Safe flight corridors and trajectory previews with independent validity."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from drone_playground.references import Trajectory, _finite


@dataclass(frozen=True)
class ConvexPolytope:
    """Either vertices [N,3] or halfspaces [M,4] satisfying A x + b <= 0."""

    vertices: np.ndarray | None = None
    halfspaces: np.ndarray | None = None

    def __post_init__(self):
        """Validate and prepare the ConvexPolytope instance after initialization."""
        if (self.vertices is None) == (self.halfspaces is None):
            raise ValueError("A polytope requires exactly one geometry representation")
        field = "vertices" if self.vertices is not None else "halfspaces"
        values = _finite(getattr(self, field))
        width = 3 if field == "vertices" else 4
        if values.ndim != 2 or values.shape[1] != width or len(values) < 4:
            raise ValueError("A bounded 3D polytope requires at least four vertices or planes")
        if field == "vertices" and np.linalg.matrix_rank(values - values[0]) < 3:
            raise ValueError("Polytope vertices require a 3D interior")
        if field == "halfspaces" and np.any(np.linalg.norm(values[:, :3], axis=1) == 0):
            raise ValueError("Halfspace plane normals must be nonzero")
        object.__setattr__(self, field, values)


@dataclass(frozen=True)
class SafeFlightCorridor:
    name: str
    polytopes: tuple[ConvexPolytope, ...]
    generated_at: float = 0.0
    valid_until: float = 0.0
    frame: str = "world"

    def __post_init__(self):
        """Validate and prepare the SafeFlightCorridor instance after initialization."""
        _validate_interval(self)
        if not self.name:
            raise ValueError("A corridor needs a layer name")
        object.__setattr__(
            self,
            "polytopes",
            tuple(ConvexPolytope(**p) if isinstance(p, dict) else p for p in self.polytopes),
        )
        if not all(isinstance(p, ConvexPolytope) for p in self.polytopes):
            raise TypeError("A corridor contains convex polytopes")


@dataclass(frozen=True)
class TrajectoryPreview:
    name: str
    trajectory: Trajectory
    generated_at: float = 0.0
    valid_until: float = 0.0
    frame: str = "world"

    def __post_init__(self):
        """Validate and prepare the TrajectoryPreview instance after initialization."""
        _validate_interval(self)
        if not self.name:
            raise ValueError("A trajectory preview needs a name")
        if isinstance(self.trajectory, dict):
            object.__setattr__(self, "trajectory", Trajectory(**self.trajectory))
        if not isinstance(self.trajectory, Trajectory):
            raise TypeError("A trajectory preview uses the physical Trajectory contract")


def _validate_interval(value):
    if (
        value.frame != "world"
        or not np.isfinite([value.generated_at, value.valid_until]).all()
        or not 0 <= value.generated_at <= value.valid_until
    ):
        raise ValueError("Inspection output requires world metres and a finite valid interval")
