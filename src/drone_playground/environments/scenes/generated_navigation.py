"""Seeded training-only geometry, independent of the Navigation8 catalog."""

from dataclasses import dataclass

import jax.numpy as jnp

from drone_playground.environments.scenes.geometry import DYNAMIC_FAMILIES, STATIC_FAMILIES
from drone_playground.environments.scenes.procedural_navigation import ProceduralNavigationScene


@dataclass(frozen=True)
class GeneratedNavigationScene:
    """A finite, reproducible bank of short static and dynamic courses.

    Reuses the recorded SANDO-style generator; no benchmark catalog or route
    is read. Difficulty, placement checks and physical primitives retain that
    generator's semantics. This is a fixed generated training distribution.
    """

    name: str = "navigation_training"
    seed: int = 81000
    per_difficulty: int = 4

    def __post_init__(self):
        """Validate and prepare the GeneratedNavigationScene instance after initialization."""
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("Training geometry requires a nonnegative integer seed")
        if type(self.per_difficulty) is not int or self.per_difficulty < 2:
            raise ValueError("At least two instances per difficulty cover both families")

    def build(self):
        static, static_manifest = ProceduralNavigationScene(families=STATIC_FAMILIES).build(
            self.seed, self.per_difficulty
        )
        dynamic, dynamic_manifest = ProceduralNavigationScene(
            families=DYNAMIC_FAMILIES, dynamic=True
        ).build(self.seed + 1, self.per_difficulty)
        dynamic = dynamic.replace(subtype=dynamic.subtype + len(STATIC_FAMILIES))
        fields = (
            "kind",
            "size",
            "origin",
            "motion",
            "params",
            "active",
            "start",
            "goal",
            "difficulty",
            "subtype",
        )
        bank = static.replace(
            **{
                field: jnp.concatenate((getattr(static, field), getattr(dynamic, field)))
                for field in fields
            },
            subtype_names=STATIC_FAMILIES + DYNAMIC_FAMILIES,
        )
        manifest = dict(
            name=self.name,
            role="training",
            generator_seed=self.seed,
            instance_count=bank.num_instances,
            bank_digest=bank.digest(),
            benchmark_geometry_used=False,
            geometry_seed_role="finite training bank; fixed before optimization",
            training_distribution=(
                "fixed generated bank: 20x10x5m courses, 15m departure-to-goal distance"
            ),
            connectivity_scope=(
                "generator's conservative horizontal grid at t=0; not all-time dynamic reachability"
            ),
            groups=[static_manifest, dynamic_manifest],
        )
        return bank, manifest
