# ADR-0002: Crazyflow, dependencies, and code conventions

**Status:** Accepted.
**Approval:** Originally recorded as 2026-10-08; exact time was not retained.
**Implementation:** Present in the package and Pixi configuration.
**Last reviewed:** 2026-10-16T05:28:58Z, working tree based on `7d59162`.

## Context

Crazyflow already provides quadrotor equations, native control, state management,
JAX stepping, and MuJoCo/MJX integration. The project adds research tasks,
sensing, method composition, and experiments around those capabilities.

## Decision

Pin official Crazyflow and use its public `Sim`, `SimData`, control, reset/step
builders, pipelines, and MJCF facilities. Add only missing project behavior in
Simulation. Consider an upstream contribution or fork only when required physics
cannot be expressed through the public API.

Use Python 3.12 and Pixi, with dependencies recorded in `pyproject.toml` and
`pixi.lock`. Public commands use `pixi run`. Follow Google-style Python docstrings,
MuJoCo naming conventions, and a 100-column Ruff limit.

## Consequences

Official Crazyflow remains the source of default physical behavior. Physics,
sensors, networks, and tasks own their respective parameters. ROS1 system
dependencies remain in an independent runtime.
