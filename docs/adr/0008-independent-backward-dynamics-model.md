# ADR-0008: Independent backward dynamics

**Status:** Accepted.
**Approval:** Originally recorded as 2026-10-09; exact time was not retained.
**Implementation:** Present in `learning/dynamics.py` and `simulation/dynamics/`.
**Last reviewed:** 2026-10-16T05:28:58Z, working tree based on `7d59162`.

## Context

The user needs LOTF or point-mass equations for differentiation while retaining
the selected forward simulation. A backward dynamics model has physical state,
inputs, parameters, and transition equations; gradient decay cannot replace it.

## Decision

Keep `simulation.dynamics` as the forward-model choice. Select the derivative
model independently with `learning.backward_model`: null uses native derivatives;
`point_mass_lag` requires acceleration actions; `lotf` requires thrust/body-rate
actions. Existing presets are `tracking_point_mass` and `tracking_lotf`.

Dynamics equations live in `simulation/dynamics/` and remain independently callable.
Learning connects their derivatives to the shared transition. Methods/controllers
own command conversion; the environment owns actual clocks, sensing, events,
and state advancement. PPO does not use pathwise dynamics derivatives and does
not execute an extra backward model.

Keep one mathematical implementation per model. Prediction and any separately
approved forward experiment may reuse it. This decision does not prohibit new
forward models, and does not require importing LOTF's ROS deployment or Betaflight.

## Consequences

Changing only the backward model must preserve forward states, events, and
measurements for identical inputs and random state. Define state mappings, units,
frames, timesteps, and unmodeled-output derivatives at the differentiation boundary.
Reject incompatible action meanings during construction.

Point-mass derivatives replace position, velocity, and acceleration. LOTF replaces
position, orientation, velocity, and derived acceleration. Unmodeled outputs retain
native Crazyflow derivatives. Training metadata records the selected model and
parameters; frozen inference does not load the backward model.

## Evidence

- [LOTF paper](https://arxiv.org/html/2508.21065v2) and [official equations](https://github.com/uzh-rpg/learning_on_the_fly/blob/master/lotf/objects/quadrotor_obj.py).
- [DiffAero](https://arxiv.org/html/2509.10247v1) and [JAX custom derivatives](https://docs.jax.dev/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html).
- [Control-model configuration](../control-models.md) and `tests/test_control_models.py`.
