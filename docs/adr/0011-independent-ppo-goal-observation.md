# ADR-0011: Goal height and distance in an independent PPO experiment

**Status:** Accepted for the separate PPO comparison.
**Approval:** Originally recorded as 2026-10-09; exact time was not retained.
**Implementation:** Optional observations and continuous height-boundary cost are present. Original evidence was four targeted checks, not convergence.
**Last reviewed:** 2026-10-16T05:28:58Z, working tree based on `7d59162`.

## Decision

`simulation.navigation_goal_observation=true` adds two features to the ten-element
OA state: goal height error divided by the 5.5 m height range, and remaining goal
distance divided by the nominal 96 m course length. The default remains off.
The depth CNN/GRU192 and LiDAR PointNet/GRU192 remain; their state projection takes
twelve inputs. Ten- and twelve-input weights are distinct contracts.

The approved comparison starts from random parameters, uses an independent sensor
critic and one-step velocity smoothing, and retains height-target, progress,
clearance, and failure costs. The added cost is
`height_boundary_weight * dt * relu(1 m - nearest_height_limit_distance)^2`.
Its default weight is zero; the original comparison used 20. Episode boundaries
and the first-failure cost remain unchanged.

## Consequences

Approval covered a separate PPO experiment. It did not extend new observations
to existing BPTT/SHAC results or combine different inputs into one controlled
comparison. Preserve original completed runs and their configurations. Checkpoint
loading verifies the observation contract. Performance and concurrency studies
must identify which observation/reward recipe they measured.
