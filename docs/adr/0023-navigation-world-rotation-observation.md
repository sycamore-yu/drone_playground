# ADR-0023: World-frame navigation actions with orientation observations

**Status:** Accepted.
**Approval recorded:** 2026-10-14T03:53:53Z. The user chose to retain world-frame actions and add orientation to the Actor input.
**Implementation:** Implemented in `0ac3c53` and integrated into the current `main` tree. Training convergence remains an experimental result.

**Last reviewed:** 2026-10-16T05:37:19Z against `main@7d59162` and the working tree.

## Context

The original Navigation input contained body-frame velocity, target velocity,
and the vertical direction, while the Actor produced world-frame acceleration.
Two states differing by a joint rotation of position and heading could have
identical inputs but require different world-frame actions. A paired-state test
with a five-degree heading difference reproduced this ambiguity. This identifies
an observation gap; it does not establish the sole cause of training failure.

## Decision

The explicit option `observation.navigation_world_rotation=true` appends the
row-major, nine-element body-to-world rotation matrix `R_WB` to Navigation state
features. It satisfies `v_world = R_WB @ v_body` and uses the existing Crazyflow
`xyzw` quaternion. The full matrix avoids Euler-angle discontinuities and gives
identical observations for `q` and `-q`.

The original input has ten elements. Orientation alone produces nineteen;
orientation with the separately approved goal height/distance features produces
twenty-one. The state projection adapts to this width; CNN/PointNet, GRU, PPO,
action coordinates, heading control, reward, sensors, and physics retain their
existing contracts. Environment constructs the features and saves the observation
configuration with the experiment and checkpoint. Frozen execution cannot
reinterpret weights with a different input width.

## Verification

Paired-state tests verify the original ambiguity, recovery of world-frame target
velocity, preservation of the original ten-element prefix, quaternion sign
invariance, and unchanged action contracts. Six CPU PPO updates cover depth and
LiDAR at widths 12, 19, and 21, including parameter changes and complete state
save/restore. These are interface and update checks.

Depth and LiDAR experiments use independent random initialization. The original
single-variable comparison retained starts at least 80 m from the goal; mixed
start distances belong to a separate experiment. This approval did not replace
BPTT/SHAC or change action coordinates. The former 23/25 and first-validation-pass
requirements are historical; ADR-0026 defines the current evaluation workflow.
