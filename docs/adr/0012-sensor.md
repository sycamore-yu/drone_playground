# ADR-0012: Snapshot sensor acquisition and one ray-query implementation

**Status:** Accepted.
**Approval recorded:** 2026-10-10T08:28:44Z.
**Created:** 2026-10-10T15:17:33+08:00.
**Implementation:** P0/P1 snapshot acquisition, scan resources, adjacent-pose interpolation, delivery delay, and compact state were integrated through [PR #1](https://github.com/sycamore-yu/drone_playground/pull/1). The upstream ray-query replacement remains rejected by its correctness checks.
**Original evidence baseline:** `v0.2@6ec50c7a441bbe169fef40acd34ae574b53428a9`, reviewed 2026-10-10T10:30:23Z with MuJoCo 3.15.0, MuJoCo-LiDAR 0.3.5, and JAX 0.11.2.
**Last reviewed:** 2026-10-16T05:28:58Z, working tree based on `7d59162`.

## Decision

Depth and LiDAR share the single JAX `Scene.raycast()` implementation.
`Scene.clearance()` separately supplies differentiable contact/clearance costs.
MuJoCo-LiDAR supplies the MID360 angle resource. The tested third-party JAX kernel
and official MJX-JAX ray function did not satisfy the existing geometry contract.
There is no permanent alternative backend, per-shape dispatch between engines,
or automatic fallback. A future replacement must pass the original tests and
replace the project intersection implementation in one change.

Current task geometry uses primitives. Replay meshes do not establish a sensor
mesh requirement. No Warp, Taichi, backend registry, or copied upstream kernel
is required by this decision.

## Scan and time contract

Load the MID360 angles once. Read them with explicit per-world phase state, never
a shared mutable generator cursor. The device profile budgets 20,000 queries per
frame at 10 Hz and 1,024 policy points. Valid return count depends on the scene.
Read consecutive 20,000-angle spans from the complete 800,000-angle resource,
wrapping at the end; do not resample a 24,000-angle generator window. The first
frame arrives at 0.1 s using the first span. Partial reset resets only those worlds.
Metadata includes the source SHA-256, phase rule, snapshot semantics, and policy
budget. The supplied approximately -7.212 to 52.164 degree range is preserved;
it does not provide manufacturer per-point timing calibration.

At acquisition time `t_k`, compute the full frame using the vehicle pose and
`Scene.positions(t_k)`. All points share that time and become available at
`t_k + latency`. Scan phase identifies directions, not distinct acquisition times.
Snapshot acquisition deliberately omits within-frame motion distortion and deskew.
For example, 3 m/s relative motion over 0.1 s spans 0.3 m; removing that motion
model cannot be reported as a pure kernel speedup.

The observation state holds scheduling, the latest measurement, phase, delivery
state, and adjacent physical pose. Interpolate noninteger frame times between
adjacent poses, aligning quaternion signs first. Delay queues follow
[ADR-0009](0009-delayed-data-in-episode-state.md). Recurrent policy memory belongs
to the actor. Long pose histories and duplicate pending point clouds were removed.

On 2026-10-11, redundant per-point `times` and `directions_body` were removed from
persistent `Measurement`. Observation broadcasts `point_times` from the single
acquisition time when needed. Sensor configuration, angles, measurements, and
frozen policy inputs stay unchanged. Old full-state PyTrees do not load into the
new template; actor parameters require no conversion. Current loading has no
legacy migration branch.

## Recorded replacement checks

`tools/check_sensor_migration.py` compared seven selected boundary cases with
`mujoco.mj_ray`. MuJoCo-LiDAR 0.3.5 disagreed on six: interior origins for four
solids, plane back-face behavior, and a ray parallel to a box face. The ordinary
external box hit agreed. This is a selected-case result, not a population accuracy
estimate. The project kept its original kernel and test expectations.

The 2026-10-10 official MJX-JAX 3.15.0 investigation found no Cylinder or HField
ray function, despite cylinders throughout the fixed OA suite. Model conversion
also rejected the existing cylinder/box collision combination, and material-free
models caused an empty `mat_rgba` indexing error. A temporary experiment disabled
candidate collisions and supplied placeholder material data. Updating dynamic
geometry produced matching box ranges of 0.600, 0.632, and 0.665 m at 0, 0.5,
and 1 s; stale geometry did not. GPU ray agreement was 430/512 at default matrix
precision and 512/512 at `highest`. The small concurrent-GPU timing was not an
isolated throughput acceptance result. These workarounds were not integrated.

## Verification boundaries

Check geometry replacement, scan-direction changes, and acquisition-time changes
separately. Fixed scene/ray/pose/time/range comparisons cover occlusion, misses,
range boundaries, interior origins, visibility, and self-exclusion. Require
batched JIT, partial reset, delayed delivery, and full-state continuation.
Acquisition gradients remain stopped for BPTT/SHAC; network, physical-state, and
continuous-clearance gradients remain available.

The original integrated sensor work included actual GPU updates for six
sensor/algorithm pairs and frozen eight-scene checks. LiDAR S03 and multiseed
convergence were incomplete. Engineering integration did not change their status.
Performance comparisons retain workload, timing semantics, rays, rates, and batch
size, and measure warmed synchronized acquisition, full updates, and peak memory.
New training and evaluation follow [ADR-0026](0026-training-benchmark-release-separation.md).

## Sources

- [Scene](../../src/drone_playground/simulation/scene.py), [observation](../../src/drone_playground/simulation/observation.py), [sensors](../../src/drone_playground/simulation/sensors.py).
- [MuJoCo-LiDAR core](https://github.com/discoverse-dev/MuJoCo-LiDAR/blob/main/src/mujoco_lidar/core_jax/mjlidar_jax.py), originally inspected Git blob `8ee01fa104b52d855162b48e9bfdae68b2226a6e`.
- [Wrapper](https://github.com/discoverse-dev/MuJoCo-LiDAR/blob/main/src/mujoco_lidar/lidar_wrapper.py), originally inspected blob `7089fceb2aa3769bdc214558873947732c7671df`, and [scan generator](https://github.com/discoverse-dev/MuJoCo-LiDAR/blob/main/src/mujoco_lidar/scan_gen.py).
