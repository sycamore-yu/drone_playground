# ADR-0013: Ideal depth generation and explicit camera configuration

**Status:** Accepted.
**Approval recorded:** 2026-10-10T08:28:44Z.
**Created:** 2026-10-10T15:17:33+08:00.
**Implementation:** Explicit camera settings and official Renderer reference tests are present. Depth and LiDAR use one ray-query path; no Warp renderer was added.
**Original evidence baseline:** `v0.2@6ec50c7a441bbe169fef40acd34ae574b53428a9`, reviewed 2026-10-10T10:30:23Z.
**Last reviewed:** 2026-10-16T05:28:58Z, working tree based on `7d59162`.

## Decision

Generate ideal depth and LiDAR through `Scene.raycast()`. Camera pixel-center rays,
intrinsics, extrinsics, optical axis, and image dimensions remain camera-specific.
Convert radial distance to optical-axis depth using the ray's optical-axis
component, and apply range/validity masks. Determine misses before conversion so
rounding cannot turn a far-clip miss into a valid return.

Use `mujoco.Renderer` and `mj_ray` as test references with aligned pixel centers,
depth definitions, visibility, and clipping. Do not require bitwise equality
between different rasterization methods. MuJoCo-LiDAR and MJX-JAX replacements
remain subject to [ADR-0012](0012-sensor.md). No renderer registry or second
environment is introduced.

## Camera settings and source attribution

`64x48` is the project's training resolution, not a D435i hardware mode. The
removed `training64x48` constructor alias is not restored. The sensor YAML declares
width, height, 30 Hz acquisition, 20 degree pitch, and a 10 m cutoff. Preprocessing
belongs to the observation configuration.

| Source | Image and preprocessing | Acquisition |
|---|---|---|
| Zhang / DiffPhysDrone `271936190b5c` | 64x48, then 4x4 max pooling to 16x12 | Normally distributed timestep around 1/15 s |
| DiffAero `291ea14196aefbebcf7387dd71f7e096c83878b7` | Default 16x9 | Environment dt 0.0333 s |
| Project baseline | 64x48, pooled to 16x12 | Sensor 30 Hz; OA method 10 Hz |

Zhang informs preprocessing; the 30 Hz choice is the project's sensor setting.
Paper-faithful timing requires a separate explicit recipe. This decision did not
change CNN architecture, pooling dimensions, or the learning algorithm.

## Replay and geometry scope

The 2026-10-11 replay update reused licensed Crazyflie 2.x STL appearance assets,
while recorded poses and the existing free joint drive playback. Measured point
clouds are replayed without recomputing intersections. Plans can appear as timed
segments. These visual meshes do not add a sensor mesh requirement. See
[replay](../replay.md).

## Evidence and future replacement

The original record reported about 7.7% higher full-update throughput after a
local precision correction and 196/200 frozen depth outcomes. Those are historical
engineering results, not convergence of the current release matrix.

A Warp replacement requires evidence that acquisition is a material bottleneck
and improves complete updates under the same resolution, rate, rays, time model,
and batch size. Measure warm synchronized latency, memory, graph-capture overhead,
and actual jit/scan/checkpoint/grad calls. Official renderer throughput figures do
not predict this project's speedup. After a candidate passes the geometry and task
checks, remove the replaced kernel; retain distance functions, preprocessing, and
human-facing replay in their existing roles.
