# ADR-0030: Racing optimization baselines and browser playback

**Status:** Accepted, 2026-10-16. **Implementation:** Separate Racing MPC feature
branch. The final training-rollout recording is implemented on the main line.

## Racing methods

Develop the Racing optimization baselines on `feature/racing-mpc-baselines`,
starting from the released Learning / Simulation / Benchmark interface.

- [UZH RPG `rpg_mpc`](https://github.com/uzh-rpg/rpg_mpc): the IROS 2018
  perception-aware MPC code is based on ACADO/ROS, not acados contouring MPC.
  Integrate it as a distinct perception-aware controller baseline, not as MPCC.
- [UZH RPG `rpg_time_optimal`](https://github.com/uzh-rpg/rpg_time_optimal):
  the Science Robotics 2021 implementation solves an offline time-optimal
  waypoint trajectory planning problem. Its output is a trajectory reference
  that needs a separate flight controller; it is not a feedback MPC solver.
- [LSY Racing MPCC](https://github.com/lihengrongfei/lsy_drone_racing_ws26):
  use its acados arc-length path, lag/contour costs and progression variables as
  the direct MPCC architecture reference. Implementation must be native to the
  current controller/setpoint conventions and existing Crazyflow flight model.

Keep solver provenance, numerical approximations and upstream license terms
explicit. Do not copy GPL or course-project source into this repository unless
its redistribution requirements are accepted. Compare real closed-loop Racing
performance using identical initial states, gate order, speed and solver-time
measurements. Do not label simplified trajectory tracking as full MPCC.

## Completed-run playback

Persist a short sample of the **actual final training rollout**, independent
of whether the user had the Dashboard open. Record MuJoCo JPEG frames and a
browser-readable animated WebP for each Autoresearch training run. The sampling
is limited to the final update's recorded one-world rollout fragment; it is
not an independently simulated evaluation episode or the full training history.
Existing older runs cannot be retroactively reconstructed without an additional
policy rollout, which must be labeled as evaluation/replay rather than training.

For remote SSH users, keep the Dashboard static-asset HTTP server as the transport.
Use the browser link to inspect saved media after training has exited. The
current feature exports raster media. Evaluate an optional interactive viewer
using [official MuJoCo WebAssembly bindings](https://github.com/google-deepmind/mujoco/tree/main/wasm)
and the [mujoco_wasm](https://github.com/zalo/mujoco_wasm) demo. A WebAssembly
viewer needs a portable MJCF/asset bundle plus recorded `qpos` and dynamic-body
poses; running its simulator forward is not equivalent to replaying a trained
policy's recorded states. It is a separate optional UI extension, not a reason
to replace RScope or the evaluation executor.
