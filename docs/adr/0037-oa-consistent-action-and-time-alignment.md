# ADR-0037: Egocentric OA defaults and temporally aligned sensor snapshots

**Status:** Implemented in public training configurations, **not yet independently
accepted for all OA combinations**. Changes to the OA task distribution and
sensor preprocessing are training-contract changes; old checkpoints are never
silently relabeled or counted as validation for the new configuration.

## Train action-frame policy

- **All 28 OA Train combinations** share the common OA training package. For
  PPO and native/PointMassLag BPTT/SHAC, the Actor emits the three-axis **net
  body-frame acceleration**, not force, thrust, body-rate or IMU specific force.
  The established `Action.setpoint` converts it with the orientation at command
  issuance to the world-frame *net* acceleration expected by the real Mellinger
  outer loop. For delayed control, the model's input is the same issued world
  acceleration; no per-tick reinterpretation or duplicate gravity compensation.
- The LOTF backward model requires **body_rate** (collective thrust in newtons,
  angular velocity in body rad/s). Its valid contract remains `frame=body`. Do
  **not** mislabel body-rate as acceleration just to unify a YAML field.
- The OA PPO recipe no longer requests the body-to-world rotation matrix in the
  Actor observation. The body-frame goal, velocity, upright gravity and mounted
  Depth/LiDAR measurements remain consistent.
- Tracking and Racing use world-frame reference errors and explicit orientation,
  so their world-frame acceleration / native attitude-thrust conventions remain
  valid. Keep all 14 historical accepted single-seed checkpoints intact.
- The old Navigation `velocity_yaw` recipe saw egocentric navigation inputs but
  compared its velocity output directly to the *world* velocity. Its public
  Navigation recipe now declares `frame=body`. The execution layer rotates the
  requested velocity at issuance to world before the unchanged velocity PID.
  Under `frame=body`, the fourth command is a **yaw offset relative to the
  current vehicle heading**. `Action.setpoint` computes
  `world_yaw_desired = wrap(world_yaw_at_issue + local_yaw_offset)` while rotating
  the velocity vector into the world PID frame. Its checkpoint identity includes
  `yaw_reference=current_heading_offset`. This resolves the earlier ambiguity
  where an egocentric Actor was required to output an unobservable absolute
  heading. Under legacy `frame=world`, yaw remains an absolute world angle and
  historical checkpoint identity is unchanged. Previously trained body-velocity
  checkpoints with absolute world yaw have a different contract and cannot be
  silently loaded as the repaired relative-yaw policy.

## Training distribution: overhead crossbeams

Original matched scenes contained side openings in tall walls, but lacked long
horizontal crossbeams that require climbing over or ducking underneath. Fixed
S06 has two such beams at about x=18 and x=82 m, z=2.2–3.2 m. Under the earlier
body-acceleration SHAC model, 22 of 25 S06 development cases hit those beams.

The existing variable beam generator (source `0bbab7d`) is now available to
both static and dynamic **matched** procedural OA training: `horizontal_beam_probability=0.7`.
Its beam positions, z-centres, spans and thicknesses vary by world/episode. Beams
are *static obstacles* even in dynamic scenes; the existing moving obstacles
remain independently sampled. Reserved start/goal clearance, feasible height
passages, and frozen benchmark XML/geometry hashes remain unchanged. This
changes the training distribution, not any held-out acceptance standard.

Earlier body-frame SHAC with no beams achieved 78/100 development (S06 3/25);
with randomly varied crossbeam training the next candidate achieved 100/100
**development** on the same four 25-case scenes. That result motivated making
this source of training coverage available in the defaults; it is not 100-case
independent release evidence and says nothing yet about other algorithms or
seeds. New default configurations must be validated from random initialization.

## Sensor acquisition time vs control time

An image or point cloud is captured at t_acq with body pose
`(p_acq, R_world_from_body(t_acq))`, and may only be delivered later. The Actor's
velocity/goal are evaluated at t_now. If a point `x_acq_body` is passed unchanged
as `x_now_body`, the Actor sees incompatible coordinates as the robot rotates or
translates. Even with configured latency=0, a low-rate LiDAR snapshot can be
older than the current policy/control tick.

For new OA defaults, `observation.{depth,lidar}.motion_compensate=true`:

```
x_now_body = R_now.T @ (p_acq + R_acq @ x_acq_body - p_now)
```

The raw measurement, validity mask, acquisition timestamp and capture pose stay
intact. LiDAR returns compensated current-body points, leaving invalid points
zero. Depth projects acquired 3-D return points into the current optical camera
using declared intrinsics and extrinsics and a nearest-depth z-buffer. Occluded
pixels and newly visible surfaces are **invalid**, never invented by silently
rerendering the present world. When capture and current poses already agree,
conversion is bypassed to avoid overhead for synchronous 10 Hz OA Depth.
Output includes `sensor_age_s` alongside the original acquisition time. When a
motion-compensated sensor is **slower than the policy** or has nonzero transport
latency, the environment now also appends one scalar to the Actor's state:
`clip(sensor_age_s * sensor_frequency_hz, 0, 10)` (age in sensor periods).
Thus the Actor can distinguish recent and stale perceptual returns. At a fully
synchronous, zero-latency capture/control boundary, no state dimension is added
and accepted input dimensions remain stable. This does not grant access to
undelivered measurements or infer the current location of a moving obstacle.

**Limits:** This compensates ego-motion of a single ideal timestamped *snapshot*;
it is not rolling-shutter correction, not per-ray LiDAR de-skewing, and not
prediction of obstacles that moved after capture. Per-point calibration and
acquisition timestamps are absent from the snapshot model. A realistic rolling
MID360 needs per-ray time and IMU/interpolated poses to compensate individual
rays, as in the FAST-LIO implementation. Benchmarking nonzero sensor latency or
fast-moving obstacles may require a declared temporal model/metadata feature
rather than pretending a stale obstacle is at its current position.

Primary reference: [FAST-LIO IMU deskewing implementation](https://github.com/hku-mars/FAST_LIO/blob/main/src/IMU_Processing.hpp).
NavRL's public [ROS navigation control](https://github.com/Zhefan-Xu/NavRL/blob/main/ros2/navigation_runner/scripts/navigation.py)
explicitly transforms commands to their final robot control coordinate frame.
P2M and DiffAero supply additional baselines for learning architectures but are
not proof of any particular policy frame convention.

## Acceptance and performance policy

- Hydra tests must resolve all **28 OA method combinations** to explicit body
  frame; assert Tracking/Racing training contracts remain identical.
- Unit tests cover nonzero yaw, exact analytic rotation, LiDAR pose transport,
  depth reprojection/masks, no fabricated hits, unchanged acquisition values,
  recorded age, zero-latency fast path, static/dynamic random beams and unchanged
  S06 geometry. Verify the `velocity_yaw` 90-degree frame conversion.
- Regression checks must cover original six-primary task behavior, checkpoint
  frame metadata and the existing fixed 42 Train recipe definitions. Train a
  minimal native BPTT/SHAC/PPO update from random parameters on the new action
  and observation contract before beginning longer release runs.
- Old OA experimental checkpoints/seed evidence with `frame=world`, the old
  training distribution or uncorrected sensor observations are **Imported Prior**,
  never evidence that the new defaults have converged.
- Keep current single GPU queue until **measured completed-candidate throughput**
  improves with concurrency. Existing six CPU Depth jobs were hundreds of
  seconds/update, versus ~0.5–0.7 s/update for matched GPU SHAC; CPU-only is
  suitable for low-cost tests/diagnostics, not simultaneous OA full training.
  If GPU utilisation is genuinely low during a prolonged phase, profile the
  benefit of scheduling auxiliary CPU evaluation or increasing GPU worker count
  in an isolated controlled experiment before changing the queue.

See ADR-0036 for the original action-frame diagnosis. Independently confirm
held-out releases after a successful development run; continue Autoresearch
without a cumulative candidate/GPU-hour search limit as authorized by the user.
