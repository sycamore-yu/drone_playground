# Navigation throughput evidence

Historical measurements below use the pre-snapshot sensor model. The
`refactor/sensor-rendering` branch changes scan directions and frame acquisition;
its correctness evidence and unresolved GPU gates are in [validation.md](validation.md).
The timings below are not speed measurements of that branch.

This record concerns bounded Navigation APG diagnostics on the RTX 4090 on
2026-10-08. It does **not** complete C6 across the 18 training cells or establish
time to convergence. The authoritative acceptance criteria remain in [spec.md](spec.md).

## Workload and measurement contract

- Physics: official Crazyflow first-principles, 500 Hz; method: 10 Hz; horizon: 32.
- Depth: batch 128, the existing D435i `training64x48` acquisition mode at 30 Hz,
  followed by the existing 12×16 policy preprocessing. This is not a 1280×720 run.
- LiDAR: batch 32, all 20,000 Mid-360 returns per frame at 10 Hz, with individual
  ray times, interpolated positions/quaternions, directions, masks and deskewing;
  the existing actor preprocessing still selects 1,024 points.
- The captured mixed-run configs include random starts, temporal gradient decay,
  altitude and perception-loss overrides. Sensor acquisition remains detached;
  physical-state, recurrent-policy and cost derivatives retain their existing roles.
- One diagnostic GPU process at a time, preallocation disabled. Timed calls block
  on the complete output tree. Compilation, warmup, transfers for assertions,
  `nvidia-smi` sampling and profiler overhead are excluded from update timings.
- Candidates use the same initialized state and alternate execution order.
  Every timing records GPU utilization, power, memory and competing process IDs.
  These are measurements under concurrency, not exclusive-GPU speed estimates.

The running depth/LiDAR APG jobs, EGO/SUPER rollouts, checkpoint evaluations and a
new LiDAR transfer-training job shared the GPU during this work. Racing ended during the first baseline
measurement. Historical update-10 logs (5.77 s depth, 4.29 s LiDAR) are therefore
not valid denominators for this change's speedup.

## Bottleneck evidence

The complete batch-32, horizon-32 LiDAR baseline trace contains 538,485 GPU events,
including 139,890 `memcpy32_post` events. This workload has many small kernels,
copies and nested control loops; GPU utilization alone is not a measure of useful
training throughput. The profiler itself materially increases wall time, so its
durations are used for attribution only.

The compiled backward HLO contains a 50-iteration physics loop under
`transpose(jvp())/checkpoint/rematted_computation/jit(_step)`. The raycast chunk
kernel executes 2,496 times, matching 32 captures × 78 complete 8,192-ray chunks
(plus each capture's remainder). Its HLO is in the forward acquisition branch.
This does not support the initial hypothesis that backward repeats all sensor
acquisition; physics recomputation and full-frame carry operations are evidenced.

`SensorObservation.update` selected every full-frame field after its capture
conditional even when no world was due. The baseline's three-array
`loop_select_fusion.35` operates on `[32,20000,3]` arrays 1,600 times per update.
Moving the per-world selections **inside the existing capture branch** preserves
mixed-world masks, while the no-capture branch directly returns the prior arrays.
Pose history and frame-clock processing still run every physical tick.

The final D01 traces compare the same state after four updates. GPU event count
drops from 536,721 to 517,777 (18,944 fewer events, 3.5%). Two unconditional
full-frame selection fusions that each ran 1,600 times disappear; the remaining
frame selections run inside capture branches. The raycast chunk kernel still
runs 2,496 times. Event counts support the optimization independently of changing
concurrent workloads; summed event durations are not critical-path wall time.

## Retained change and measured throughput

Only [`simulation/observation.py`](../src/drone_playground/simulation/observation.py)
changed. No sensor resolution, point count, ray timing, physics rate, loss,
derivative rule, reset contract, public interface or checkpoint field changed.
APG and Scene remain byte-identical to their initial snapshots.

Each row has four synchronized before/after pairs with alternating order:

| Device / scene | Batch × horizon | Original median s/update | Changed median s/update | Interactions/s before → after | Throughput gain |
| --- | --- | ---: | ---: | ---: | ---: |
| LiDAR / S01, fixed input state | 32 × 32 | 2.336130 | 2.126131 | 438.3 → 481.6 | 9.9% |
| Depth / S01, successive shared states | 128 × 32 | 1.294299 | 1.108782 | 3164.6 → 3694.1 | 16.7% |
| LiDAR / D01, successive shared states | 32 × 32 | 3.517481 | 3.408211 | 291.1 → 300.5 | 3.2% |

All 12 pairs favored the changed version. These are short, concurrent workload
comparisons, not an isolated or all-scene sustained speedup. In particular, the
Depth pairs have appreciable timing spread (1.156–1.511 s original,
1.062–1.200 s changed). Do not extrapolate them to a universal 16.7% gain.

Compiled temporary memory stays about 552.31 MB for LiDAR and changes from
108.29 to 109.86 MB for Depth. Process VRAM includes JAX's allocator and multiple
compiled variants; it is recorded separately in each event log.

## Candidate results

| Candidate | Numerical result | Resource/performance result | Decision |
| --- | --- | --- | --- |
| Move full-frame selections into capture | LiDAR S01 and D01 bitwise identical; Depth maximum parameter difference 2.98e-8, exact physical/sensor state; successive-state comparisons pass | Measured gains above | Implemented |
| Whole-step checkpoint with `prevent_cse=False` | Parameter mismatch 1.80e-5 exceeds `atol=rtol=1e-5` gate | Compiled temporary memory about 552 MB | Rejected |
| Remove whole-step checkpoint | Parameter mismatch 1.23e-5 already fails gate | Compiled temporary memory 7.96 GB vs 0.552 GB baseline; diagnostic process reached 10,202 MiB in `nvidia-smi` | Rejected |
| Checkpoint actor only | Parameter, physical-state, GRU and gradient-norm comparisons fail gate | Median 2.649 s vs 2.336 s baseline; temporary memory 444 MB | Rejected |

The checkpoint alternatives preserve the mathematical expression but change
compiled floating-point evaluation enough to fail the requested numerical gate.
They are benchmark-only copies, not changes to APG. JAX documents that CSE
prevention can be unnecessary inside a scan, but this application's measured
numerical outcome governs adoption: [JAX checkpoint API](https://docs.jax.dev/en/latest/_autosummary/jax.checkpoint.html).

The full-sensor CPU branch check passed 120 physical ticks for each device in
moving scene D01, with different world clocks, an inactive world, translated and
rotated poses, exact per-leaf equality, noninteger 30 Hz capture timing and zero
sensor-acquisition derivatives.

Both device comparisons include every final `Measurement` field, pose history,
deskewed point array, physical/task state, recurrent and objective history,
optimizer state, parameters and metrics (including unclipped actor gradient
norm). Float comparisons use `atol=rtol=1e-5`; integers and masks must be exact.
The harness also uncovered a typed-PRNG comparison error on its first candidate
run; the corrected harness compares raw key bits and the affected run was rerun.

**Final source validation:** 51 selected existing CPU tests passed in 481.92 s.
They cover complete scene/sensor tests (including full acquisition and moving
geometry), observation timing, masked resets, all PPO/APG/SHAC perception updates
with checkpoint continuation and inference export, SHAC terminal-value state
gradients, and temporal physics-gradient decay. Existing CPU learning fixtures
use reduced LiDAR acquisition; the GPU rollout comparisons and the separate CPU
branch check above use the full configured 20,000-return acquisition. Ruff passed
for the modified source and the three diagnostic scripts.

The initial pytest invocation failed before collection because an ambient ROS
plugin required `lark`. The successful invocation isolates project tests with
`PYTHONPATH=` and `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`; no dependencies were changed.

## Reproduction and artifacts

The harness, captured configs, original source snapshots, candidate source,
compiled HLO, memory/cost analyses, timings, concurrency records and JAX trace are
under [`tmp/agents/navigation-performance/`](../tmp/agents/navigation-performance/).

```bash
env XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda MUJOCO_GL=egl \
  OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
  JAX_COMPILATION_CACHE_DIR=tmp/agents/navigation-performance/cache \
  .pixi/envs/default/bin/python tmp/agents/navigation-performance/compare.py \
  lidar --variants baseline current --rounds 4 --advance --out lidar-repeat

env JAX_PLATFORMS=cpu MUJOCO_GL=egl OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
  .pixi/envs/default/bin/python tmp/agents/navigation-performance/check_sensor_branch.py

env PYTHONPATH= PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 JAX_PLATFORMS=cpu MUJOCO_GL=egl \
  OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .pixi/envs/default/bin/pytest -q \
  -o cache_dir=tmp/agents/navigation-performance/pytest-cache \
  --basetemp=tmp/agents/navigation-performance/pytest-tmp \
  tests/test_environment.py::test_sensor_clock_and_masked_sensor_history_reset \
  tests/test_scene_sensors.py \
  tests/test_learning.py::test_perception_real_update_and_complete_sensor_checkpoint \
  tests/test_learning.py::test_shac_terminal_value_state_gradient \
  tests/test_learning.py::test_temporal_decay_affects_only_incoming_physics_derivative \
  tests/test_navigation_sampling.py::test_masked_random_reset_preserves_inactive_world_and_sensor
```

Keep the existing source snapshots when reproducing: they define the baseline.
`--advance` compares successive real updates from a shared carried state;
without it, repetitions use a fixed state to hold work constant. Use a fresh
`--out` directory for each invocation. Trace overhead is excluded from timings.
`observation.patch` provides the small reviewable source diff despite the initial
workspace's untracked source files. `source-sha256.json` identifies the final
supporting source and lockfile; the earlier runs' candidate source is retained.

## C6 follow-through

For each of the 18 cells, continued training must record
actual interaction/update counts, compile and steady-state wall time, GPU memory,
competing workloads, checkpoint evaluation results and compute to C1–C5 success.
Exercise all mixed-scene blocks, including asynchronous resets, before generalizing
these short comparisons. Resume existing checkpoints only after loading the new
source in a fresh process; running jobs do not reload Python source.

The local benchmark uses its own persistent compilation-cache directory. Any
shared cache or packaging changes are recorded separately; caching does not improve
the measured steady-state update itself.
