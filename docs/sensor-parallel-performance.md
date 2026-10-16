# Procedural sensor throughput

Baseline: `a7e9157` (`main`). Candidate: `perf/sensor-parallel-throughput`.

## Method

Benchmarks ran on the same RTX 4090, with native Crazyflow first-principles dynamics, procedural static obstacle avoidance, 208 obstacle slots, horizon 32, and native BPTT. Each process used a fresh JAX/CUDA runtime with `XLA_PYTHON_CLIENT_PREALLOCATE=false`. We synchronized complete `Trainer.update()` calls using `jax.block_until_ready`. The original baseline used two warm-ups and five timed updates; the other runs used two warm-ups and six timed updates. JIT compilation is excluded from the step/s figures. Each environment contributes 32 policy steps per update.

The baseline source in `results/convergence/20261016T110804Z-iter1000/source/new_drone_playground` was checked byte-for-byte against `a7e9157` for `bptt.py`, `trainer.py`, `environment.py`, `sensors.py`, `procedural_scene.py`, and `observation.py`.

## Full GPU update throughput

| Depth | Worlds | Original step/s | Candidate step/s | Candidate GPU memory |
| --- | ---: | ---: | ---: | ---: |
| 30 Hz, 64×48 | 128 | 2,426 | 6,133 | 1,531 MiB |
| 30 Hz, 64×48 | 256 | 2,519 | 4,214 | 2,039 MiB |
| 30 Hz, 64×48 | 512 | — | 3,397 | 3,589 MiB |
| 10 Hz, 64×48 | 128 | — | 7,322 | 1,351 MiB |
| 10 Hz, 64×48 | 256 | 5,476 | 9,030 | 1,943 MiB |
| 10 Hz, 64×48 | 512 | — | 9,655 | 3,393 MiB |
| 10 Hz, 64×48 | 1,024 | — | 15,088 | 5,631 MiB |
| 10 Hz, 64×48 | 2,048 | — | 17,081 | 13,131 MiB |

At 30 Hz, 128 worlds has the best measured candidate throughput. At 10 Hz, 2,048 worlds yields the best throughput among the measured sizes, but uses considerably more device memory and makes fewer optimizer updates for the same interaction budget. Training quality at 10 Hz requires a matched-budget comparison before changing the default sensor frequency. The existing 30 Hz configurations remain unchanged; `oa_static_depth_10hz` and `oa_dynamic_depth_10hz` are opt-in experiments.

## Per-change GPU ablation

The next results all use Depth at 30 Hz, 128 worlds, and the same native BPTT configuration.

| Changes enabled | Steps/s | Ratio against baseline |
| --- | ---: | ---: |
| Original | 2,426 | 1.00× |
| Selective depth reset only | 2,450 | 1.01× |
| Raycast geometry optimization only | 5,859 | 2.42× |
| Both changes | 6,133 | 2.53× |

The raycast change provides most of the full-training gain. The masked reset adds about 4.7% over the raycast-only case for this six-update probe. This is a short-run estimate; reset frequency varies with episode length and batch size.

## CPU geometry and reset tests

These tests ran with JAX CPU, 8 fixed CPU cores, two warm-ups and five timed repetitions. They are separate from the full-GPU results.

| Operation | Inputs | Original | Candidate | Gain |
| --- | --- | ---: | ---: | ---: |
| Depth raycast | 8 worlds, 208 slots, 3,072 rays/world | 177.2 ms | 91.9 ms | 1.93× |
| MID-360 raycast | 8 worlds, 208 slots, 1,024 rays/world, 40 m | 45.09 ms | 25.54 ms | 1.77× |
| Partial depth reset | 128 worlds, 8 restarting, 208 slots, 64×48 | 2,357.3 ms | 140.1 ms | 16.8× |

The raycast checksums agree exactly between original and candidate implementations. The masked-reset figure includes both geometry pruning and selective sensor capture; the GPU ablation isolates the effect of selective capture on complete training updates.

## BPTT rematerialization

A small CPU experiment found a 7.7% reduction in update time without `jax.checkpoint(step)`. The RTX 4090 comparison for 128 worlds and 10 Hz depth gave the opposite result:

| BPTT | Update throughput | GPU memory |
| --- | ---: | ---: |
| `jax.checkpoint(step)` enabled | 7,322 step/s | 1,351 MiB |
| Disabled | 5,831 step/s | 1,931 MiB |

Disabling rematerialization reduces GPU throughput by 20.4% and raises memory use by about 43%. The branch retains the original BPTT implementation and adds no configuration switch.

## Correctness and merge decision

The suite covers exact first-return ray geometry, moving obstacles, broad-phase overflow, camera and LiDAR observation contracts, a partially filled final gather chunk, selective reset with real depth hits, and deterministic checkpoint continuation. The recorded runs include 37 procedural/performance checks, 67 sensor/navigation checks, six PPO/BPTT/SHAC perception-update checks, and the additional 35-world indexing test. The final version repeats the dedicated performance tests after the last ray-selection change.

- **Merge the raycast optimization** after normal review. It produces the largest measured GPU gain and preserves sensor results in the tested scenes.
- **Keep the selective reset on this branch for review.** It is numerically verified and accelerates the reset operation, but contributes a smaller end-to-end benefit at 128 worlds. The branch preserves its exact-scan fallback.
- **Merge the opt-in 10 Hz presets** if faster sensor-frequency experiments are useful. Keep the 30 Hz default until trained-policy results justify a change.
- **Do not change BPTT rematerialization.** The original version is faster and uses less GPU memory in the target measurement.

A concurrent formal GPU training process prevented additional end-to-end LiDAR profiling. LiDAR geometry was checked on CPU and the sensor training/restore tests passed. No changes are made to `main` by this experiment.

Steady-update benchmarking can be repeated from the repository root with:

```bash
pixi run python tools/benchmark_training_throughput.py \
  --experiment oa_static_depth --algorithm bptt \
  --worlds 128,256 --warmup 2 --updates 8 \
  --output results/performance/sensor_parallel.json
```

The per-run raw data and independent one-off probes reside in this worktree's ignored `tmp/perf-study/` folder.
