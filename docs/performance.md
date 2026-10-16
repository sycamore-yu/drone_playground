# Navigation throughput evidence

## 2026-10-10：Sensor / render 迁移后的 RTX 4090 实测

`refactor/sensor-rendering` 采用 MuJoCo-LiDAR 非重复角度表及瞬时帧采集。
GPU 恢复后，`JAX_PLATFORMS=cuda` 使用 RTX 4090（驱动 595.84、JAX 0.11.2），
在独立 GPU 进程上执行完整 APG 更新。比较的原版是 `v0.2@6ec50c7` 的源码快照，
两者均保持 Crazyflow 500 Hz、控制 10 Hz、相同射线数和 batch × horizon；
不比较不同观察模型的数值损失是否相同。

| 工作负载 | v0.2 中位 s/update | 新版中位 s/update | 完整更新吞吐提升 |
|---|---:|---:|---:|
| LiDAR / D01，32×32，20,000 rays | 0.81135 | 0.73041 | 11.1% |
| Depth / S01，128×32，64×48@30Hz | 0.83697 | 0.77691 | 7.7% |

每次排除第一次 JAX 编译更新，记录三次稳定更新并同步全部输出。
`nvidia-smi` 在每次更新之后采样的显存占用分别为 LiDAR 2,685→2,693 MiB、
Depth 1,159→1,161 MiB，不等同于峰值显存。
以上为**局部最高精度几何矩阵变换修复后的最终实测**。此前的
`0.71510 s / 0.76167 s` 是修复前的中间结果，不能与最终精度合同混用；
精度修复由官方 `mujoco.Renderer` 的 GPU 深度对照触发并通过专项测试。
六种 Depth/LiDAR × PPO/APG/SHAC 均完成 GPU 参数更新；
冻结 Depth 全八场景 196/200 成功、LiDAR 175/200 成功，
LiDAR S03 的 0/25 与原传感器上的相同权重复测一致。
数值原始记录、性能对照范围、源文件版本、失败门禁和复现命令见
[验证记录](validation.md)。

这是**整体传感器模型迁移**的性能结果。MuJoCo-LiDAR 0.3.5 JAX 求交未通过
几何正确性门禁，运行时仍为原有 `Scene.raycast()`；
MJX-Warp Batch Renderer 没有安装或进行本项目基准测试。
不同库的选型与扫描时间保真局限，见
[raycasting fidelity review](research/sensor-raycasting-fidelity-review.md)。


以下历史测量采用逐射线时间与合成 MID360 扫描，
不是本次快照模型的 GPU 性能结果。

## v0.2：并行环境数与验收开销优化（2026-10-10）

### MID360 策略射线优化（旧扫描模型实验，2026-10-10）

MuJoCo 的 `mj_ray`/`mj_multiRay` 只计算**请求的射线**，没有强制每帧
20,000 射线的引擎协议。20,000 是 Livox MID360 所公布的典型硬件吞吐
`200,000 points/s ÷ 10 Hz`；本平台用非重复合成角度序列模拟该扫描，
**不是**官方扫描器驱动或厂商真实非重复扫描轨迹重建。

旧版实验每个 0.1 秒完整生成 20,000 条射线、在 JAX 中逐点求交及按时间插值，
然后确定性抽取 1,024 个索引交给 PointNet/GRU。因此对只有 1,024 点输入的
训练，约 94.88% 的射线计算不会进入网络。

本次选定**显式、只针对新配方的稀疏策略射线模式**：
`navigation_lidar_scratch_sparse.yaml` 和
`navigation_lidar_scratch_ppo_asymmetric_sparse.yaml` 设置
`sensor.ray_sampling=policy`。内部保留真实设备规格
`points_per_frame=20000`，直接根据完整扫描的均匀选点索引只计算
`point_count=1024` 条射线；**相位、每点采集时间、FOV、
量程、deskew、运动障碍求交和有效掩码不变**。
`ray_sampling=full` 保留原生 20,000 点全帧场景测试及 RScope 回放。

这是一项**改变训练传感器取得的原始点云尺寸**的配方选择，不就地更改
旧训练 run，也不得在同一个 checkpoint 恢复时混换新旧采样模式。
CPU JAX 对移动障碍、机体运动条件下的独立结果进行了专项对照：
策略所见 64 条稀疏射线与先计算完整 257 条再抽取的结果在位置/量程上
不超过约 1 毫米的 float32/XLA 差异，方向误差不超过 2e-5，
时间戳/掩码一致。训练实际使用 1024/20000；上述专项试验不是其
RTX4090 稳态加速倍率。GPU 正被正式训练占用，未执行新的独占吞吐实验。

优化策略优先级：先减掉不进入策略的射线计算，再在 GPU 空闲时
根据同一训练配方比较 `num_envs` 的**稳态 interactions/s**
和时间到收敛，而不是人为追求 24 GB 显存全部占满。
MuJoCo Playground 的 JAX PPO 也允许减少中间评测和指标回调频率。

已批准 [ADR-0015](adr/0015-three-stage-learning-acceptance.md)：训练期间使用
每主场景 8 回合轻量筛查，正式候选检查保留各主场景 25 回合，冻结
benchmark 还完整报告 S06/D06。此前每轮全八场景、每场景 25 回合的反复
筛查已不再是新 run 的默认协议。该调整减少中途评测的计算量，但仍保留
最终三种子 C2/C3/C4 原始阈值。

GPU 性能方面保留 `XLA_PYTHON_CLIENT_PREALLOCATE=false`，**显存占用不是
吞吐优化目标**。并发运行过程中 RTX 4090 经常报告约 100% 的 GPU 活跃时间，
但不能据此推断 CUDA 核心达到理论峰值。下面原有的同步更新性能测量
已表明 Crazyflow 500 Hz、多帧传感器和可微展开存在很大成本。

本次新增专用入口 `tools/benchmark_training_throughput.py`，执行的是真实
`Environment + Trainer.update`，不是只测物理空转；在 JIT 预热后
使用 `jax.block_until_ready` 同步计时，记录每秒真实交互数、
每次更新墙钟耗时、GPU 显存与功率。正式训练进程仍在运行时，
**该入口会拒绝新 GPU 测试**，避免通过资源争抢得到无效对照数据。

```bash
# 显示计划，不占用 GPU
python tools/benchmark_training_throughput.py --dry-run

# 只在正式矩阵 GPU 训练暂时完全空闲时运行，不中断现有作业
python tools/benchmark_training_throughput.py \
  --experiment navigation_lidar_scratch --algorithm apg \
  --worlds 32,64,128,256 --warmup 2 --updates 8 \
  --output results/performance/lidar-apg-world-count.json

python tools/benchmark_training_throughput.py \
  --experiment navigation_depth_scratch_safe --algorithm shac \
  --worlds 64,128,256,512 --warmup 2 --updates 8 \
  --output results/performance/depth-shac-world-count.json
```

不提前认定 batch256 比 batch64 更快。增加 `num_envs` 同时改变每次更新的样本数；
**需要同时报告 updates/s、interactions/s、显存与达到 L3 成功率门槛所需总时间**。
跨算法或不同传感器的吞吐不可直接作为同算法扩 batch 的因果结论。
参考 [JAX 官方计时建议](https://docs.jax.dev/en/latest/benchmarking.html)、
[JAX GPU 显存分配](https://docs.jax.dev/en/latest/gpu_memory_allocation.html)、
[Isaac Lab 训练性能建议](https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/training_guide.html)。

---

## 旧版诊断（2026-10-08）

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

## L4 efficiency follow-through

For each of the 18 cells, continued training must record
actual interaction/update counts, compile and steady-state wall time, GPU memory,
competing workloads, checkpoint evaluation results and compute to L3 success.
Exercise all mixed-scene blocks, including asynchronous resets, before generalizing
these short comparisons. Resume existing checkpoints only after loading the new
source in a fresh process; running jobs do not reload Python source.

The local benchmark uses its own persistent compilation-cache directory. Any
shared cache or packaging changes are recorded separately; caching does not improve
the measured steady-state update itself.
