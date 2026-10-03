# 操作手册

开发命令从项目根目录执行。安装后的 `drone-playground` 可在其他目录运行；`runtime.output_root` 指定产物根目录，默认当前目录。配置、资产和基准从安装包读取。

## 环境准备

```bash
python3 scripts/tools/setup.py sources
pixi install --locked
pixi run python -c "import jax; print(jax.devices())"
```

固定源码缓存位于 `tmp/sources/`，其中包含 Crazyflow、LOTF 和 Brax。源码提交与补丁由 `patches/sources.json` 声明，`scripts/tools/setup.py` 负责准备和校验缓存，数值包版本由 `pixi.lock` 固定。依赖校验使用临时Git索引比较“固定提交＋声明补丁”，包含补丁新增文件；不会重置缓存工作树或改写其暂存区。迁移项目目录后重新执行锁定安装，使解释器、可编辑包及脚本入口指向新路径。

`pixi run`会在Python启动前设置`SCIPY_ARRAY_API=1`。直接调用锁定解释器时也设置该变量，避免SciPy先导入后造成JAX姿态计算的Tracer转换错误。

本地源码导出使用已确认的提交：

```bash
python3 scripts/tools/export_source.py /tmp/drone-playground-source.tar.gz --revision HEAD
```

目标文件必须不存在。导出仅包含该提交的文件，不包含Git历史、未提交修改、未跟踪文件或忽略的实验产物；同一提交的重复导出摘要相同。包内`SOURCE_MANIFEST.json`记录来源提交、文件类型、执行位及SHA-256。解压后按上面的固定环境命令安装。无Git历史的运行保留`code.commit=null`，通过`manifest.json`中的`code.archive`记录来源、实际修改／删除／新增文件及`source-snapshot.tar.gz`摘要。实际源码快照包含本地改动，遵循项目忽略规则，实验输出不进入快照；符号链接按链接保存，不读取其外部目标。该清单用于内容追溯，不是下载来源的数字签名。源码导出不代表18格质量达标或远端发布。

## 训练与恢复

```bash
pixi run train experiment=control/ppo env=hovering --cfg job
pixi run train experiment=control/ppo env=hovering runtime.device=gpu run_id=ppo-hover-new
pixi run train experiment=control/bptt env=tracking runtime.device=gpu run_id=bptt-tracking-new
pixi run train experiment=control/shac env=racing runtime.device=gpu run_id=shac-racing-new
```

正式训练采用 GPU，并按显存与预算排队。先核对已有 `state.json`、`result.json` 和进程身份，再确定新运行或恢复。`training.warm_start` 表示参数热启动；具备完整状态恢复能力的训练器使用 `training.resume`。恢复时保持所记录的模型、网络、优化器及输入合同。

单独创建环境不需要训练配方：

```python
import jax
import drone_playground

env = drone_playground.load("hovering", overrides=["runtime.device=cpu"])
try:
    state = env.reset(jax.random.key(0))
    state = env.step(state, env.hover_action)
finally:
    env.close()
```

`load()` 直接调用 Hydra。`algorithm` 选择训练更新规则；训练所得 Policy 与运行方法分开。没有 registry。

新运行自动写入`results/runs/<task>/<method>/<run_id>/`。Task／Method 来自解析配置，默认 `run_id` 为UTC时间戳加训练 seed；读取状态可用`pixi run status --run-id=<标识>`。检查点和评测报告属于该 run，选定结果由`results/selected/`引用，不再复制成另一棵目录。

在 CUDA 可用的主机上，可用 `JAX_PLATFORMS=cuda,cpu` 将 GPU 设为 JAX 默认后端，同时允许代码显式使用 CPU；项目的 `runtime.device` 也应选择 `gpu`。例如：

```bash
JAX_PLATFORMS=cuda,cpu pixi run train experiment=control/ppo env=tracking \
  runtime.device=gpu run_id=ppo-tracking-new
```

该变量按顺序初始化后端，不表示 CUDA 不可用时自动回退；无 CUDA 时使用 `JAX_PLATFORMS=cpu` 和 `runtime.device=cpu`。批量仿真与训练可受益于 GPU；文件、协议及纯 Python 几何检查不会因该变量提速，小计算还可能受编译和传输开销影响。详见 [JAX 平台配置](https://docs.jax.dev/en/latest/config_options.html#platforms)。

```bash
# 从已完成的正式参数启动另一组独立试验。
pixi run train experiment=control/bptt env=tracking \
  training.warm_start=results/runs/tracking/bptt/<source-run>/checkpoints/<checkpoint>.pkl \
  training.num_envs=16 training.policy_updates=1024 algorithm.horizon_length=40 \
  runtime.device=gpu run_id=bptt-tracking-warm-start
```

检查点恢复的具体配置兼容性由训练器核对。BPTT 的完整恢复检查原目标预算与行为配置，适用于原预算内的未完成阶段；已完成训练可通过参数热启动开展新试验。正式基线的完整训练命令保存在[历史命令索引](../artifacts/verification/final-acceptance/commands.md)；重新运行时使用独立标识和适当预算。

## 随机化与动力学组合

随机化属于训练配置。以下命令展示参数位置，实际范围须匹配选定模型：

```bash
pixi run train experiment=control/bptt env=hovering \
  training.domain_randomization.enabled=true \
  '+training.observation_noise.position_std_m=0.02' \
  '+training.reset_randomization.orientation_half_width_rad=[0.1,0.1,0.2]' \
  '+training.command_distribution={kind:position,distribution:uniform,low:[-1,-1,1],high:[1,1,2]}' \
  runtime.device=gpu run_id=randomized-hover

# LOTF 是 dynamics source；任务、网络和训练器保持通用。
pixi run train experiment=control/bptt env=tracking \
  dynamics@env.dynamics=lotf_high_fidelity \
  controller@env.controller=rates \
  algorithm.gradient.transition=simplified_dynamics_jacobian \
  runtime.device=gpu run_id=lotf-tracking
```

将 Dynamics preset 改为 `lotf_simplified` 即使用简化动力学；前向模型与 `algorithm.gradient.transition` 的反向规则独立选择。Navigation 组合使用 `env.task.physics_freq=1000`，并在自定义评测下选择适合的协议；标准 Navigation8 的模型条件仍由原协议规定。模型拒绝没有物理作用的参数：拟合姿态与 LOTF simplified 不接受惯量 DR，PointMassLag 只接受 `motor_strength`／`lag` 倍率。外力、力矩使用 N／Nm，点质量使用加速度 m/s²；测量噪声和动作误差单独声明。

初态分布由对应 experiment 的 `training.reset_randomization` 声明。`training.scene_distribution.type` 声明 fixed／generated／procedural，固定库和固定 command range 不叫 curriculum。

## 冻结评测与回放

```bash
pixi run eval \
  checkpoint=results/runs/tracking/bptt/<source-run>/checkpoints/<checkpoint>.pkl \
  runtime.device=gpu evaluation.episodes=32 evaluation.record_replays=true \
  run_id=recheck-bptt-tracking

pixi run play replay=results/runs/tracking/bptt/recheck-bptt-tracking/rollouts
pixi run play replay=results/runs/tracking/bptt/recheck-bptt-tracking/rollouts \
  visualization.publish=false
```

数值报告和选模记录始终保存；普通 train/eval 的完整 RScope/MuJoCo replay 默认关闭，设置 `evaluation.record_replays=true` 才写入 `rollouts/`。Brax 训练显式启用 `training.publish_live=true` 时也会记录回放供实时查看。`play checkpoint=...` 自动记录本次执行轨迹；查看已有轨迹直接选择 `replay`。正式 Benchmark 由对应 `benchmarks/` specification 固定 cases、种子、预算与指标，不通过额外 Environment role 区分。

### 历史检查点

当前运行只接受配置 v4。历史 v3 权重、sidecar 和正式报告保持不变。先准备经过核对的完整 v4 解析配置，再显式创建新副本：

```bash
pixi run python -m drone_playground.artifacts.migration \
  results/runs/<task>/<method>/<old-run>/checkpoints/<old>.pkl \
  tmp/migrated/<new>.pkl --config tmp/reviewed-v4-config.json
```

工具核对输入、网络和物理合同，保留来源配置及参数摘要。目标必须不存在。普通 Brax 推理权重保持字节不变；旧循环状态的类路径只在该显式工具内转换。含旧环境结构的 BPTT/SHAC 完整状态不能直接当作新架构续训状态，应选择其推理检查点做参数热启动。新架构产生的完整状态使用正常 `training.resume`。

## 原生规划器与 MPC

```bash
# 初次准备或明确重建 ROS 容器时执行。
bash ros_integrations/ros1/setup.sh

pixi run eval experiment=papers/super env=navigation/static \
  +evaluation.protocol=benchmarks/navigation.yaml evaluation.episodes=2 runtime.device=cpu \
  run_id=super-static-new
```

原生规划器安装脚本会重建项目命名的 ROS 容器；执行前确认既有规划任务已结束。现有容器独立于 Python 工作目录，具体镜像、提交和补丁见 `ros_integrations/ros1/versions.env`、`ros_integrations/ros1/patches/` 及[原生集成说明](../ros_integrations/ros1/README.md)。

新增原生评测在每回合的`native/.../decision-trace/`记录适配器→下游执行器边界：当前机体状态、实际物理输出、执行参考、有效期和控制命令。Reference、Setpoint、SFC 和轨迹预览按内容摘要保存，走廊及预览各自保留有效期。RPC 使用 v2；旧外部服务需用 v2 SDK 重建，历史冻结运行包不改写。

原生与`pipeline`的悬停／跟踪／竞速评测使用`evaluation.seed_start`作为首个重置种子，显式的0也有效；留空时训练内 `checkpoint_eval` 从20000、最终 `benchmark` 从30000开始。新的`eval/report.json`逐回合保存实际初始位置、速度、xyzw姿态，以及启用相应延迟模型时的`delay_requested_ms`和`delay_effective_ms`。旧版曾忽略自定义首种子，回归及已完成报告的影响检查见[重置合同凭据](../artifacts/verification/native-control-reset-contract.json)。

```python
from drone_playground.artifacts.decisions import load_native_decisions

for frame in load_native_decisions("results/runs/<task>/<method>/<run_id>/native/hard/0/decision-trace"):
    print(frame["tick"], frame["time"], frame["reply"].get("output"), frame["command"])
```

读取会验证摘要并恢复三类输出对象，可在相同控制器配置下，从回合初态顺序重放执行参考，不必再次调用异步规划器。该记录位于物理转移之前；`command=null`表示控制器未返回命令。即使命令已生成，仍需按tick与物理轨迹中对应的转移配对，才能确认已执行。导航的转移记录在同回合`case-trace/`，控制任务的轨迹在运行目录`rollouts/`。记录开销可能影响原生异步调度，不能据此承诺再次调用规划器会得到同一路径。

```bash
# 完整 acados 数值测试所需的局部依赖。
pixi run setup-acados
JAX_PLATFORMS=cpu pixi run test
```

acados v0.5.1 可通过 `ACADOS_SOURCE_DIR` 指向已经验证的构建。默认局部位置为 `tmp/p3p4/optimization/acados`。该目录与 `tmp/sources/` 是实际依赖缓存，清理时按依赖处理。

## 场景维护

现役场景从安装资源 `assets/scenes/navigation/catalog.xml` 及对应场景 MJCF 加载。确需重建六张 SANDO 来源主场景时，显式提供固定来源的 worlds 目录和新的候选输出目录：

```bash
python3 scripts/tools/build_navigation.py --sando-worlds /path/to/pinned-sando/worlds \
  --output tmp/navigation-candidate
```

脚本保留原生成数学，先核对源 world 摘要，再生成候选 MJCF，并从现役资产保留 S06／D06。候选输出不覆盖正式资产。修改几何后需要新的协议与校验记录。当前导航 XML 的摘要及原目录身份在 `benchmarks/navigation-mjcf-verification.json`。

## 维护检查

本地第一版18格选择直接查看`results/selected/v1-18-cells.json`；真实执行位于`results/runs/<task>/<method>/<run_id>/`，预览、诊断和迁移历史位于`results/scratch/`。更新选择视图：

```bash
pixi run python scripts/tools/organize_experiments.py --apply
```

Selection 只引用已有 run、report 和 checkpoint，不复制权重或回放。迁移前复制式结果包保留在`results/scratch/legacy/main_result/`作历史证据；清理 run 前先检查`selected/`引用。

```bash
pixi run lint
JAX_PLATFORMS=cpu pixi run test
```

历史 final-acceptance 汇总器及对应一次性实验脚本已归档到 `tmp/retired-experiments/`，不再作为现役运行入口。源码克隆自身包含公开证据索引；完整产物检查需要对应结果包。原点云任务已保存45000次完整状态并停止，见[冻结工作树生命周期](notes/research/branch-lifecycle.md)。

## 回放显示

新导航回放自动显示实际深度／MID360视场；原生方法和模块链在保存真实输出时还显示规划轨迹及可选SFC。配套XML与mj_unroll必须一起保留。历史文件可复制增强，操作和数据合同见[回放可视化](notes/research/replay-visualization.md)。工程预览统一放在`results/scratch/previews/`或`results/scratch/replays/`；它们不增加正式质量通过数。
