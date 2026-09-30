# 操作手册

所有命令从项目根目录执行。当前工作位置为 `simulation_dev/drone_playground`，代码、依赖缓存和正式产物各自拥有明确目录。

## 环境准备

```bash
python3 scripts/tools/fetch_sources.py
pixi install --locked
pixi run python -c "import jax; print(jax.devices())"
```

固定源码缓存位于 `tmp/sources/`，其中包含 Crazyflow 和 LOTF。源码提交与补丁由 `third_party/sources.yaml` 记录，数值包版本由 `pixi.lock` 固定。依赖校验使用临时Git索引比较“固定提交＋声明补丁”，包含补丁新增文件；不会重置缓存工作树或改写其暂存区。迁移项目目录后重新执行锁定安装，使解释器、可编辑包及脚本入口指向新路径。

`pixi run`会在Python启动前设置`SCIPY_ARRAY_API=1`。直接调用锁定解释器时也设置该变量，避免SciPy先导入后造成JAX姿态计算的Tracer转换错误。

本地源码导出使用已确认的提交：

```bash
python3 scripts/tools/export_source.py /tmp/drone-playground-source.tar.gz --revision HEAD
```

目标文件必须不存在。导出仅包含该提交的文件，不包含Git历史、未提交修改、未跟踪文件或忽略的实验产物；同一提交的重复导出摘要相同。包内`SOURCE_MANIFEST.json`记录来源提交、文件类型、执行位及SHA-256。解压后按上面的固定环境命令安装。无Git历史的运行保留`code.commit=null`，通过`manifest.json`中的`code.archive`记录来源、实际修改／删除／新增文件及`source-snapshot.tar.gz`摘要。实际源码快照包含本地改动，遵循项目忽略规则，实验输出不进入快照；符号链接按链接保存，不读取其外部目标。该清单用于内容追溯，不是下载来源的数字签名。源码导出不代表18格质量达标或远端发布。

## 训练与恢复

```bash
pixi run train method=learning/ppo env=hovering --cfg job
pixi run train method=learning/ppo env=hovering runtime.device=gpu run_id=ppo-hover-new
pixi run train method=learning/bptt env=tracking runtime.device=gpu run_id=bptt-tracking-new
pixi run train method=learning/shac env=racing runtime.device=gpu run_id=shac-racing-new
```

正式训练采用 GPU，并按显存与预算排队。先核对已有 `state.json`、`result.json` 和进程身份，再确定新运行或恢复。`training.warm_start` 表示参数热启动；具备完整状态恢复能力的训练器使用 `training.resume`。恢复时保持所记录的模型、网络、优化器及输入合同。

新运行自动写入`experiments/tmp/YYMMDD/<run_id>/`，日期为UTC启动日期。读取状态可用`pixi run status --run-id=<标识>`，无需手动查找日期。旧报告和历史命令中的`experiments/<run_id>/...`引用由现役读取接口兼容定位，原始记录不改写。

在 CUDA 可用的主机上，可用 `JAX_PLATFORMS=cuda,cpu` 将 GPU 设为 JAX 默认后端，同时允许代码显式使用 CPU；项目的 `runtime.device` 也应选择 `gpu`。例如：

```bash
JAX_PLATFORMS=cuda,cpu pixi run train method=learning/ppo env=tracking \
  runtime.device=gpu run_id=ppo-tracking-new
```

该变量按顺序初始化后端，不表示 CUDA 不可用时自动回退；无 CUDA 时使用 `JAX_PLATFORMS=cpu` 和 `runtime.device=cpu`。批量仿真与训练可受益于 GPU；文件、协议及纯 Python 几何检查不会因该变量提速，小计算还可能受编译和传输开销影响。详见 [JAX 平台配置](https://docs.jax.dev/en/latest/config_options.html#platforms)。

```bash
# 从已完成的正式参数启动另一组独立试验。
pixi run train method=learning/bptt env=tracking \
  training.warm_start=experiments/final-acceptance-bptt-tracking-t0/checkpoints/step-0000655360.pkl \
  training.num_envs=16 training.policy_updates=1024 algorithm.horizon_length=40 \
  runtime.device=gpu run_id=bptt-tracking-warm-start
```

检查点恢复的具体配置兼容性由训练器核对。BPTT 的完整恢复检查原目标预算与行为配置，适用于原预算内的未完成阶段；已完成训练可通过参数热启动开展新试验。正式基线的完整训练命令保存在[历史命令索引](verification/final-acceptance/commands.md)；重新运行时使用独立标识和适当预算。

## 冻结评测与回放

```bash
pixi run eval \
  checkpoint=experiments/final-acceptance-bptt-tracking-t0/checkpoints/step-0000655360.pkl \
  runtime.device=gpu evaluation.split=heldout evaluation.episodes=32 \
  run_id=recheck-bptt-tracking

pixi run play replay=experiments/tmp/260928/final-acceptance-heldout-bptt-tracking-v1/rollouts
pixi run play replay=experiments/tmp/260928/final-acceptance-heldout-bptt-tracking-v1/rollouts \
  visualization=headless
```

RScope 回放使用同目录的轨迹、场景和元数据。查看已有轨迹直接选择 `replay`；冻结策略重新执行选择 `checkpoint`。每个回合保留真实活动区间和最终终止状态。

## 原生规划器与 MPC

```bash
# 初次准备或明确重建 ROS 容器时执行。
bash native_planners/setup.sh

pixi run eval method=paper/super env=navigation/static \
  evaluation=navigation_v2 evaluation.episodes=2 runtime.device=cpu \
  run_id=super-static-new
```

原生规划器安装脚本会重建项目命名的 ROS 容器；执行前确认既有规划任务已结束。现有容器独立于 Python 工作目录，具体镜像、提交和补丁见 `native_planners/versions.env`、`native_planners/patches/` 及[原生集成说明](../native_planners/README.md)。

新增原生评测在每回合的`native/.../decision-trace/`记录适配器→下游执行器边界：当前机体状态、实际收到的完整物理输出、执行参考、有效期和控制器生成的命令。相同Trajectory／Waypoint／Motion Cmd按内容摘要共用存储，`index.json`记录两个压缩文件的摘要、命令字段和SI单位；中断时也保留已收到的决策。

原生与`pipeline`的悬停／跟踪／竞速评测使用`evaluation.seed_start`作为首个重置种子，显式的0也有效；留空时开发集从20000、留出集从30000开始。新的`eval/report.json`逐回合保存实际初始位置、速度、xyzw姿态，以及启用相应延迟模型时的`delay_requested_ms`和`delay_effective_ms`。旧版曾忽略自定义首种子，回归及已完成报告的影响检查见[重置合同凭据](verification/native-control-reset-contract.json)。

```python
from drone_playground.evaluation.decision_archive import load_native_decisions

for frame in load_native_decisions("experiments/<run>/native/hard/0/decision-trace"):
    print(frame["tick"], frame["time"], frame["reply"].get("output"), frame["command"])
```

读取会验证摘要并恢复三类输出对象，可在相同控制器配置下，从回合初态顺序重放执行参考，不必再次调用异步规划器。该记录位于物理转移之前；`command=null`表示控制器未返回命令。即使命令已生成，仍需按tick与物理轨迹中对应的转移配对，才能确认已执行。导航的转移记录在同回合`case-trace/`，控制任务的轨迹在运行目录`rollouts/`。记录开销可能影响原生异步调度，不能据此承诺再次调用规划器会得到同一路径。

```bash
# 完整 acados 数值测试所需的局部依赖。
bash scripts/tools/setup_acados.sh
JAX_PLATFORMS=cpu pixi run test
```

acados v0.5.1 可通过 `ACADOS_SOURCE_DIR` 指向已经验证的构建。默认局部位置为 `tmp/p3p4/optimization/acados`。该目录与 `tmp/sources/` 是实际依赖缓存，清理时按依赖处理。

## 场景维护

现役场景直接从 `assets/scenes/navigation/catalog.json` 加载。确需重建六张 SANDO 来源主场景时，显式提供固定来源的 worlds 目录：

```bash
python3 scripts/tools/build_navigation.py --sando-worlds /path/to/pinned-sando/worlds \
  --output tmp/navigation-catalog.json
cmp assets/scenes/navigation/catalog.json tmp/navigation-catalog.json
```

脚本先校验来源 world 文件摘要，再重建主场景，并从现役目录保留 S06／D06 两张固定3D扩展。它不依赖历史场景 v1–v4，也不从本机目录结构猜测依赖位置。改变几何须另立协议与校验记录。

## 维护检查

本地第一版18格结果直接打开`experiments/main_result/v1-18-cells/README.md`。中间结果在`experiments/tmp/<日期>/`。每格的日期／种子目录包含报告、配置、选定权重和回放入口；`qualification.json`区分正式质量、开发结果及用户接受的例外。更新本地结果视图：

```bash
pixi run python scripts/tools/organize_experiments.py \
  --pointcloud-run primary-pointcloud-short32-seed0-t0-20260930 --apply
```

不带`--apply`仅查看迁移计划。该工具拒绝移动活跃运行；冻结工作树只链接，报告与权重不改写。目录组织与验收状态由[结果凭据](verification/experiment-layout.json)记录。

```bash
pixi run lint
JAX_PLATFORMS=cpu pixi run test
python3 scripts/tools/summarize_final_acceptance.py \
  --selection docs/verification/final-acceptance/selection.json \
  --output tmp/final-acceptance-check.json
```

最后一条命令读取本地正式结果包，核对30个单元、训练来源、参数和回放摘要。源码克隆自身包含公开证据索引；完整产物检查需要对应结果包。原点云任务已保存45000次完整状态并停止，见[冻结工作树生命周期](research/branch-lifecycle.md)。
