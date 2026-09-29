# 操作手册

所有命令从项目根目录执行。当前工作位置为 `simulation_dev/drone_playground`，代码、依赖缓存和正式产物各自拥有明确目录。

## 环境准备

```bash
python3 scripts/tools/fetch_sources.py
pixi install --locked
pixi run python -c "import jax; print(jax.devices())"
```

固定源码缓存位于 `tmp/sources/`，其中包含 Crazyflow 和 LOTF。源码提交与补丁由 `third_party/sources.yaml` 记录，数值包版本由 `pixi.lock` 固定。迁移项目目录后重新执行锁定安装，使解释器、可编辑包及脚本入口指向新路径。

## 训练与恢复

```bash
pixi run train method=learning/ppo env=hovering --cfg job
pixi run train method=learning/ppo env=hovering runtime.device=gpu run_id=ppo-hover-new
pixi run train method=learning/bptt env=tracking runtime.device=gpu run_id=bptt-tracking-new
pixi run train method=learning/shac env=racing runtime.device=gpu run_id=shac-racing-new
```

正式训练采用 GPU，并按显存与预算排队。先核对已有 `state.json`、`result.json` 和进程身份，再确定新运行或恢复。`training.warm_start` 表示参数热启动；具备完整状态恢复能力的训练器使用 `training.resume`。恢复时保持所记录的模型、网络、优化器及输入合同。

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

pixi run play replay=experiments/final-acceptance-heldout-bptt-tracking-v1/rollouts
pixi run play replay=experiments/final-acceptance-heldout-bptt-tracking-v1/rollouts \
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

```bash
# 完整 acados 数值测试所需的局部依赖。
bash scripts/tools/setup_acados.sh
JAX_PLATFORMS=cpu pixi run test
```

acados v0.5.1 可通过 `ACADOS_SOURCE_DIR` 指向已经验证的构建。默认局部位置为 `tmp/p3p4/optimization/acados`。该目录与 `tmp/sources/` 是实际依赖缓存，清理时按依赖处理。

## 维护检查

```bash
pixi run lint
JAX_PLATFORMS=cpu pixi run test
python3 scripts/tools/summarize_final_acceptance.py \
  --selection docs/verification/final-acceptance/selection.json \
  --output tmp/final-acceptance-check.json
```

最后一条命令读取本地正式结果包，核对30个单元、训练来源、参数和回放摘要。源码克隆自身包含公开证据索引；完整产物检查需要对应结果包。原点云50000次更新运行按[冻结工作树生命周期](research/branch-lifecycle.md)处理。
