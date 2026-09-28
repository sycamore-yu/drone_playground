# 版本 3 操作手册

从项目根目录运行。首次执行 `python3 scripts/tools/fetch_sources.py`，随后 `pixi install --locked`。固定缓存存在时重复执行会核对提交和补丁。

## 入口与配置

```bash
pixi run train method=learning/ppo env=racing
pixi run eval method=paper/super env=navigation/static evaluation=navigation_v1
pixi run play checkpoint=experiments/<运行>/checkpoints/<权重>.pkl
pixi run play replay=experiments/<运行>/rollouts visualization=headless
```

主任务环境：`hovering`、`tracking`、`tracking/random`、`racing`、`navigation/static`、`navigation/dynamic`。

具名论文环境：`paper/lotf_hover`、`paper/lotf_tracking`、`paper/pointcloud_flight` 和 `paper/pointcloud_navigation`。

只解析配置可执行 `pixi run python scripts/train.py method=learning/ppo env=racing --cfg job`。组件组重新选择使用挂载路径，例如 `sensor@env.sensor=d435 observation@env.observation=navigation_depth`。

## 小预算工程检查

```bash
pixi run train method=learning/ppo env=hovering runtime.device=cpu \
  run_id=ppo-hover-engineering-check training.num_envs=8 \
  training.num_timesteps=256 training.num_evals=2 training.development_episodes=2 \
  algorithm.batch_size=8 algorithm.num_minibatches=1 \
  algorithm.unroll_length=16 algorithm.num_updates_per_batch=2
```

该预算测试训练管线。正式策略训练沿任务配方指定预算和独立留出集完成。每次新运行使用独立 `run_id`。

## 点云训练与旧状态迁入

```bash
pixi run python scripts/tools/migrate_artifact.py <可信旧状态.pkl> <新目录>
pixi run train method=paper/pointcloud_flight training.resume=<迁入状态.pkl>
pixi run eval checkpoint=<已选状态.pkl> env=paper/pointcloud_navigation \
  evaluation.training_run=<对应训练运行目录>
```

完整恢复必须保持训练预算、网络、目标、场景和算法的行为配置。重组验收用原状态的相同小预算配置证明续训，完整 50000 更新训练仍在原点云工作树中。`scripts/tools/run_pointcloud_pipeline.py` 为当前配置生成对应训练/评测阶段命令，`summarize_pointcloud.py` 默认按完整预算检查，阶段汇总需显式 `--allow-stage`。

## 原生规划器和 MPC

```bash
bash native_planners/setup.sh
pixi run eval method=paper/ego_planner env=navigation/static \
  runtime.device=cpu evaluation.episodes=1 method.port=55201
pixi run eval method=paper/super env=navigation/dynamic \
  runtime.device=cpu evaluation.episodes=1 method.port=55211
bash scripts/tools/setup_acados.sh
pixi run eval method=optimization/attitude_mpc env=racing runtime.device=cpu
pixi run eval method=optimization/sampling_mpc env=racing \
  runtime.device=cpu method.decision.prediction_device=gpu
```

并发原生运行使用不同端口；`evaluation.episodes` 在 navigation 中表示每个难度的回合数。已有 acados 构建可以通过 `ACADOS_SOURCE_DIR` 指定，构建产物保存在当前运行自身目录。物理执行与预测设备分别记录。

## 回放与状态

`play` 生成实际闭环轨迹后进入已有 RScope 流程；`visualization=headless` 保留导出及路径核对，并保持当前查看器选择。已有轨迹使用 `replay=...`，数据源只读。

```bash
pixi run status --runs-root experiments
pixi run python -m pytest tests/test_architecture_v3.py -q
pixi run test
```

日志和验证摘要在 `docs/verification/architecture-v3/`，每个原始运行保留完整配置、进程、依赖、补丁、指标、事件和回放。

## 冻结权重的局部执行修改

```bash
pixi run eval checkpoint=<当前版本权重.pkl> runtime.device=cpu \
  env.execution.dynamics.forward=so_rpy
```

该覆盖沿用权重保存的任务、场景、机型、输入和时序，只替换指定动力学字段。使用 `dynamics@env.execution.dynamics=crazyflow` 会替换整个动力学预设；使用 `env=...` 会替换完整环境，随后检查冻结输入契约。每次变化进入新的运行记录。
