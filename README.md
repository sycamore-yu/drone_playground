# Drone Playground

面向无人机学习、规划和控制的可组合仿真实验平台。通过一份方法配方与一份完整环境预设，运行训练、独立评测及实际轨迹展示。当前配置版本为 3。

平台采用 JAX 原生实现优先的路线，复用锁定版本的 Crazyflow、Brax 和 LOTF；原生 EGO-Planner、SUPER 与 acados 通过明确适配进入同一任务环境。控制研究重点是轨迹跟踪层，机体状态由仿真提供，传感器提供深度或点云测量。

## 当前可运行能力

PPO 的公开配方覆盖悬停、轨迹跟踪、竞速、静态导航和动态导航。APG、SHAC、D.VA、LOTF 与点云论文具名训练器保留各自参数更新语义。两种现有 MPC 与 EGO/SUPER 支持冻结条件下的独立评测及展示。

本次重组的短训练证明参数更新、保存、重载和闭环执行。完整训练预算和任务表现单独验收，逐项结果见 [架构重组验证](docs/verification/architecture-v3/README.md)。LOONG、AERO-MPPI、AC-MPC 的身份记录在来源清单中，其完整 JAX 实现属于后续具名任务。

## 环境与方法

```text
方法：允许的信息 → 神经策略 / 在线优化 / 具名组合 → 轨迹或控制命令
环境：场景 + 任务 + 传感测量 + 观测表示 + 命令执行与动力学
训练：环境交互或可微展开 → 训练目标 → 参数更新 → 开发集选模
评测：固定条件和试次 → 共享闭环 → 原始事件、全分母统计、回放
```

场景文件只定义外部几何及运动。目标、重置、到达、过门、碰撞终止与时限属于任务。`env` 是这些组件的组合入口，学习与优化方法均使用完整环境。训练目标由独立 `objective` 配置选择；当前 Brax 任务状态保留奖励字段，以便沿用既有训练和评测数值语义。

## 安装

需要 Linux、Git、Pixi。原生规划器另需 Docker，GPU 训练另需匹配的 NVIDIA 驱动。

```bash
# 在项目根目录执行，锁定源码进入项目自管缓存。
python3 scripts/tools/fetch_sources.py
pixi install --locked
```

源码来源及提交见 `third_party/sources.yaml`，依赖解析见 `pixi.lock`。首次源码获取脚本只使用 Python 标准库；清单采用兼容 YAML 的 JSON 表示。LOTF 所需机体和场景资源随固定源码缓存保留。

完整测试包含实际 acados 求解。先执行 `bash scripts/tools/setup_acados.sh`，或设置已有构建的 `ACADOS_SOURCE_DIR`，随后执行 `pixi run test`。

## 训练

```bash
pixi run train method=learning/ppo env=hovering
pixi run train method=learning/ppo env=tracking
pixi run train method=learning/ppo env=racing
pixi run train method=learning/ppo env=navigation/static
pixi run train method=learning/ppo env=navigation/dynamic

# Learning on the Fly 与点云论文分别选择。
pixi run train method=paper/lotf env=paper/lotf_hover
pixi run train method=paper/pointcloud_flight
```

CPU 验证使用 `runtime.device=cpu`；预算使用 `training.num_timesteps` 或具名算法的 `training.policy_updates`。PPO 新入口的各任务短训练命令和实际参数增量在验证目录中逐项记录。

## 评测和展示

```bash
pixi run eval method=paper/super env=navigation/static evaluation=navigation_v1
pixi run eval method=paper/ego_planner env=navigation/dynamic evaluation=navigation_v1
pixi run eval method=optimization/attitude_mpc env=racing
pixi run eval method=optimization/sampling_mpc env=racing

# 实际检查点文件由其旁边的元数据恢复方法、网络和输入契约。
pixi run eval checkpoint=experiments/<运行标识>/checkpoints/<检查点>.pkl
pixi run play checkpoint=experiments/<运行标识>/checkpoints/<检查点>.pkl
pixi run play method=paper/super env=navigation/static
pixi run play replay=experiments/<运行标识>/rollouts
```

`play` 在线运行与正式评测共享执行层，结束后发布实际回放包到 RScope；现有回放模式只读取轨迹产物。服务器验证可选 `visualization=headless`，生成与核对回放且保持当前查看器选择。优化配方的训练入口在创建运行目录前检查训练能力。

EGO/SUPER 运行环境由 `native_planners/` 自管：`bash native_planners/setup.sh`。acados 构建使用 `bash scripts/tools/setup_acados.sh`，已有固定构建可通过 `ACADOS_SOURCE_DIR` 指定。

## 高级组合

```bash
# 组件的实际所有者可直接覆盖。
pixi run train method=learning/ppo env=tracking env.execution.dynamics.forward=first_principles

# Hydra 配置组重选使用其挂载路径。
pixi run train method=learning/ppo env=navigation/static \
  sensor@env.sensor=d435 observation@env.observation=navigation_depth

# 显式两控制步命令延迟；默认零附加延迟。
pixi run train method=learning/ppo env=hovering runtime.action_delay_steps=2
```

`dynamics` 产生实际状态；方法内部 `prediction` 提供未来预测；`algorithm.gradient` 选择直接或具名代理导数。动力学实现和算法配置各自拥有唯一来源。组合约束在启动时检查，真实支持范围由方法配方与验证证据共同界定。

## 数据、协议与来源

`assets/scenes/navigation/catalog.json` 保存已验收的八个几何场景，公开名称为 navigation；静态和动态视图引用同一资产。`benchmarks/navigation/v1/` 保存协议、划分和几何验收证据。历史场景编号及资产摘要保持可核对。

每次运行独占 `experiments/<运行标识>/`，保存解析配置、提交和补丁、依赖、实际进程命令、状态、检查点、评测和回放。历史版本检查点通过显式复制迁移：

```bash
pixi run python scripts/tools/migrate_artifact.py <可信旧检查点.pkl> <新目标目录>
```

迁移检查源摘要，重定位序列化类型路径，保留数组和优化器状态，并拒绝覆盖目标。Pickle 产物只应来自可信的本地实验。

## 导航与边界

完整实际目录见 [项目目录](docs/project-tree.md)，模块职责见 [架构](docs/architecture.md)，运行说明见 [操作手册](docs/runbook.md)，当前任务见 [状态](docs/status.md)。许可与上游差异见 [第三方声明](THIRD_PARTY_NOTICES.md)。

继承的原生规划器参数仍采用 2 m/s 速度上限；96 m 的导航位移与 40 s 时限在该速度约束下需要另行校准。架构回归保留这组条件及其超时结果，后续配方调参须生成新的实验身份。
