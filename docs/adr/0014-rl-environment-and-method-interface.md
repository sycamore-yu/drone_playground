# ADR-0014：Brax 环境合同与可训练 Method 解耦

**Status:** Accepted · 2026-10-09

**Implementation:** 独立分支 `feat/brax-rl-decoupling` 已实现可配置 Actor/Critic/Method、独立 EpisodeObjective、Method 动作概率接口、具体环境子类及可选 BraxTrainingEnv。Brax 原生 PPO、APG 均通过短程接口验证。完整质量矩阵仍需独立验收。Brax State 由适配器持有；`EnvState` 是内部单一真实物理状态。

## Context

当前 Simulation 使用 Crazyflow/JAX、显式 `EnvState` 与独立的 `observe/cost`；Learning 的 PPO、APG、SHAC 已共用环境，但 `Trainer` 仍固定构造具体 Actor/Critic，且包含 Navigation 与具名感知损失。旧仓库直接继承 Brax `Env` 并调用原生 PPO/APG，也额外维护了各算法的训练及恢复入口。

参考：Brax `brax.envs.base.Env/State`、MuJoCo Playground `MjxEnv` 和 `wrap_for_brax_training`、Isaac Lab `DirectRLEnv/ManagerBasedRLEnv`。它们共同使用环境交互合同，将网络与参数更新留在学习算法内部；各项目的物理、批量、任务组织方式不同。

## Decision

1. **算法独立于具体网络、动作与运行方法。** 构造阶段选择网络、Critic、可训练 Method、控制输入与损失。训练算法依赖传入的策略计算／环境执行操作，不能据 `state/depth/lidar`、Task 名称或输出维数决定网络和动作含义。动作维度与单位由选中的 Action 合同决定；网络创建和循环记忆初始化归具体 Actor；Critic 的输入权限与 Actor 独立。PPO 保留必要的动作分布和 `log_prob`，APG/SHAC 保留可微前向计算链。
2. **一个 JAX 物理环境核心，对齐 Brax 交互合同。** 保留 `reset(rng)` 和 `step(state, action)` 的纯函数行为；用于 Brax 的 State 包含 `pipeline_state`、`obs`、`reward`、`done`、`metrics`、`info`，提供 `observation_size`、`action_size`、`backend`。保留原有 Crazyflow SimData、Task、传感器与延迟状态的真实结构，不在运行时构造第二套同义物理真值。
3. **批量边界只存在一处。** Crazyflow 已使用 `Sim(n_worlds=num_envs)`。Brax 上游默认 `VmapWrapper` 只能应用于真正单世界环境；适配已批量的环境时不再重复 `vmap`。显式验证随机键形状、单世界／批量世界和只重置完成世界。
4. **保留实验语义。** 随机起点与物理扰动由 reset 生成；`terminated`、`truncated` 区分并保留终止前末态观测；训练与冻结评测使用独立初态／场景／随机种子；训练状态保存参数、优化器、RNG、当前环境与方法所需历史。对旧 checkpoint 的结构变更采用显式迁移或独立新实验，不默默重新解释。
5. **明确奖励与辅助损失的归属。** Task/Environment 负责任务奖励、终止规则及物理评价；与 Zhang/Liu 等特定网络／方法相关的辅助预测项及其历史属于 Learning Objective。训练算法只调用已构造好的目标函数并更新网络参数。保留已经冻结的奖励数值和系数，改变定义时新建实验配方。
6. **Brax PPO/APG 为可选对照，不替换现有实现。** 实验入口直接调用选定版本的 `brax.training.agents.ppo/apg`，消费同一环境合同；验证环境真的产生训练更新及独立回合数据。其包装、日志、选模与保存功能仅作为该算法入口的必要适配，不增加一套持久的通用 Runner／Trainer 生命周期。当前项目 PPO/APG/SHAC 继续作为主实现。
7. **原生 Planner 独立于可微路径。** EGO/SUPER 继续宿主执行，共用 Task、轨迹、控制器和评测。L2O-SANDO 的 Network → 可微 QP → Controller → Crazyflow 属于 Method 内部前向和反向数学；通用环境不解析时间因子、走廊或 QP 私有变量。
8. **具体任务环境类继承共享物理基类。** `BaseEnv` 执行 Crazyflow 物理、传感器、时钟、延迟和动作输入；`TrackingEnv`、`RacingEnv`、`NavigationEnv` 分别定义观测、奖励与任务成功／失败规则。保留 `TaskState` 的回合计数与事件表示，以及现有 `Task` 中的参考、边界及配置元数据，避免重复实现轨迹生成和场景参数。动态构造通过唯一的 `make_environment(task=...)` 入口选择具体类，不创建另一份环境管理器或注册表。

## Scope and acceptance

实施时分两组独立验收：

- **原生执行一致性：** 原来的 PPO/APG/SHAC、State/Depth/LiDAR 组合继续训练；改变网络不修改 Task，改变动作接口不修改网络编码器；物理转移、训练损失、终止／截断、恢复与冻结评测按相同配方回归。
- **Brax 互通：** 指定依赖版本后，用原生 Brax PPO 在真实 Crazyflow 环境完成非零参数更新，保留样本数、随机化与终止统计。必须使用实际 Brax 训练调用验证，不以 `isinstance(Env)` 或单步测试代替训练验收。APG 再进行独立对照。

2026-10-09 实现记录：`learning/brax_adapter.py` 按 `brax.envs.base.Env/State` 接入已批量运行的 Crazyflow。原生 PPO 通过 `wrap_env=False` 避免重复 `VmapWrapper`，适配器负责完成世界重置及 `episode_metrics/episode_done/truncation`。对照入口为 `tools/smoke_brax.py`；此前已用旧仓库的独立环境（Brax 0.14.2、JAX 0.9.2、Crazyflow 0.3.2）执行 4 世界／64 步 PPO 训练。该验证用于检查接口与参数更新，不能用于比较收敛速度。新仓库默认 Pixi 环境未安装 Brax，可通过可选依赖 `.[brax]` 使用。此依赖组合在新版默认 JAX 版本下尚未单独验收。

同一可选入口已完成原生 APG 的 2 次更新、4 世界、2 步展开，返回有限且非零的训练梯度范数。独立环境类与重置的定向测试通过，CLI 38 项回归通过。一次完整 CPU 套件在执行约 66 个用例后收到 SIGTERM；这次运行不记为全量验收通过。需要再次完整验收时使用隔离执行预算，保留各测试文件的原始结果。

## Runner 与未来扩展

不新增公共训练 `Runner` 类。`cli.py` 已负责实验编排、停止条件与冻结验收，`Trainer` 负责参数更新与恢复；`simulation/runner.py` 负责独立冻结执行。沿用三个真实职责，不建立同名中间层。若将来 Task 参考数据进一步归入环境子类，应保证保留唯一的参考轨迹、约束配置和历史数值语义。
