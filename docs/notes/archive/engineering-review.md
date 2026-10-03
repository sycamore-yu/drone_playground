# Robot learning 工程审计

本项目的专业标准是：配置改变真实执行，实验条件可核对，指标与结论相符，失败完整计数，代码只保留实际使用的能力。本次按已有论文配方与 Navigation8 目标修正工程问题。

## 已修正的问题

| 问题 | 现役实现与验证方式 |
|---|---|
| 动力学 DR 缺乏实际行为保证 | Crazyflow 在训练 reset 采样质量、电机能力及支持的阻力参数；刚体模型可采样惯量，拟合姿态模型拒绝无响应的惯量 DR。保存实际参数；不支持的组合拒绝启动。行为测试验证质量改变运动、刚体惯量改变角速度，以及随机键的可重复性。名义选模和 benchmark 关闭训练随机化；鲁棒性评测显式选固定条件。 |
| 配置频率与真实时钟可能分离 | `env.task.freq` 唯一决定控制步长，删除 `execution.frequency_hz`。25Hz 测试实际推进20个500Hz物理步。宿主模块自己的调用频率仍保留。 |
| 通用代码嵌入 Navigation8 的实验限制 | 通用层检查动作、观测、动力学及频率兼容性。50Hz／10Hz、0.5m、300秒、场景角色与阈值由唯一的 [navigation protocol](../../../src/drone_playground/benchmarks/navigation.yaml) 校验。普通自定义导航可以使用25Hz、0.3m和其他时限。 |
| 固定64回合、后缀06决定场景角色 | 回合、种子、主要场景和验收规则读取协议及显式覆盖。测试使用不同场景名称和每场景3回合，验证选模无编号约定；通用导航实际执行每场景2回合的八个独立初态。 |
| 测量、选模、交付混在 Evaluator | Evaluator 返回原始统计；`evaluation/protocols.py` 应用选模和验收规则。修改配置阈值可改变判定，无需修改指标计算。 |
| 旧 split、字段、迁移和报告回退继续存在 | 现役只接受 Environment role `train`／`eval` 和 config version 3；checkpoint evaluation 是训练内选模活动，Benchmark 是正式评测规格。删除旧产物迁移、旧评测别名、历史目录猜测及无消费者汇总器。冻结证据留存原文。 |
| PPO 与 SHAC 按算法各自解释超时 | 任务声明 `time_limit_kind`。悬停、跟踪使用 truncation；导航和竞速使用 termination。Brax 固定源码补丁使 PPO 从重置前的末态计算价值，保留截断转移的奖励，并切断跨回合 GAE。 |
| 决策耗时与传感频率口径不一致 | 同步测量“观测已可用→可执行命令”，排除物理推进、观测生成和归档；去掉首次预热，记录p50、p95和超过控制周期的比例。传感记录名义频率、实际捕获频率、周期和仿真时间戳；50Hz控制下30Hz请求实际捕获25Hz。 |
| 每篇方法增加构造和调度分支 | Hydra 声明 task constructor、trainer、policy evaluator 与评测入口。保留任务、观测、动作、可微性和原生执行的真实能力差异。 |
| 同图结果缺乏解释条件 | `manifest.json` 与 `result.json` 保存 `experiment_conditions`：场景、动力学、观测／传感器、动作单位、控制器、频率、延迟、安全余量及方法；构造后补充真实时钟与条件摘要。 |

删去无消费者的 v2 配方夹具、旧控制与学习汇总器、旧迁移 CLI、旧目录整理与日期猜测、重复协议和声明配置、旧传感观测类、Simulation／PX4 空包及无调用者的小函数。还删除了未接入的 `initial_condition_randomization` 字段与重复的点云 task 延迟字段；实际延迟只读取 runtime。当前协议没有独立开发集；训练内 checkpoint evaluation 使用普通 `eval` 环境语义。

## 追加审计的三类问题

| 类型 | 仓库中的实际问题 | 标准概念与最小修改 |
|---|---|---|
| 机制缺失或没有一等配置 | 测量噪声与训练内点云增强散落；运行时外力与源场景噪声混用 | 增加有实际消费者的 `observation_noise`、`action_noise`、`disturbance`。测量只改变策略输入，模型参数在 reset 采样，外力按物理时钟作用。 |
| 私人化或误导命名 | `free_course_v1` 同时包办位置、速度、场景相位；固定 scene bank 被称为 curriculum；点云／深度适配被当作不同任务 | 初态拆为 `reset_randomization` 字段；固定库称 `scene_distribution`／training distribution；深度和点云共用 RecurrentNavigationEnv，名字只保留在 preset 或来源身份中。 |
| 独立概念混在一起 | LOTF 同时定义模型、任务、方法、训练器与评测器；LSY scene 默认注入噪声；点云训练内评测复用带 DR 的训练对象；Hydra 后加载的默认配置覆盖环境中的实际条件 | LOTF 仅提供两种动力学，四类通用任务复用；场景只拥有几何和运动；构造同 Task 的名义 eval_env；修正配置加载顺序并验证真实执行。 |

训练分布分别声明初态、command 和 scene：position goal、velocity command、reference 与障碍几何分离；fixed／generated／procedural 对应实际生成时机。固定场景、固定速度范围、固定混合权重都不是 curriculum，当前没有自适应课程，因此删除误称并不新增空接口。

| 机制 | 训练时 | 冻结评测时 |
|---|---|---|
| Dynamics DR | reset 采样质量／惯量／电机等真实参数 | eval 使用标准参数 |
| Reset randomization | 采样位置、姿态、速度和相位 | 固定种子与协议初态，逐回合保存 |
| Command／scene distribution | 可采样目标与训练几何 | 固定目标、固定 benchmark 几何 |
| Observation noise | 可启用测量误差与丢测 | eval 关闭训练测量噪声 |
| Policy exploration | 由训练算法的策略分布拥有 | 冻结策略执行，不更新参数或归一化统计 |
| Action uncertainty／disturbance | 显式控制误差和运行时外力 | eval 关闭训练 effects |

LOTF full 使用原生刚体、旋翼与 Betaflight 更新；simplified 使用原生简化方程。两者提供给同一 Hovering／Tracking／Racing／Navigation 环境，没有独立 task、method、trainer、evaluator 或 split。源码原先无开关的 ±15% 推力随机化已退役，改由显式参数随机化控制；full 支持惯量，simplified 拒绝无效惯量配置。PointMassLag 的加速度模型没有质量／惯量响应，明确拒绝这些字段，支持电机增益和时间滞后倍率。

按本会话修改前的源码备份核对，本轮审计清除33个文件、5527行旧代码／配置／夹具，不计并行会话的任务 preset 改名。新机制保留实际消费者。隐式扰动竞速的四份旧数值样本已清除，原跟踪数值仍逐项回归。

## 两类实验

`study.type=method_reproduction` 保留各方法的论文观测、动力学、控制接口与训练规则，Navigation8 上比较完整方法的任务表现。第一版18格属于这种方法与任务的交付矩阵。

`study.type=controlled_comparison` 要求显式 `conditions_id`，固定观测、动力学、动作、控制器、频率、延迟和安全条件，仅替换被研究的算法。运行条件与摘要用于核对共同设置，完整网络、奖励、随机化和训练预算保存在解析配置中。比较算法时也须核对这些配置；相同地图本身不能证明算法对照条件相同。

专业设计允许这两类实验并存。无需为了主表而改掉论文方法的实际模型。

## 对此前六项建议的澄清

- **点质量从哪里来**：`models/point_mass.py` 的 `PointMassLag` 是点云文献方法及导航迁移使用的加速度驱动模型，有一阶滞后；通用 Crazyflow 使用自己的无人机动力学。这是方法重建的模型选择，本次保留。
- **18格为什么不是统一算法排名**：格子验收完整方法在任务上的表现；不同方法可以使用不同传感器、动力学和执行接口。现在显式记录条件，排名或算法消融再采用 controlled comparison。
- **为什么不新增训练地图**：Navigation8 是当前确认的 benchmark，用户没有提出未见地图泛化目标。本次不要求接入新训练分布。
- **当前有没有 DR 或噪声**：已有初态、初始速度扰动和动作传输延迟；启用 Crazyflow DR 后还会改变真实动力学。默认使用理想测量；现在可显式添加状态测量误差、深度／点云米制误差与丢测，且不会改写真实状态和奖励。初态、模型参数、测量误差、动作误差和运行时外力分别配置。
- **为什么不做 wheel**：当前约定是源码与锁定 Pixi 环境，足以完成项目目标。[Crazyflow 0.3.2 发布 wheel](https://pypi.org/project/crazyflow/0.3.2/) 是上游的分发选择，不构成本项目额外要求。
- **统计报告意味着什么**：逐训练种子、逐场景保留结果，不把失败从分母移除。置信区间与平滑性指标用于需要它们的论文比较，本次不增加未声明的交付门槛。

## duration 与 DiffAero

`duration` 是单回合最多运行的仿真秒数。`horizon_length` 是训练器每次展开的步数，两者属于不同层。

例如持续悬停到10秒时，任务仍可以继续；训练为了有限回合而重置，这属于截断，价值从10秒末态延续。导航要求300秒内到达，到期属于任务结束，后续价值为零。碰撞、越界和数值失效属于真实终止。

DiffAero 的 [BaseEnv](https://github.com/flyingbitac/diffaero/blob/291ea14196aefbebcf7387dd71f7e096c83878b7/env/base_env.py) 将 `max_time / dt` 转成步数并单独返回 truncated；[SHAC](https://github.com/flyingbitac/diffaero/blob/291ea14196aefbebcf7387dd71f7e096c83878b7/algo/SHAC.py) 对截断读取 `next_obs_before_reset` 计算末态价值。本次沿用这一任务与算法的职责划分。

## 对比依据

| 上游固定版本 | 本次采用的依据 |
|---|---|
| [IsaacLab](https://github.com/isaac-sim/IsaacLab/blob/28a37cecdd433c22d9eabd6a5954add9f13a8951/source/isaaclab/isaaclab/envs/mdp/events.py) | mass／inertia 随机化写入实际仿真参数，任务配置表达具体条件。 |
| [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground/blob/ef4fefc13033c0468af4ef651847f5348af0c7d7/learning/train_jax_ppo.py) | 同任务分别构造 train env 与 eval env，普通训练内评测无需另外定义开发数据集。自动选最好权重是本项目的显式规则，不声称所有上游默认如此。 |
| [Flightmare](https://github.com/uzh-rpg/flightmare/blob/d4218aedac18cbe9364a0a0df10ab992c4b65e4f/flightlib/src/objects/quadrotor.cpp) | 四旋翼执行模型是具体方法条件；论文点质量模型无需冒充相同刚体模型。 |
| [FlightBench](https://github.com/thu-uav/FlightBench/blob/af88801836c4a21c3e80f8a3e2182961afcb73a1/flightbench/script/bag_parser.py) | 从实际轨迹计算明确指标，记录各方法实际执行条件。 |

[MuJoCo Playground Go1 task](https://github.com/google-deepmind/mujoco_playground/blob/ef4fefc13033c0468af4ef651847f5348af0c7d7/mujoco_playground/_src/locomotion/go1/joystick.py) 分开声明 noise、command 与 perturbation；[IsaacLab velocity task](https://github.com/isaac-sim/IsaacLab/blob/28a37cecdd433c22d9eabd6a5954add9f13a8951/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py) 分开 commands、reset／push events 和会实际调整难度的 curriculum。本项目采用这些职责语义，保留 JAX 无人机模型的实际能力边界。

新增行为验证覆盖：测量噪声改变输入且保持物理状态与奖励不变；延迟命令仍应用动作误差；外力改变三种模型的速度；LOTF full 的质量与惯量改变响应；点质量电机／滞后改变加速度；初态姿态随机化、相同种子重现、名义竞速关闭隐藏扰动。

公开入口短训练 `professional-lotf-shared-20261001` 使用通用 BPTT 与 LOTF high-fidelity dynamics，完成8个决策步、策略参数变化 L2 为0.11998，保存并选出权重。该工程短训练只验证共享 Task、Dynamics 与 diffRL 梯度链路，不增加原18格收敛或质量通过数。

针对性验证覆盖本轮39项行为检查，均已有通过记录；另有点云训练、评测、固定场景采样和回放的15项检查通过。固定场景按实际实例数采样到声明的训练并行数，回放初帧速度读取真实执行初态。lint、差异格式与当前文档链接检查通过；固定源码获取与锁定安装通过。存在多个会话同时编辑，按用户要求停止完整回归，后续合并后统一执行。

验证与当前运行状态统一记录在 [status](../../status.md)；机制见 [architecture](../../architecture.md)，操作见 [runbook](../../runbook.md)。工程短训练验证执行路径，不增加既有18格策略质量通过数。
