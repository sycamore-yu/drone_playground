# 归档：项目建立前的设计讨论

历史记录，旧的待定项与选项保留作为推导来源；当前执行范围以 [已接受规格](../../.scratch/drone-platform/spec.md) 为准。
原始文档逐字副本与校验值保存在 experiments/evidence/brax-20260925/source-documents。

---

# Crazyflow 无人机平台：设计对齐记录

日期：2026-09-25。状态：讨论草案，尚未批准完整架构与实施计划。
本文记录已知要求、源码事实、建议模块及待决定项；术语见 `../../CONTEXT.md`。
当前文件位于已有 Crazyflow 工作区，最终项目是否独立建仓仍待决定。

## 1. 用户当前明确要求

- 以 Crazyflow 为底层，延续 GenesisDroneEnv 的完整无人机研究与比较平台构想。
- 覆盖普通强化学习、BPTT/APG、SHAC，以及优化式规划和控制方法。
- 支持不同动力学、MID-360 感知、静态及动态环境。
- 提供训练过程观察、历史轨迹重放和远程查看能力，参考 rscope 的使用体验。
- 方法行为以作者参考实现为依据，平台集成优先复用既有接口。
- 当前讨论聚焦 Crazyflow；本次只开展设计对齐和文档记录。

### 2026-09-25 本轮已确认

- 首版直接使用 Crazyflie 预置机型及其参数。现实飞机标定、换机体在实际迁移时进行。
- MID-360 首先作为虚拟观测模型；当前机体仍按 Crazyflie 参数运行，传感器载荷建模另列。
- 优先组合 Crazyflow 及课题组现有实现，依次考虑原配置、接口适配和必要新增代码。
- 训练库只保留 Brax。SHAC / D.VA 作为同一 JAX 训练体系内的算法扩展；其他训练库仅供阅读。
- 训练观察只采用 rscope 的周期轨迹采集和重放；当前讨论取消额外网页查看器的实施方案。
- 感知训练允许 D.VA 式观测梯度处理；完整传感器反传属于可选研究项。
- Genesis 的 100/40/40 秒和 0.5 米协议只作为历史比较项。新任务优先采用作者定义，再明确必要改动。
- 每轮集中提出前提已清楚的多个决策问题。

## 2. 已核对的已有项目事实

### Crazyflow

- 本地提交：`36f584d114d9d331f0cee0fe4b9066f821c0fbfd`。
- 本轮开始时已有改动为 `pyproject.toml`、`pixi.lock`；MuJoCo/MJX 升级的完整测试结论另行核对。
- 正式注册任务为位置到达、速度跟踪、降落和八字轨迹跟踪；`DroneEnv` 为基础类。
- 四种动力学：`first_principles`、`so_rpy`、`so_rpy_rotor`、`so_rpy_rotor_drag`。
- 三种拟合动力学支持状态/姿态控制；角速度、力矩和电机控制要求 `first_principles`。
- `Sim` 可用于初始化及构建计算函数；可微时间展开使用函数式接口。
- 无人机动力学在 JAX 中推进；MJX 用于场景运动学、接触和射线查询。
- 现有采样控制示例可作为基础；论文完整学习实验的复现程度需要逐个训练器核验。

直接依据：`crazyflow/envs/__init__.py`、`docs/user-guide/dynamics/index.md`、
`crazyflow/sim/functional.py`、`crazyflow/sim/sim.py`、`SKILL.md`。

### GenesisDroneEnv

- 本地提交：`fbf628d0f5972cd684e4aea1fd181dd687ff5719`，本轮读取时工作树干净。
- 已批准协议 `three-task-v1`：连续航点 100 秒、八门竞速 40 秒、安全导航 40 秒。
- 航点与导航使用严格距离 `<0.5 m`；导航首次安全到达即结束。
- 竞速通过运动线段与目标门平面求交，检查方向、门序和门洞；完整圈时间单列。
- 任务核心唯一产生事件；评测器聚合；同一步失败优先于新增有效到达/过门。
- 正式状态观测包含自身状态、目标和任务几何；训练奖励与评测定义分别管理。
- 旧项目 PPO 明确固定 RSL-RL；本轮新平台统一选择 Brax，旧实验继续保留其来源。
- 旧项目四种动力学有自己的定义，与 Crazyflow 四模型不能逐名称对应。
- 现有 `config/sensor/lidar.yaml` 为 36×5 射线、20 米量程的配置；MID-360 保真度需要另建明确契约。
- 正式训练预算、留出测试集、失败记录和独立任务推进规则可继承；旧实验矩阵不自动迁移到新平台。

直接依据：旧仓库 `CONTEXT.md`、`AGENTS.md`、`docs/THREE_TASK_PLAN.md`、
`docs/EVALUATION_PROTOCOL.md`、`docs/EVALUATOR_REFERENCES.md`。

## 3. 建议模块与参考来源

下表为待确认的模块划分；参考表示复用其设计或作者实现，不代表整套项目已接入。
该表表达职责归属，不要求为每行新增一套实现。优先直接使用已有动力学、控制器、任务数学、
Brax PPO/APG、LSY 竞速核心和 rscope；实际新增集中于环境/记录适配、SHAC/D.VA 扩展以及缺少的感知/动态任务。

| 模块 | 建议承担的职责 | 主要复用或参考 |
|---|---|---|
| 机体与动力学配置 | 质量、惯量、推力、电机、控制增益、载荷；区分训练/预测/评测模型 | Crazyflow 参数化、系统辨识及动力学模块 |
| 函数式环境核心 | 显式环境状态、批量重置/步进、随机数、传感器缓存、事件和回报 | Crazyflow Functional API；MuJoCo Playground 环境与包装器边界 |
| 任务与场景 | 连续航点、竞速、静态/动态导航；场景种子、障碍运动及门几何 | GenesisDroneEnv 三任务核心；lsy_drone_racing；FlightBench |
| 传感器与信息边界 | MID-360 扫描序列、量程、外参、时间戳、噪声；状态与深度观测 | Crazyflow MJX 射线；MuJoCo-LiDAR；Livox 官方规格/驱动 |
| 地图与动态感知 | 点云累积、占据/未知空间、方法所需距离场、障碍检测/跟踪/预测 | EGO-Planner、Fast-Planner、SANDO 的原生处理链 |
| 学习训练器 | Brax PPO、BPTT/APG，SHAC/D.VA 扩展；检查点、归一化与回合边界 | Brax training；算法依据 NVlabs/DiffRL、jax_shac、HaoxiangYou/D.VA |
| 优化式方法适配 | 复用已有控制器入口、求解器与任务执行循环 | 首选 Crazyflow sampling.py、lsy_drone_racing AttitudeMPC；按任务补 GCOPTER/EGO/SANDO |
| 统一闭环执行 | 物理/控制/感知/规划各自频率、时间戳、动作保持、轨迹失效、进程通信 | Crazyflow 控制语义；FlightBench 方法边界；作者控制器 |
| 评测与实验管理 | 固定事件协议、独立种子、重评、样本/时间效率、延迟、约束残差、结果清单 | GenesisDroneEnv 冻结协议；FlightBench 指标组织；DiffAero 配置组合 |
| 记录、远程重放与部署边界 | 方法无关的轨迹记录、周期评估、模型快照、传感器及奖励分项；后续实机接口 | rscope；后续实机接口参考 lsy_drone_racing |

## 4. 方法接入及比较边界：建议

执行主链：任务/场景 → 当前允许观测 → 方法 → 带类型和时间戳的输出 → 控制执行 → Crazyflow。
任务核心消费真实执行状态并产生事件；评测与记录读取事件及执行轨迹。

学习策略可输出姿态等低层指令。轨迹规划器保持其轨迹表示，在边界采样为状态参考。
NMPC 输出其原设计控制量。原生 C++/ROS 方法采用外部适配；JAX 训练环直接使用函数式环境。
原生执行链和公共跟踪器比较分别声明。算法代价/训练奖励保持独立，外部评测条件共同冻结。

建议首批代表为 PPO、BPTT/APG、SHAC、已有 AttitudeMPC 和已有采样式 MPC。
GCOPTER、EGO-Planner、Fast-Planner、Fast-Racing、SANDO 按静态导航、竞速和动态感知任务的需要扩展。
上述名单及交付顺序待用户决定，不作为已批准训练矩阵。

本轮统一选择 Brax，验证记录见 `brax-integration-verification.md`。
四种动力学的原生 PPO/APG 参数更新已通过 CPU 短程测试；这证明接入能力，
正式任务收敛与跨种子表现由后续训练评价。SHAC/D.VA 需要新增算法实现，未作为已通过项目列出。
旧 Genesis 的 RSL-RL 训练结果继续保持原身份，不并入新平台 Brax 的结果。

Brax APG 当前显式向量化环境，并拒绝字典观测；适配需要处理单环境/批量轴和观测编码。
Crazyflow 的 `n_worlds` 批量轴与训练器向量化职责必须明确归属，避免重复批量化。
SHAC 以原论文/官方实现为算法依据，JAX 社区实现辅助移植；终端价值参数冻结与状态梯度保留分别检查。

## 5. 影响实验结论的关键契约：待确认

### 动力学与控制

区分“同模型下的优化/学习能力”与“不同训练/预测模型在共同评测机体上的迁移能力”。
首轮可以让训练和评测使用同一模型。共同第一性原理评测作为跨模型研究候选，待用户决定。
这三个名称描述模型的用途，可以全部引用同一配置；无需复制三套动力学实现。
四种 Crazyflow 动力学的直接比较优先采用共同支持的姿态控制；电机级研究形成独立实验组。
机体参数、控制增益、积分方式和频率均进入清单；换动作接口会改变闭环含义。

### 感知与动态环境

场景真值属于仿真器和评测器；策略、规划器及其预测器仅读取所在信息组允许的数据。
静态真值地图、真实传感器测量、动态障碍真值状态/未来轨迹分别声明，避免隐藏的信息优势。
障碍物位姿在仿真时间上更新，并在射线/碰撞查询前同步。
对高速碰撞，采用何种子步检查/扫掠碰撞规则需要跟机体尺度和时间步共同确定。

MID-360 需要扫描方向随时间变化、外参、每点时间、视场、量程、噪声、无回波及必要运动畸变。
训练射线预算和部署输入预处理分别记录；训练/评测分辨率差异需做显式验证。
传感器梯度策略是算法变体：观测端停止梯度时，仍保留策略参数经动作、动力学到损失的导数路径；
该估计与完整观测闭环导数分别标记，不能假设二者相同。

### 时序与预算

同步闭环用于隔离算法行为，实时闭环用于评价计算延迟影响；首版正式采用哪个协议待确认。
实时协议需指定超时、旧轨迹保持、轨迹过期以及安全处理的责任归属，并记录触发次数。
统一任务时长与观测更新规则，分别报告训练交互量、训练时间、评估/记录开销及在线决策延迟。
采样式规划器内部候选轨迹展开计入在线计算开销，与学习训练交互分别报告。

### 评测继承

建议保留各上游任务自己的回合和成功语义，先统一结果记录。旧 Genesis 的
0.5 米/100 秒/40 秒仅列为备选定义；任何新采用的数值明确标为新协议。
评测失败是实验结果；接口错误是工程缺陷；方法不支持的组合明确标为不适用。
共享错误修复只重跑受影响范围；一种方法得分低不阻塞其他授权实验。

## 6. 远程训练观察：建议

复用 rscope 的周期评估和轨迹重放思路，记录器覆盖学习、优化控制和规划方法。
保存机体状态、动态障碍、mocap、控制、观测、回报分项、任务事件、模型/随机化快照及运行标识。
训练参数快照用于生成少量重放轨迹；采集计算和文件写入仍有开销，应单独测量。
仅接入原版 rscope 的桌面远程重放。每次运行独立保存轨迹，另给 rscope 当前活动目录提供导出。

本次复核发现：Playground 当前 `learning/train_jax_ppo.py` 的回调中
`rscope_handle.dump_rollout(params)` 被注释，单加 `--rscope_envs` 不能作为完整采集已运行的证据。
原版 rscope 支持远程 SSH 文件拉取和桌面查看；SHAC/规划器导出与网页交互需要新增适配。

## 7. 依赖与验证

当前新候选 MuJoCo-LiDAR 的 `pyproject.toml` 声明 Python `>=3.10,<3.14`。
已有 Crazyflow 测试环境使用 Python 3.14；接入前应确定双方支持的 Python 环境并锁定依赖。
Python 版本与 MuJoCo 版本属于不同版本轴；本次不修改现有环境。

最小必要验证覆盖：环境前向/反向及有限差分、回合终止/截断与重置前观测、
动态几何与射线一致性、方法输出单位/坐标/时间戳、约束残差和求解状态、重放复核。
共享任务核心由各训练器/外部方法适配器共同消费；用同输入检查行为，避免各方法复制任务逻辑。
正式实验锁定代码、解析后配置、模型、传感器、测试清单及评测版本。

## 8. 待决定的设计树

| 编号 | 决策 | 依赖 | 推荐起点 | 状态 |
|---|---|---|---|---|
| Q1 | 首版机体定位及实机约束 | 用户研究目标 | 直接使用 Crazyflie 预置参数，实机迁移时再标定 | 已确认 |
| Q2 | 首批任务、方法和交付范围 | 已有完整平台目标 | 任务采用各自作者定义，执行/记录共享入口 | 待用户确认 |
| Q3 | 单机还是多机正式任务 | Q2 | 首版单机，各世界独立；多机协作另立任务语义 | 待用户确认 |
| Q4 | 训练/预测/评测动力学与控制组 | Q1、Q2 | 姿态层比较四模型；电机级独立；评测机体固定 | 待用户确认 |
| Q5 | 信息组、MID-360 精度及动态感知职责 | Q1、Q2 | 状态/几何真值与感知组区分；保留方法原感知链 | 待用户确认 |
| Q6 | 首个学习库及梯度策略 | Q2、Q4、Q5 | 仅使用 Brax；允许 D.VA 式传感器梯度处理 | 已确认，接入测试见专门报告 |
| Q7 | 事件协议数值、实时规则与训练预算 | Q1、Q2、Q4 | 继承已有事件原则，再冻结适用阈值及资源口径 | 待用户确认 |
| Q8 | 工程归属与长期交付 | 模块边界 | 独立研究包依赖固定 Crazyflow；也可同仓分层，保留单一正式路径 | 待用户确认 |
| Q9 | 训练观察与实机范围 | Q1、Q2、Q8 | 仅使用 rscope；实机迁移留在后续阶段 | 已确认 |

讨论按决定的依赖关系推进；源码事实直接核对，每轮保留清晰的推荐和待答问题。
不可逆的重要选择经明确确认后再形成独立 ADR；完整设计确认后进入实施计划。

## 9. 已核查参考入口

- Crazyflow 函数式接口：https://learnsyslab.github.io/crazyflow/user-guide/functional-api/
- Crazyflow 动力学兼容表：https://github.com/learnsyslab/crazyflow/blob/main/docs/user-guide/dynamics/index.md
- MuJoCo Playground：https://github.com/google-deepmind/mujoco_playground
- Brax APG：https://github.com/google/brax/blob/main/brax/training/agents/apg/train.py
- SHAC 官方：https://github.com/NVlabs/DiffRL
- JAX SHAC 参考：https://github.com/Andrew-Luo1/jax_shac
- MuJoCo-LiDAR：https://github.com/discoverse-dev/MuJoCo-LiDAR
- Livox MID-360 规格：https://www.livoxtech.com/cn/mid-360/specs
- 竞速环境：https://github.com/learnsyslab/lsy_drone_racing
- 导航比较平台：https://github.com/thu-uav/FlightBench
- 轨迹优化：https://github.com/ZJU-FAST-Lab/GCOPTER
- 静态局部规划：https://github.com/ZJU-FAST-Lab/ego-planner
- 竞速规划：https://github.com/ZJU-FAST-Lab/Fast-Racing
- 动态感知规划：https://github.com/mit-acl/sando
- 远程训练观察：https://github.com/Andrew-Luo1/rscope
- Playground 采集回调：https://github.com/google-deepmind/mujoco_playground/blob/main/learning/train_jax_ppo.py
- 设计访谈：https://github.com/mattpocock/skills/blob/main/skills/engineering/grill-with-docs/SKILL.md

参考入口读取于 2026-09-25。正式采用时固定各仓库提交，核对许可证和实际接口。

## 10. 本轮原实现调查与复用提案

### 可以复用的任务

- Crazyflow `FigureEightEnv`：默认 10 秒八字参考轨迹，距离回报，地面/边界终止。
  `ReachPosEnv` 是固定目标位置控制；与按到达事件连续切换航点的任务分别命名。
- `lsy_drone_racing/control/train_rl.py::RandTrajEnv` 已有随机航点三次样条轨迹：
  默认 10 个构造点、15 秒轨迹。其训练器当前是 PyTorch；本项目复用任务/配置，训练统一交给 Brax。
- `lsy_drone_racing/envs/race_core.py` 已包含函数式数据、步进、重置、过门及碰撞逻辑，
  同时支撑单机、多机和向量环境。优先适配这套实现。
- `config/level0.toml`：4 个实体门，`gate_order=[1,2,3,4,2]`，环境 50 Hz；
  `DroneRaceEnv` 默认 1500 步，相当于 30 秒。评测报告完成状态、过门数量和完成用时。
- 竞速文档名义门洞宽 0.4 米，当前 `gate_passed` 调用传入 `(0.45,0.45)`，
  函数使用各维的一半作为坐标边界。应保留作者判定为参考，并明确是否另设几何一致版本。
- `amacati/so3_primer/benchmarks/crazyflow` 已有 PPO/SAC/TD3、配置、扫描和保存权重重放；
  `simulate.py` 加载 `cfg.toml` 和 `policy.pt`，逐回合累计回报。
- 同组 `safe-control-gym` 已有稳定/轨迹跟踪、约束与扰动、模型式和学习式控制器、
  `BaseExperiment`；其运行物理是 PyBullet，适合借鉴实验职责，不能当作现成 Crazyflow 环境直接运行。

### 优化式方法的原始接入证据

1. Crazyflow `examples/control/sampling.py` 的输出经 `sim.attitude_control` 后由 `sim.step` 执行。
   主仿真为 `first_principles`，预测为 `so_rpy_rotor_drag`。
   当前精英更新是 `jnp.mean(candidates[elite_indices], axis=0)`；方法名保留采样式 MPC，
   与使用指数代价权重的严格 MPPI 分别记录。
2. `lsy_drone_racing/control/attitude_mpc.py` 使用 Crazyflow 的
   `so_rpy.symbolic_dynamics_euler` 构造 acados 模型。
   `AttitudeMPC.compute_control` 调用 `solve()` 后返回 `get(0, "u")`。
   `scripts/sim.py` 统一调用 `controller.compute_control(obs, info)` 与 `env.step(action)`。
3. MuJoCo MPC 倒立摆示例 `python/mujoco_mpc/demos/agent/cartpole.py`：
   `agent.set_state(...)` → `agent.planner_step()` → `data.ctrl=agent.get_action()` → `mujoco.mj_step`。
4. Hydrax 的 `Task` 提供 `running_cost` 和 `terminal_cost`，物理预测来自 MuJoCo MJX。
   它是 MPPI/CEM 等算法参考；直接使用其默认物理会改成 MuJoCo 预测模型，采用时要显式选择。

本地 Crazyflow 0.3.2 已不提供 `crazyflow.drones.load_params`，竞速核心及姿态 MPC 文件仍导入它。
因此这些上游源码值得复用，但与本地当前版本接合需要版本适配；本轮未声称竞速/MPC完整运行通过。

### 评测提案

推荐“任务规则沿用各自作者定义 + Brax 训练指标 + rscope 轨迹记录”。
姿态/轨迹跟踪先用 Crazyflow；竞速先用 LSY；导航单独选择既有来源。
MuJoCo Playground/Brax 提供回合回报、长度、metrics 和检查点回调的组织方式，
任务自己的成功规则仍来自所选环境。
`safe-control-gym::BaseExperiment` 的训练/评测环境和轨迹指标分工适合作为优化控制实验参考。
FlightBench 用于导航场景/指标的备选，Genesis 协议作为连续航点/循环竞速的另一备选。

### 尚需本轮决策的独立问题

1. 基础任务先用 FigureEight + RandTraj，还是首版必须有按到达切换的连续航点？推荐已有两种轨迹任务。
2. 竞速沿用 LSY 完整门序与终止定义，还是保留 Genesis 八门循环计圈？推荐 LSY。
3. 导航首版优先已有静态导航协议，还是静态/动态导航同时构建？推荐明确指定一个上游导航来源后扩展动态。
4. 原作者完整执行链作为默认比较，还是先固定公共跟踪器只比较规划器？推荐默认作者执行链，组件消融另列。
5. 首批优化方法是现有采样 MPC + AttitudeMPC，还是还必须有 GCOPTER/EGO/严格 MPPI？推荐先复用两条已有链。
6. 评测采用任务原生规则加统一记录，还是重新统一各任务阈值？推荐前者。
7. 不同动力学先做同模型训练/评测，还是首轮就统一回第一性原理模型测试？推荐先同模型并保留跨模型评测入口。
8. 感知任务首批并行做 MID-360 和深度，还是指定一个主输入？D.VA 原实现针对图像，点云版本需要明确编码与变体定义。

补充原始来源：
- https://github.com/DietrichGebert/ponytail/blob/main/README.md
- https://github.com/learnsyslab/lsy_drone_racing/blob/main/lsy_drone_racing/control/attitude_mpc.py
- https://github.com/learnsyslab/lsy_drone_racing/blob/main/lsy_drone_racing/control/train_rl.py
- https://github.com/learnsyslab/lsy_drone_racing/blob/main/lsy_drone_racing/envs/race_core.py
- https://github.com/learnsyslab/lsy_drone_racing/blob/main/config/level0.toml
- https://github.com/learnsyslab/lsy_drone_racing/blob/main/scripts/sim.py
- https://github.com/amacati/so3_primer/blob/main/benchmarks/crazyflow/simulate.py
- https://github.com/learnsyslab/safe-control-gym/blob/main/safe_control_gym/experiments/base_experiment.py
- https://github.com/google-deepmind/mujoco_mpc/blob/main/python/mujoco_mpc/demos/agent/cartpole.py
- https://github.com/vincekurtz/hydrax/blob/main/README.md
- https://github.com/HaoxiangYou/D.VA
