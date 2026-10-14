# drone_playground 功能规格

2026-10-09 批准的后续架构调整见[文档入口](README.md)：构造阶段的方法组合、保留 Crazyflow 前向的训练导数、原生物理随机化及结果布局。对应 ADR 明确区分已批准设计与实际实现；本规格中的首发验收条件保持不变。

## 1. 目标

基于 **Crazyflow** 建立可扩展的四旋翼学习、规划与控制研究平台。平台统一前向物理、场景、传感器、任务判定和实验记录，支持在明确的实验条件下比较学习方法、优化控制和完整自主飞行系统。

研究采用两种口径：


| 研究口径   | 实验约定                            | 结果                       |
| ------ | ------------------------------- | ------------------------ |
| 具名方法适配 | 明确论文来源、网络、观测、损失、动力学、控制链及移植改动    | 报告方法在 Crazyflow 中的实际表现   |
| 受控组件对照 | 固定任务、场景、观测权限、前向物理、时钟、动作、预算和下游控制 | 报告所更换模块对任务质量、训练效率和计算量的影响 |


平台支持结构化学习与优化器组合，例如网络预测轨迹时间分配、代价参数或约束，由 Planner/MPC 负责可行的运动求解。方法内部可包含网络、优化器和预测模型；公共接口只约定实际输入、输出、时钟和状态。

## 2. 用户故事

1. 作为研究者，我希望通过配置创建 Tracking、Racing 或 Navigation 环境，以便使用统一物理和任务规则进行实验。
2. 作为研究者，我希望独立运行 Planner、Controller、Policy 和冻结模型，以便在不启动训练器时比较方法。
3. 作为研究者，我希望将新的规划、学习或联合优化方法接入相同环境，以便持续扩展方法集合。
4. 作为研究者，我希望分别使用 Depth 和 LiDAR 训练感知策略，以便比较传感方式与训练算法。
5. 作为研究者，我希望训练 PPO、APG/BPTT 和 SHAC，以便比较任务表现、学习曲线与计算成本。
6. 作为研究者，我希望使用实际的 EGO-Planner 与 SUPER，以便评价端到端的原生规划执行链。
7. 作为研究者，我希望可重复地评测每个独立回合并在 RScope 查看轨迹，以便分析成功、失败和决策过程。
8. 作为研究者，我希望保存参数、训练状态、配置与来源，以便恢复训练并复核冻结结果。



## 3. 架构与执行合同

项目是一个 Git 仓库和一个可安装 Python 包。**Simulation** 负责 Task、Scene、Sensor、Method、闭环执行、评测和回放；**Learning** 负责训练采样、损失、参数更新、优化器和恢复。Learning 依赖 Simulation；冻结方法通过 Simulation 独立运行。

```text
Learning: PPO / APG(BPTT) / SHAC
                    ↓
Simulation: Task + Scene + Sensor + Method + Evaluation
                    ↓
             Crazyflow Control
                    ↓
          Crazyflow step / reset
```

Simulation 使用 Crazyflow 的 `Sim`、`SimData`、原生控制器、纯 JAX 步进函数、Pipeline 和 MJCF/MJX 场景能力。物理推进及已有低层控制直接调用 Crazyflow。新方法以自身实现、配置和必要适配扩展，无需修改物理积分器。

2026-10-09 的控制模型改动已接入[独立反向模型](adr/0008-independent-backward-dynamics-model.md)、[任务/动作解耦及控制器](adr/0001-simulation-learning-boundary.md)、[延迟缓冲](adr/0009-delayed-data-in-episode-state.md)和原生物理随机化。两种 MPC 与 SUPER 理想跟踪使用相同的方法组合；Trajectory 保持宿主实现。实际命令见[使用说明](control-models.md)，本次针对性验证不替代历史收敛验收。

### 3.1 方法组合

```text
状态、测量、目标
    ├── Planner ──> Path / Trajectory ──> Controller ──┐
    └── Policy / 联合 MPC ──> 控制设定值 ───────────────┤
                                                       ↓
                                           必要的低层控制
                                                       ↓
                                                Crazyflow
```

- **Path** 描述几何路线；**Trajectory** 有明确执行时间、有效期和实际提供的运动导数。
- **Setpoint** 明确坐标系、单位和控制层级：速度、加速度、姿态、机体系角速度、力矩或电机转速。
- 方法声明输入权限、输出合同、运行频率、状态 reset 和失败行为；相关控制器执行剩余的控制转换。
- JAX 方法使用显式参数、方法状态、RNG 和纯函数，实现批量、JIT 及所需梯度；原生 C++/ROS 方法使用宿主执行与进程适配。
- 配置选择具体实现，接口按实际角色保持小而稳定。任务在更换方法时保持成功、碰撞与超时定义。

方法可以内部结合学习、搜索、地图、SFC、MPC 或神经轨迹优化。EGO-Planner、SUPER、AllocNet、MPCC、AC-MPC、ViTFly 是代表性方法，接口面向持续增加的无人机算法。

### 3.2 原生 C++ / ROS

EGO-Planner 与 SUPER 在独立 ROS1 Noetic 环境中运行，通过 Protobuf/gRPC 连接 Simulation。通信接口传递机体状态、传感测量、目标和规划结果，并提供初始化、回合重置、决策请求与资源释放。C++ 直接使用 Protobuf 生成的服务接口。Simulation 负责控制执行、物理时间和任务评价；JAX 方法在进程内运行。

## 4. Task 与场景


| Task           | 参考和场景                            | 主要结果                |
| -------------- | -------------------------------- | ------------------- |
| **Tracking**   | Hover、Figure-eight、随机样条等时间参考；空场景 | 完成率、位置 RMSE、坠毁与超时   |
| **Racing**     | LSY Level0 赛道，规定过门顺序 `1→2→3→4→2` | 合法完赛、碰撞、漏门、完赛时间     |
| **Navigation** | 固定 Navigation8；静态与动态分别评价         | 安全到达、碰撞、越界、超时、净空与用时 |


资产位于 `assets/scenes/`：

- Navigation8 主场景：静态 `S01/S02/S03`，动态 `D01/D02/D03`；三维扩展场景 `S06/D06` 仍完整报告。
- Navigation 世界边界为 `[0,-20,0.5]` 至 `[100,20,6]` m，名义起点 `(2,0,3)` m、终点 `(98,0,3)` m。
- `catalog.xml` 指定固定实例，`boundary.xml` 定义边界；动态障碍在仿真时间更新 mocap 位姿。运动语义及素材来源见 [场景资产说明](../assets/scenes/README.md)。
- 训练可用程序化随机图元；冻结 Benchmark 引用指定场景及其已确定的几何身份。

Navigation 基准：目标半径 `0.5 m`、机体碰撞球半径 `0.07 m`、最长 `300 s`，名义速度上限 `20 m/s`。重置时位置扰动半宽 `(0.25,0.25,0.10) m`、速度各轴 `±0.1 m/s`、yaw `±5°`，初态净空至少 `0.15 m`。同刻事件按**碰撞→数值失败→越界→到达→超时**判定。

**Episode（回合）**是一次独立初始化到首次终止的完整执行；评测包含成功、碰撞、失败和超时。物理步、训练更新、训练种子与回合分别计数。环境区分任务终止（termination）和采样截断（truncation），只重置结束的批量实例。

场景、传感器、碰撞和回放读取同一份几何及同一时刻的位姿；规划和策略的观测只包含实验允许的信息。

## 5. 传感器与时间

平台使用具名虚拟设备配置，硬件参数与策略预处理分别声明。


| Sensor profile          | 设备参数                                                               | 策略输入                         |
| ----------------------- | ------------------------------------------------------------------ | ---------------------------- |
| **D435i**               | Depth FoV 约 `87° × 58°`，支持 `1280×720 @30 FPS` 模式；有效量程按设备模式配置       | CNN/GRU 可使用独立的低分辨率、逆深度和历史预处理 |
| **Mid-360**             | 水平 `360°`、垂直约 `−7°～52°`，`10 Hz`、约 `200k points/s`，非重复扫描；量程按目标反射率配置 | PointNet/GRU 接收有时间与有效点信息的点云  |
| **uniform_lidar_paper** | `180×30`、2° 等角度网格                                                  | Liu 论文式对照配置                  |


相机内外参、深度定义、有效值、LiDAR 射线方向、扫描相位、量程与误差模型属于 Sensor；降采样、历史堆叠和网络归一化属于 Observation。记录物理时间、传感采集时间、观测可用时间、方法调用周期及动作延迟；非整数频率按实际时间调度。Mid-360 的 20k 全帧来自设备公开点率而非 MuJoCo 内置约定；`sensor.ray_sampling=policy` 仅计算全帧中供 1024 点策略使用的确定性选定射线，保留原全帧时序并显式区分训练配方。回放/传感器完整性仍支持 `full` 模式。

D435i 预设提供设备标定；训练的 `64×48、30 Hz、20°、10 m` 由 Sensor 配置明确声明。Mid-360 使用 MuJoCo-LiDAR 0.3.5 的角度表，默认每帧连续读取 20,000 条。两种传感器均以同一帧时刻的机器人和动态障碍位姿生成快照，各点时间相同；延迟交付与采集时间分别记录。扫描相位、源文件哈希及快照语义随运行和 checkpoint 保存。详见 [sensor](adr/0012-sensor.md) 与 [render](adr/0013-render.md)。本功能分支的模型变化尚未完成冻结任务验收。

## 6. Learning：算法与具名方法

训练器为 **PPO、APG/BPTT、SHAC**。APG/BPTT 是一个训练算法入口，不同任务通过具名网络、目标及导数配置实现。

### 6.1 训练矩阵与共享 Actor 网络

| 训练算法 | Tracking / State | Racing / State | Nav 静态 / Depth | Nav 动态 / Depth | Nav 静态 / LiDAR | Nav 动态 / LiDAR |
|---|---|---|---|---|---|---|
| **PPO** | State MLP | State MLP | Zhang CNN/GRU | Zhang CNN/GRU | Liu PointNet/GRU | Liu PointNet/GRU |
| **APG/BPTT** | State MLP | State MLP | **Zhang 2025 CNN/GRU** | **Zhang 2025 CNN/GRU** | **Liu 2026 PointNet/GRU** | **Liu 2026 PointNet/GRU** |
| **SHAC** | State MLP | State MLP | Zhang CNN/GRU | Zhang CNN/GRU | Liu PointNet/GRU | Liu PointNet/GRU |

首发共 **18 个验收单元**（3 种训练算法 × 6 个任务／感知条件）。表格中的 Zhang/Liu 表示**Actor 网络架构来源**；两篇论文自身采用可微仿真训练，PPO/SHAC 是本平台新增的算法组合。静态与动态须分别冻结评测；同一配方可使用一份混合场景训练策略进行两组评测。独立训练种子与评测回合分别计数。

### 6.2 网络、训练算法与未达标调整

**统一 Actor Backbone（策略主干）：**

| 任务与观测 | 三种算法统一的 Actor Backbone | 物理动作输出 | 架构来源 |
|---|---|---|---|
| Tracking / State | MLP **[256,128]**，LayerNorm + ELU | 姿态／推力 | [DiffAero MLP](https://github.com/flyingbitac/diffaero/blob/main/cfg/network/mlp.yaml) |
| Racing / State | MLP **[256,128]**，LayerNorm + ELU | 姿态／推力 | 同一 DiffAero MLP；竞速使用任务所需的观测与门／轨迹参考 |
| Navigation / Depth | CNN **32/64/128** →192 维特征与状态融合→**GRU192** | 净加速度 | [Zhang 2025](https://doi.org/10.1038/s42256-025-01048-0) |
| Navigation / LiDAR | PointNet **64/128/1024**、max pooling →192 维特征与状态融合→**GRU192** | 净加速度 | [Liu 2026](https://rasevents.org/uploads/documents/pdfviewer/a9/f4/233762-1123.pdf) |

同一任务与传感器条件下，对齐 Actor 输入字段、主干、动作头及 Crazyflow 控制转换。Tracking 与 Racing **共享网络结构，各自保留任务观测与参考语义**。Depth 的速度估计辅助头沿用 Zhang 方法配方，是否加入辅助损失须在训练配置中声明；PPO 和 SHAC 使用各自的 **Critic／价值网络**与学习目标。

| 算法 | 更新与梯度职责 | 参考实现 |
|---|---|---|
| PPO | rollout、GAE、策略裁剪、价值更新、随机策略 | [DiffAero](https://github.com/flyingbitac/diffaero)、[VisFly](https://github.com/SJTU-ViSYS-team/VisFly) |
| APG/BPTT | Crazyflow 可微展开、直接梯度；Zhang/Liu 感知方法还包括时间敏感度衰减 | DiffAero、[VisFly-Lab/APG](https://github.com/Fanxing-LI/APG)、Zhang/Liu |
| SHAC | 短时可微展开、Actor/Critic 更新、末端价值、终止与截断 | DiffAero、VisFly-Lab |

网络、辅助损失与梯度规则的论文来源及参考参数见 [学习方法研究](research/learning-methods.md)。Zhang/Liu 原方法采用简化质点；本平台将其网络与训练方法适配至 Crazyflow，并验证物理控制转换与反向规则。

**训练未达标时的调整规则：**先通过训练诊断核查观测和动作量纲、损失、reset、终止处理及梯度传播。若未达到 L3，可参考 DiffAero、VisFly、VisFly-Lab、Zhang 和 Liu 的原始配置，调整 Actor／Critic 层数与宽度、GRU、输入预处理、学习率、batch size、rollout 长度、归一化、奖励／损失权重和梯度衰减等。

每次调整保存独立的实验配置、参数差异、调整原因及训练／评测记录。**进行训练算法的受控比较时，应在相同的新 Actor 配置下重新对照各算法**，并明确记录算法特有的 Critic、优化目标及梯度差异。使用轻量筛查和正式 `checkpoint_eval` 选模，以 `benchmark` 做独立冻结验证；每次验收记录其使用的协议版本，详见 [ADR-0015](adr/0015-three-stage-learning-acceptance.md)。

### 6.3 L3 的收敛测量指标（不再独立编号）


| 测量内容 | 正式验收门槛 |
|---|---|
| 随机初始化 | 每个训练配方 **3 个独立训练种子**；严禁从旧模型初始化，允许同一 run 的完整训练状态断点恢复 |
| Tracking | 每种子正式 **100 回合**；完成率 **≥95%**，成功回合位置 RMSE **≤0.25 m** |
| Racing | 每种子正式 **100 回合**；合法完赛率 **≥90%**，记录过门顺序与完赛时间 |
| Navigation | 每种子、每个主场景正式 **25 回合**，每场景 **≥23/25** 安全到达；S01/S02/S03 与 D01/D02/D03 分别计数，S06/D06 仅报告 |
| 三阶段 | 训练中轻量筛查：导航六主场景各 8 回合、状态任务 8 回合；通过后完整 checkpoint validation；固定通过的权重，以独立测试种子执行一次冻结 benchmark |


筛查、完整选模与最终冻结测试使用不重叠的随机种子区间。训练过程记录失败和未达标状态，三个算法均需真实参数更新、保存与冻结闭环。旧 run 保留原始合同及参数，不混改历史结论；新训练使用 [ADR-0015](adr/0015-three-stage-learning-acceptance.md) 与 [ADR-0018](adr/0018-training-patience-and-frozen-selection.md)：首次完整达标立即独立冻结一次；≥800 updates 后连续 6 次轻量筛查无明显改善则以 `early_stopped` 结束配方。上表用于 L3 的研究结论，不构成第三套验收编号。GPU 吞吐、墙钟时间、显存与到达门槛的计算成本归 **L4** 统一记录。

## 7. Simulation 验收 S1—S4


| 编号 | 验收内容 | 证据 |
|---|---|---|
| **S1** | **安装与物理正确性**：Pixi 锁定依赖、可安装包/资产、Crazyflow 相同子步下的状态一致性 | 可安装结果与固定种子物理对照 |
| **S2** | **任务环境与传感器**：Tracking/Racing/Navigation、静态及动态场景、D435i/MID360 观测、传感时钟与规划器信息权限 | 环境/几何/传感器专项验证；完整 20k 与稀疏策略射线模式区分 |
| **S3** | **方法扩展和原生接入**：Policy、Planner、Controller 组合，EGO-Planner/SUPER 通过 C++ gRPC + ROS 真实执行 | 原生控制轨迹、求解日志与独立 S3 演示 |
| **S4** | **回放与可复现性**：RS​​cope 场景/姿态/传感命中回放，记录种子、环境身份、延迟、所有失败回合和耗时 | 可复查的 Result 及回放；显示不修改物理 |


**S3 中原生 EGO/SUPER 的具体要求保持不变**：每种方法至少 10 个静态回合、10 个动态回合，覆盖主场景；在 `S01/S02/S03` 每场景至少一次安全到达。动态只要求完整记录求解、控制、碰撞和超时，**不设动态成功率门槛**。学习型 Navigation 仍按 L3 独立阈值验收。

## 8. Learning 验收 L1—L4


| 编号 | 验收内容 | 证据 |
|---|---|---|
| **L1** | **训练算法正确性**：PPO、APG/BPTT、SHAC 的采样、梯度、参数更新、回合 reset、终止/截断、GRU、恢复 | 算法专项小规模测试，不每次训练都重跑全套 |
| **L2** | **感知学习闭环**：State MLP、Zhang CNN/GRU、Liu PointNet/GRU 的输入、动作、导数、传感器到训练/冻结闭环 | 观测和网络专项验证；冻结 Actor 无特权信息泄露 |
| **L3** | **从零收敛**：3 算法 × Tracking、Racing、Nav Static/Dynamic × Depth/LiDAR，共 18 单元；每配方 3 种子；遵守 §6.3 和三阶段选模 | 完整 checkpoint validation + 独立冻结 benchmark；逐种子 100/25 回合结果 |
| **L4** | **计算效率与证据**：真实 JAX/Crazyflow 吞吐、训练耗时、GPU 资源、权重/优化器恢复、收敛所需交互数 | TensorBoard、Orbax、Results 与独占吞吐对照；不要求填满 VRAM |


Simulation 与 Learning 使用同一 Crazyflow 前向和同一 Task 判定。**v0.2 的 18 单元正式验收全部从随机网络参数训练**，不得用旧仓库、参考项目或历史 run 权重初始化。DiffAero、旧仓库和论文只用于参考奖励、网络、观测和超参数。现行矩阵结果由被 Git 忽略的 `results/scratch_v02/matrix.json` 汇总，历史验收数据不能替代。八项验收的合并决策参见 [ADR-0017](adr/0017-eight-acceptance-gates.md)；旧 C1–C6、L1–L6、S1–S8 仅用于解读历史记录。

## 9. 工具、配置与交付

- **Pixi**：`pyproject.toml` 的 `[tool.pixi]` 与 `pixi.lock` 管理 Python、GPU/CPU 依赖；统一 `pixi run` 的安装、仿真、训练、评测、播放与检查任务。ROS1 的系统环境使用独立容器。
- **Hydra/YAML**：唯一的实验配置组合来源。参数分别由 Task、Scene、Sensor、Method、Learning、Benchmark 持有，输出完整解析配置。
- **RScope**：导出 MJCF/mesh、`.mj_unroll` 与相应回放数据，支持实际轨迹、场景运动、点云命中与 Planner 结果显示。
- **Google Python Style + MuJoCo 风格**：Python 3.12，Ruff 100 列、Google docstrings；详见 [STYLEGUIDE](../STYLEGUIDE.md)。

```text
results/<run_id>/
  config.yaml
  events.out.tfevents.*
  run.json
  checkpoints/
    model_000025/
      training/          # Orbax 训练状态
      metadata.json
    model_000050/
      training/
      metadata.json
    model_000075/
      training/
      actor/             # 选模成功导出的 Orbax 策略
      metadata.json
      screening/         # 可选筛查记录
      report.json        # 可选完整选模报告
  episodes.csv           # 正式冻结评测后生成
  report.json            # 正式冻结评测总表
  replays/
    best/                # 开启回放时生成 RScope
```

新训练使用 [ADR-0016](adr/0016-results-layout-orbax-tensorboard.md)；`run.json`
仅保留训练来源、进度及完成状态，不复制 TensorBoard 指标流，取消 `metrics.jsonl` 与
`eval/`。回放仅在请求时记录；历史 ZIP 结果不移动或改写。

## 10. 首发边界

首发覆盖单机四旋翼、三任务、18 个训练验收单元、Depth/LiDAR、EGO/SUPER 原生接口、RScope 和 Pixi。可扩展到多机、传感融合、其它控制方法及具名论文适配。首发只使用 **S1–S4、L1–L4 共八项独立验收**；§6.3 的任务质量与种子条件是 L3 内部指标，不再单列完成任务。
