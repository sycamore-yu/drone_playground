# drone_playground 领域术语

用于四旋翼学习、规划、控制及实验比较的统一语言。本文只定义领域概念；源码组织、外部依赖及实现决策记录在 ADR 和功能规格中。

## 任务与试验

**Task（任务）**：
定义一次飞行试验要实现的目标，以及成功、失败与进度的语义。
_Avoid_: Method、Algorithm、Scene

**Tracking（轨迹跟踪）**：
飞行器按照给定的时间参考运动，偏差是主要评价对象。
_Avoid_: Navigation

**Racing（竞速）**：
飞行器按规定方向和顺序通过赛道门，并在完成合法飞行的前提下评价用时。
_Avoid_: 仅按轨迹 RMSE 判定竞速

**Navigation（导航）**：
飞行器从初始位置安全到达目标，需考虑障碍物与任务限制。
_Avoid_: Vision-Language Navigation（除非明确标注为 VLN）

**Scene（场景）**：
试验中的空间几何、障碍物、门及其规定的动态变化。
_Avoid_: Task、Benchmark Case

**Episode（回合）**：
从一次独立初态开始，到首次成功、失败或任务截止为止的一次完整飞行试验；成功与失败都计入试验次数。
_Avoid_: 物理步、训练更新次数、训练种子、场景数；将失败后的重试并入原回合

**Training Seed（训练种子）**：
决定一次独立参数训练随机过程的起始条件；同一已训练策略可以在许多不同回合上接受评测。
_Avoid_: 评测回合种子、场景编号

**Benchmark Case（基准案例）**：
预先确定的场景、初态、目标、外部条件及评价协议。
_Avoid_: Training Run

## 物理信息与运动

**Dynamics（动力学）**：
将无人机状态及作用于机体的物理输入映射为下一状态的运动规律。
_Avoid_: Planner、Controller

**Backward dynamics model（反向求导模型）**：
训练中用于计算状态和动作灵敏度的动力学模型。它有自己的状态、输入和转移方程，可以与实际推进模型相同或不同。
_Avoid_: 逆动力学、时间倒放、梯度衰减

**Derivative rule（导数规则）**：
训练反向传播时使用的输入变化与输出变化之间的关系。可以使用前向运动方程的真实导数，也可以使用明确声明的近似导数。
_Avoid_: 将导数规则等同于另一套前向动力学

**Gradient decay（梯度衰减）**：
按规定因子缩放传播的梯度，同时保持所处位置的前向数值不变。
_Avoid_: 物理阻尼、执行器滞后

**Transport delay（送达延迟）**：
测量或命令产生之后，到接收方可以使用它之间的时间。
_Avoid_: 采样周期、物理执行器的响应时间常数

**Sensor Measurement（传感器测量）**：
传感器在确定时刻和坐标定义下获得的物理观测量。
_Avoid_: Observation

**Observation（方法观测）**：
某一方法被允许使用的测量、状态、目标及其表示形式。
_Avoid_: 全部仿真真值

**Privileged Information（特权信息）**：
仿真中存在、但当前方法在指定实验条件下未获准使用的真实状态、地图或未来信息。
_Avoid_: 默认可用的观测

**Path（路径）**：
具有方向及参数化的几何运动路线，不隐含执行时刻。
_Avoid_: Trajectory

**Trajectory（时标轨迹）**：
带明确时间关系及有效区间的参考运动，可以包含位置、速度、加速度等实际给定的量。
_Avoid_: 只有几何航点的 Path

**Waypoint（航点）**：
空间中的目标点或有序目标，不自动具有到达时间。
_Avoid_: 完整 Trajectory

**Setpoint（控制设定值）**：
对飞行器某一级控制目标的要求，具有明确的量纲、坐标系和时间语义。
_Avoid_: 含义不明的 Command 数组

**Control interface（控制接口）**：
约定接收的控制量、顺序、单位、坐标及有效范围。姿态、角速度和加速度接口描述不同物理输入。
_Avoid_: 由任务名称或传感器名称推断控制量

## 方法与研究

**Planner（规划器）**：
根据允许的信息计算用于完成任务的路径、轨迹或其它规划结果；其内部可以包含学习与优化。
_Avoid_: 将 Planner 等同于非学习算法

**Controller（控制器）**：
根据状态、参考或设定目标产生更低层的控制请求；内部可以包含优化器或学习模块。
_Avoid_: 将 Controller 等同于 PID

**Policy（策略）**：
根据当前允许的观测及可能的内部记忆产生决策的映射。
_Avoid_: 将 Policy 等同于训练算法

**Method（运行方法）**：
用于实际闭环执行的一种决策链，可以是 Planner、Controller、Policy 或它们的组合。
_Avoid_: Algorithm（训练更新规则）

**Training Algorithm（训练算法）**：
利用采样、损失或示范数据调整可学习参数的更新方法。
_Avoid_: 冻结 Method

**Forward Simulation（前向仿真）**：
从给定状态和决策出发运行飞行物理、场景与测量并得到下一状态和事件的过程。
_Avoid_: Training

**Ideal tracking（理想跟踪）**：
假设实际位置和速度等于参考值，并从参考加速度和航向构造姿态。它表示零跟踪误差的理想闭环响应；具体软件接口和实验用途由单独批准的设计确定。
_Avoid_: 实际物理闭环、MPC 控制器

**Controlled Comparison（受控组件对照）**：
明确固定除研究变量之外的实验条件，以比较某个算法或模块变化的影响。
_Avoid_: 不同传感器与执行链的完整系统比较

**Named Adaptation（具名适配）**：
将一个有明确来源的方法迁移到新试验条件，并记录保留及修改的部分。
_Avoid_: 声称未经修改的原方法复现

**Convergence Acceptance（收敛验收）**：
按照预先定义的验证协议与任务指标，训练方法稳定达到约定质量的判定。
_Avoid_: 仅损失下降、参数变化或训练程序成功退出

**From-scratch Training（从零训练）**：
每个独立训练运行从本次种子产生的随机 Actor、Critic 和新优化器状态开始，不导入其它
运行得到的权重。恢复同一次运行的检查点仍属于这次从零训练。
_Avoid_: 只重建优化器或 Critic 就称为从零；使用其它算法或种子的权重作为共同初始化

**Pretrained Initialization（预训练初始化）**：
使用已有训练权重作为新运行的起点，再执行新训练；来源可以是本项目，也可以是外部参考。
_Avoid_: 将预训练初始化等同于未训练；只报告继续训练成本而忽略初始化来源

## 关系

- 一个 **Task** 可以在多个 **Scene** 或 **Benchmark Case** 中执行。
- 一个 **Method** 可以包含多个 Planner、Controller 或 Policy；它们的内部数学不等于平台公共接口。
- **Sensor Measurement** 是物理测量，**Observation** 是特定方法实际允许接收的输入。
- **Training Algorithm** 更新方法参数，冻结的 **Method** 仍可独立前向执行。
- **Path** 与 **Trajectory** 的区别是时间语义，不是文件或网络输出维数。
