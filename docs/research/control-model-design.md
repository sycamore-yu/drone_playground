# 控制接口、反向模型与方法组合：设计和代码审查

**Status:** Proposed · 2026-10-09

本文件提出完整实现建议，不是新的批准记录。已经批准的原则见 [ADR-0001](../adr/0001-simulation-learning-boundary.md)、[ADR-0008](../adr/0008-independent-backward-dynamics-model.md) 和 [ADR-0009](../adr/0009-delayed-data-in-episode-state.md)。SE3Controller、文件改名和本文件的具体接口尚未批准。

## 1. 目标与判断标准

同一个 Tracking、Racing 或 Navigation 任务可以选择不同控制接口和控制器。正向继续选择现有 Crazyflow 模型；训练可以单独选择 LOTF 或 PointMass 反向模型。PPO、APG 和 SHAC 使用相同的环境交互入口。

判断改动是否有效，看三个实际操作：换 SO3 不需要改任务和损失；换 CNN 不需要改动作维数；换反向模型不改变同一次前向飞行。默认实验配方可以同时选择多个组件，但这种搭配不固化成通用运行代码的限制。

## 2. 怎样定义当前的问题

这里使用已有概念：**不必要耦合（unnecessary coupling）**、**职责分离（separation of concerns）**和**重复知识**。不是每个 `if` 或默认值都有问题。

本项目的具体判断是：两个本可独立变化的选择被代码强行绑定，导致改变其中一个时，必须修改本不应变化的模块。例如 Tracking 是任务选择，姿态是控制接口选择；使用 `task == tracking` 推导四维姿态动作，就把两个不同原因的变化绑定起来。

需要在多处修改同一事实，是这种耦合的可观察后果。修复目标是让一个事实有一个负责模块，而不是为每个分支创建新类。[单一职责原则的原始解释](https://blog.cleancoder.com/uncle-bob/2014/05/08/SingleReponsibilityPrinciple.html)按变化原因解释内聚与耦合，支持这个判断。

以下约束应保留：Task 决定如何判定过门；深度编码器检查输入图像形状；力矩控制检查前向模型是否支持转动力学。这些约束来自真实语义，不属于任意绑定。

## 3. 命名：保留 methods，不改成 motion

`motion.py` 表示运动，无法说明模块是运动数据、运动学公式，还是 Planner/Controller 的组合。当前 `methods.py` 与领域词典中的 Method 对应，改名不会解决职责混合。

建议保留 `methods.py` 中的 Planner、Controller、Policy 协议及 PlannerController 组合；新增 SO3 时，把两个具体控制器放入一个 `controllers.py`。现有 `policy.py` 负责冻结加载和策略执行适配。两个具体控制器不需要各自一层包。

Path、Trajectory、Setpoint 暂时保留原位置。只有它们造成实际的循环依赖或独立复用困难时，再按 `references.py`、`setpoints.py` 等真实职责移动；不建立泛化的 `types.py`、`contracts.py` 或额外 SDK。

建议的相关文件，不包括无需修改的模块：

```text
simulation/
  environment.py        实际状态、时钟、执行、任务及传感
  methods.py            小接口和 PlannerController
  controllers.py        Mellinger、SO3 的可独立数学实现
  policy.py             冻结策略加载、动作解码和调用
  networks.py           网络结构、输出头和循环状态初始化
  observation.py        测量历史与可单独选择的观测编码
  dynamics/
    lotf.py             LOTF 简化模型数学实现
    point_mass.py       PointMassLag 数学实现
learning/
  trainer.py            共用训练转移调用点
  dynamics.py           需要独立文件时放置模型求导连接
```

最后一个文件只在连接逻辑达到独立职责时建立。不会为选择一个函数而增加几层转发。

## 4. 由控制接口决定动作

### 4.1 每个模块负责什么

| 模块 | 自己决定 | 从外部接收 |
|---|---|---|
| Task | 成功/失败/进度、参考需求、任务误差 | 实际状态、场景和任务参数 |
| Sensor | 原始测量、标定、采样时钟 | 场景和物理位姿 |
| Observation | 允许的字段、坐标变换、网络预处理 | 已送达测量和被允许的状态/目标 |
| 控制接口 | 动作含义、顺序、单位、坐标、上下界 | 已解析的控制配方与标称参数 |
| Actor | 编码器、输出计算和内部记忆 | 观测形状和控制接口给出的输出维数 |
| Controller | 从参考或上一级指令计算下一级控制量 | 所需机体观测、标称参数及显式记忆 |
| Environment | 实際物理、时间、命令送达、事件和测量更新 | 构造好的执行函数与组件 |
| Trainer | 采样、梯度、优化器、更新 | 构造好的环境、Actor、损失和模型求导连接 |

控制接口不是一类新的调度器。先使用现有 Setpoint 的层级/坐标标签和一份上下界数据表达；动作维数由数值布局导出。只有两个实际调用方需要共享行为时，才提取小型数据结构或函数。

### 4.2 物理动作与归一化动作分开

Policy 输出通常是 `[-1,1]` 中的数。所选控制接口解码为物理量，Controller 根据需要进行下一层转换。换控制接口可以改变输出维数，但不能暗改同一维的含义。

| 策略接口 | 维数 | 物理含义 | 标称静止悬停 |
|---|---:|---|---|
| 净加速度 | 3 | 世界系或声明坐标系的 m/s² | 初始速度为零时 `[0,0,0]` |
| 姿态与推力 | 4 | roll/pitch/yaw 为 rad，总推力为 N | 水平姿态与约 `mg` 的推力 |
| 推力与角速度 | 4 | 总推力为 N，机体系角速度为 rad/s | 约 `mg` 的推力和零角速度 |

四维并不等于同一个接口。具体字段顺序遵循各原生边界，并通过一次明确转换对齐。

动作解码只存在一份。日志、训练代价和反向模型读取解码后的物理量，不再次编写 `action * 6.0`。正则项需要区分请求的加速度与实际加速度；换成姿态策略后不能无声地改变原论文损失的含义。

### 4.3 构造流程与配置

构造阶段读取 Task/Sensor 配置、创建 Method 和 Controller，确定其物理输出层级，再选择相容的 Crazyflow 原生 Control。Actor 输出维数由策略控制接口获得。控制器构造接收参数与频率，不要求读取整个已初始化 Environment。

```text
Policy → 一次动作解码 → 所需控制转换 → 物理 Setpoint
Planner → Trajectory → 注入的 Controller → 物理 Setpoint
                                             ↓
                                命令送达 → Crazyflow
```

Policy 的下游转换与轨迹跟踪 Controller 不强制是同一个函数；依据实际输入声明选择。直接输出原生设定值的 Policy 不添加空控制器类。

下面是待审配置示意。除了 `learning.backward_model` 外，这些新嵌套字段尚未批准或实现：

```yaml
simulation:
  dynamics: first_principles
method:
  name: policy
  action:
    level: acceleration
    frame: world
    low: [-6.0, -6.0, -6.0]
    high: [6.0, 6.0, 6.0]
  actor:
    kind: state
learning:
  algorithm: apg
  backward_model: point_mass_lag
```

Actor 的 MLP/CNN/PointNet 选择与动作输出头分开。此示例表示 MLP 输出三维加速度；将任务改为 Tracking 不会强制变成四维。具体下游控制转换由实验配方明确提供，不能由任务名推断。

控制器的物理输出决定所用 Crazyflow Control，不再要求用户填写第二份必须一致的层级字段。参数值继续从 Hydra/YAML 获取；无需同时增加全局 Registry。

### 4.4 状态、运行与保存

JAX 方法在调用时显式携带参数、记忆和随机状态；Actor 提供自己的初始记忆，runner 不创建固定的 192 维数组。宿主求解器可以持有原生对象，但必须有回合重置和关闭语义。

PlannerController 持有当前有效轨迹、规划/控制时钟和对应状态。Mellinger 只需要当前参考点时采样当前时刻；MPC 需要预测时域时取得一段未来参考。不能为统一接口而提前把整条轨迹缩成一个位置点。

Environment 保留原生 SimData 和现有 EnvState，不建立另一份完整 DroneState。Controller 只接收其允许使用的字段，防止把仿真真实随机参数自动泄漏给方法。

冻结 Checkpoint 保存实际网络结构、记忆规格和动作布局。仅有 `kind=depth` 不能唯一确定输出维数。完整恢复还保存方法记忆、延迟状态和反向模型配置。

## 5. 反向模型对齐究竟解决什么

它解决的是：策略输出的数字，是否正好是用于求导的模型所接受的物理量。

例子一：策略要求“向前净加速度 1 m/s²”。前向把这个请求转换为倾角和推力，交给 Crazyflow；PointMass 反向模型也接受加速度请求，并估计把 1 改为 1.01 对后续位置和速度的影响。两边请求相同，执行响应可以不同。

例子二：策略输出总推力和机体系角速度。Crazyflow 前向使用 body-rate 控制入口；LOTF 简化模型用相同的推力/角速度估计状态变化。论文用质量归一化推力，代码入口可以用 N 并在函数内除以质量，这个单位转换必须保留。

姿态策略的“倾斜到 10°”与角速度模型的“以 10°/s 转动”不是同一请求，不能直接喂入同一个四维数组。需要显式的姿态控制转换，或使用一个独立的角速度配方。更换反向模型不会自动改变已有策略输出。

调用边界需覆盖输入缩放、坐标转换、必要控制器和时间长度。若模型使用压缩状态，还要映射模型输出的导数回调用方的状态字段。JVP/VJP 不要求显式分配一个大 Jacobian 矩阵。

LOTF 保留位置、旋转与速度；PointMassLag 保留位置、速度和滞后加速度。PointMass 未描述的姿态、电机状态不能默认填单位导数或全部断开。具体配方应声明只替换哪些块、其余块是否使用原生导数或一个明确的姿态响应模型。这些细节是实现研究，不在本文件假定已经批准。

延迟也在同一转移里对齐。40 ms 后才生效的动作，反向模型不能让它立刻影响位置；缓冲状态的导数应把作用时间传回原来的动作。

验证分成前向一致性与导数一致性：替代模型不改变实际 rollout；近似导数对照其声明的模型方程和完整输入转换，不能要求它等于高保真前向的有限差分。未指定反向模型时则验证原生导数。

## 6. 已核实的其他职责绑定

以下为当前工作树的源码事实和修改建议，不表示已运行对应故障。行号于 2026-10-09 读取；工作树还会变化。

| 问题 | 源码证据 | 影响 | 建议 |
|---|---|---|---|
| Task 决定动作，Environment 固定 Mellinger/attitude | `environment.py:90–114,197–218` | Tracking 不能直接选加速度，独立控制器需改核心环境 | 控制配方决定动作和低层入口；构造时注入 |
| Sensor 名字同时决定 Actor、动作和损失 | `cli.py:77`；`networks.py:64–71,116–118`；`trainer.py:127–157` | Depth 更换网络或控制层级时波及训练器 | 传感、观测编码、网络和损失由配方组合；通用训练器不按设备名选论文 |
| 192 维循环状态分散在调用方 | `networks.py:60,114`；`trainer.py:188`；`runner.py:93` | 更换 GRU 大小需改多个模块，MLP 也拿到无用数组 | 初始记忆由 Actor 生成，运行方只保存返回状态 |
| 加速度尺度重复 | `environment.py:202`；`trainer.py:324` | 改动作范围可能使实际命令与损失使用不同尺度 | 共用一次物理解码；损失显式取请求量或测量量 |
| 观测排列成为 Trainer 的隐含知识 | `environment.py:250–252`；`trainer.py:386–391` | 改观测顺序就改变辅助速度监督目标 | 观测编码提供明确的辅助目标，网络仅接收声明字段 |
| 感知采集与特定张量形状绑定 | `environment.py:105–110`；`observation.py:140–159` | 原生规划器也被绑定到 Zhang 64×48 采集方案 | 采集保存原始 Measurement，Actor 专用编码单独调用 |
| Racing 固定赛道与门序重复 | `environment.py:81–82`；`tasks.py:90–108,201–213`；`scene.py:222–223`；`runner.py:134–135` | 改门数、门序和赛道时多处同步 | 门几何来自 Scene、顺序来自任务实例；记录读取实际任务合同 |
| 方法名字和传输方式决定评测分派 | `evaluation.py:73–84,124–132,176–214`；`runner.py:62–89` | 加新 Controller/Planner 需修改通用评测；ROS 身份隐含选 S6 | 构造时选择执行函数，评测协议由配置明确传入，保留原门槛 |
| 随机真实质量参与动作缩放 | `environment.py:203–205`；`methods.py:105–109` | 扰动实验中策略输出隐含使用真实质量，但 Mellinger 又绑定标称质量 | 统一标称缩放与实际物理参数的权限；明示是否提供真实参数 |
| 通用 Trajectory 暂为宿主 NumPy 对象 | `methods.py:22–58` | 学习式 Planner 的轨迹无法直接进入 JAX 求导组合 | 需要可微轨迹时，用现有类型表达 JAX pytree；宿主校验在 JIT 外，不维护两套同义轨迹 |

优先同时处理动作/网络/尺度和方法构造，其次处理测量编码与循环记忆。门序和轨迹类型分别随对应扩展改动，不以一次重写整个仓库作为目标。

## 7. 延迟状态：一个具体例子

相机在 1.000 s 采集图像，30 ms 后送达。策略以 50 Hz 运行，因此在 1.040 s 才读取这帧并发送命令。命令再延迟 40 ms，1.080 s 才开始生效。等待期间物理持续执行旧命令。

命令缓冲位于 EnvState 的执行状态；测量缓冲位于 ObservationState。含义只是“每个环境带着自己的历史数据”，不是每条数据一个 Python 对象或每个缓冲区一个进程。

只对当前已送达数据做读取。reset 按 mask 清空；恢复复制历史及随机状态。固定整数步延迟可以使用循环索引；当前多频率传感和按秒定义的延迟适合保存送达时刻。二者实现的是相同的延迟缓冲概念。

延迟初期尚无有效测量时保持无效，不能预填未来采样。命令初值应按声明的控制接口选择，不能把所有接口的零向量都当悬停。随机延迟是否每回合固定、是否支持逐条抖动仍是实现选择；建议首版固定或按回合采样，避免先引入乱序网络协议。

## 8. SO3 与 SE3 的参考

SO3 已批准加入。第一份实现对照 EGO 的 `SO3Control.cpp`：它计算世界系力向量和目标姿态。JAX 移植保持数学计算可独立调用，ROS 不进入控制公式。将输出转换为 Crazyflow 所需的推力/姿态或更低层输入时，明确剩余控制链，避免重复应用姿态内环。

待选 SE3 有两个直接参考：

| 来源 | 已有实现 | 参考价值与条件 |
|---|---|---|
| [RotorPy SE3Control](https://github.com/spencerfolk/rotorpy/blob/main/rotorpy/controllers/quadrotor_control.py) | NumPy/SciPy 版本及 PyTorch 批量版本；输出推力、力矩、姿态和电机指令等 | 与现有 Python 工作流较接近；不是 JAX 即插即用实现；角速度参考有简化 |
| [RotorS LeePositionController](https://github.com/ethz-asl/rotors_simulator/blob/master/rotors_control/include/rotors_control/lee_position_controller.h) | Eigen/C++ 几何跟踪控制器，含控制分配 | 适合对照轨迹到电机的完整控制链，需要单位、旋翼顺序和机体参数适配 |

SO(3) 描述旋转，SE(3) 描述旋转和平移；控制器文件名不能当作能力等级。EGO 名为 SO3Control 的类本身也使用位置和速度误差。是否同时加入 SE3 应看是否要比较其力矩控制、角速度参考、控制分配等实际差异。

SO3/Mellinger 都是轨迹跟踪选项。将来纳入输出力矩或电机命令的 SE3 实现时，直接使用 Crazyflow 相容的低层入口，不能再经过一个产生力矩的姿态控制器。SE3 的加入和来源尚未决定。

## 9. 三个框架最值得借鉴的部分

### Agilicious：参考采样与控制计算分开

[官方 Pipeline](https://agilicious.readthedocs.io/en/latest/modules/pipeline.html)先取得状态和参考，再采样控制目标，调用外环/内环控制器，最后通过 bridge 传给执行器。[控制器模块](https://agilicious.readthedocs.io/en/latest/modules/controller.html)提供 MPC、几何控制、PID 和 INDI。

本项目应保留同一 Trajectory 对当前点及未来时域的访问，让 Mellinger/SO3 与 MPC 读取各自真正需要的参考。低层执行接在控制器输出之后，不能所有控制器都再套一遍姿态控制。无需照搬完整 Pipeline 管理器。

### Nav2：成功判定不归具体控制算法

[Controller Server](https://docs.nav2.org/rolling/configuration_and_development/configuration_guide/core_servers/controller_server/)分别接收控制器、进度检查器和目标检查器。控制器插件不是任务完成语义的所有者。

本项目继续让 Task 判定到达、碰撞、过门和超时；构造阶段决定具体 Method 和执行函数。新控制器不改任务判定，也不因为它使用 ROS 就自动选择较宽松的验收。Hydra 已经足够选择实现，无需另建 Nav2 式 ROS Server。

### safe-control-gym：运行控制器与它相信的模型分开

[BaseController](https://github.com/learnsyslab/safe-control-gym/blob/main/safe_control_gym/controllers/base_controller.py)提供 `select_action`、reset/close 和可配置的 prior model。[项目](https://github.com/learnsyslab/safe-control-gym)将环境扰动、先验模型以及多个控制算法放在同一比较框架。

本项目应区分实际 Crazyflow 参数、Controller/MPC 的标称或预测模型、训练反向模型。相同的模型数学实现可以按需要复用，默认值可引用现有配置；不要把这三个用途自动视为同一份真实参数。该先验模型机制与反向模型并非同一种算法，其参考价值是信息与参数归属的分离。

## 10. 实施顺序与验证

建议先把现有配方原样表示成控制接口配置，验证既有策略解码结果不变；再注入 Mellinger、加入 SO3；随后接入 LOTF/PointMass 反向模型和延迟缓冲。逐步验证同一条可运行链，不另起一套训练器。

| 变化 | 必须能证明的行为 |
|---|---|
| 任务与动作解耦 | 同任务可构造加速度/姿态/角速度配方；不支持的物理组合在构造时报错 |
| 控制器注入 | 同一状态/参考下更换 Controller 不改变 Task 规则；SO3 数值与选定上游实现一致 |
| 网络与接口分离 | 输出维数来自控制接口；更换记忆大小不修改 runner；冻结加载恢复完整规格 |
| 反向模型 | 前向结果不变；JVP/VJP 与声明的近似模型和变量映射一致；默认仍为原生导数 |
| 延迟 | 未到期不生效；独立世界、局部 reset 和恢复正确；梯度回到实际产生动作的时刻 |
| 现有数据 | 旧权重和验收结果保留；若动作语义改变则新建配方，不把旧权重重新解释 |

只有具体范围再次获得批准后才实施新增接口和文件移动。已批准的原则不需要反复确认。

## 11. 证据范围

本轮读取当前本地源码和既有 ADR；仓库尚无 Git 提交，因此用文件路径、函数和读取时行号定位，不宣称是稳定发布版本。源码基线摘要保存在 `tmp/architecture-controls-20261009/before.json`。

DiffAero 本地对照版本为 `291ea14196aefbebcf7387dd71f7e096c83878b7`，位置为 `/home/tong/tongworkspace/reference_repos/diffaero-upstream-291ea14/`；重点文件是 `env/base_env.py`、`dynamics/base_dynamics.py`、`dynamics/pointmass.py`、`dynamics/quadrotor.py`、`dynamics/controller.py`。该版本从模型读取动作维数和动作上下界，Quadrotor 内部绑定 RateController。本项目要独立比较 Controller，所以保留外部注入而不照搬这个内部绑定。

LOTF 本地对照为旧仓库的 `tmp/sources/lotf/lotf/objects/quadrotor_obj.py`，重点为 `_step_jvp` 和 `simplified_dyn`。论文：[LOTF III-B/III-E](https://arxiv.org/html/2508.21065v2)、[DiffAero III-A/IV-D](https://arxiv.org/html/2509.10247v1)。官方接口参考：[Crazyflow Control](https://learnsyslab.github.io/crazyflow/user-guide/control/)、[JAX 自定义导数](https://docs.jax.dev/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html)。

审查没有运行新训练或控制器验证。源码证据说明当前绑定；上述修改效果须由实施后的测试确认。
