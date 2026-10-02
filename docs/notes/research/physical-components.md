# 物理模块组合的工程证据

公共接口为Trajectory、Waypoint和具名MotionCmd。模块按输入／输出合同连接；C++模块不要求参与可微训练链，仍须核验真实状态、测量、时钟、单位、有效期及失败行为。协议接入不表示某个尚未移植的研究算法已经实现。

## 神经几何输出

`networks.physical_outputs.PhysicalActionDecoder`定义输出物理意义，而不以网络隐层维度代替它：

- Waypoint：每个点三个输出，按保存的米制尺度解码为世界坐标有序航点。锚点可为当前位置、任务目标或世界原点，容差单独保存。
- Trajectory：九个输出分别确定终点位置、速度和加速度；固定时长五次曲线从当前位置／速度、零参考加速度出发。它提供真正的未来时域，没有障碍或动态可行性保证。
- MotionCmd：保留既有检查点的具名动作解码。

数值解码支持JAX JIT、批量和求导；轨迹内部采样对输出的导数已与有限差分核对。`save_policy(..., physical_decoder=...)`记录全部解码参数及输出维度，`frozen_neural`核对它们、观测字段、物理模型和策略频率。公开`experiment=learning/geometric`现可训练几何头，下游选择明确的JAX PD跟踪器；它不等于任意宿主模块链均支持求导。

例如，已取得相应物理头的冻结检查点后可以配置：

```yaml
method:
  implementation: pipeline
  output: trajectory
  stages:
  - implementation: frozen_neural
    checkpoint: experiments/<run>/checkpoints/<snapshot>.pkl
    frequency_hz: 10
  - implementation: native_service
    algorithm: super
    input: waypoint
    output: trajectory
    frequency_hz: 10
    parameters:
      use_upstream_waypoints: true
    deployment:
      address: <already-running-adapter-address>
```

同时选择实际需要的传感器和`env.execution.tracker`，如AttitudeMPC；采用已有服务地址时，宿主不会拥有或关闭外部进程。SUPER／EGO仍需要其原ROS运行环境。已有通用`deployment.command`也可管理本地可执行程序或容器内进程，C++ SDK不依赖ROS。

ROS航点模式在能力表中声明并要求上游Waypoint，按容差推进顺序；重复相同航点不会丢失进度，新序列重新计数，Reset清空进度。原任务goal在此模式下只作上下文。默认任务目标模式与原基线保持独立。

## 已执行的代表链

以下均用未训练的零参数神经网络验证接口，不能作收敛证据。逐次运行及报告摘要见[机器凭据](../../artifacts/verification/physical-components.json)。

| 组合 | 实际时长／调用次数 | 工程结果 |
|---|---|---|
| network→Waypoint→最小jerk→Trajectory→AttitudeMPC | 1s；10／10／50次 | 50次真实MPC执行，无缺失命令 |
| network→Trajectory→AttitudeMPC | 1s；10／50次 | 修复时钟边界后50次真实MPC执行，无缺失命令 |
| network→Waypoint→SUPER→Trajectory→AttitudeMPC | 3s；30／30／150次 | 44次真实MPC执行；启动／预测时域不足时明确回退 |
| network→Waypoint→EGO→Trajectory→AttitudeMPC | 3s；30／30／150次 | 15次真实MPC执行；启动／预测时域不足时明确回退 |

四例均未通过短回合任务的RMSE质量判据。SUPER／EGO结果验证了真实容器算法接受上游航点并返回完整曲线，尚不能说明这些组合适合导航或稳定控制；必须继续独立质量验证。最小jerk曲线采用明确启发式时长，有限曲线不足MPC预测时域时不能伪造未来参考。

首次神经Trajectory试验只有49/50次MPC执行。最小回归复现了`35 * 0.02`与`35 / 50`的浮点差异；执行器现与Trajectory采样共用1ns时钟容差，新的真实acados试验达到50/50。实际不足10ms的预测时域仍被拒绝。


## JAX组件训练入口

`experiment=learning/geometric`复用既有PPO／SHAC／BPTT更新器、环境事件、物理模型和延迟队列。网络输出单个Waypoint或一段五次Trajectory，先经过`jax_waypoint_tracking`或`jax_trajectory_tracking`变成四维姿态／推力，再进入原执行链。多步物理梯度已与有限差分比较；静态悬停处的PD梯度也保持有限。此入口目前采用策略与PD同频，连续多航点的进度仍使用宿主模式，不隐式重解释为单点。

默认轨迹PD明确采样当前曲线前方0.1秒的位置、速度和加速度；原ROS当前执行样本、当前轨迹PD及此JAX前视PD分别命名。选择零前视会令每次新曲线的起点与当前状态相同、几何头失去有效作用，因此训练配置要求前视大于零且不超过轨迹时域。

几何策略的`goal_source=observation_reference`从已声明的观测字段恢复第一个世界参考点，不读取隐藏的未来真值。冻结到宿主后保持这一来源，外部规划器自己的任务目标不会替换它；改尺度、锚点、时长或目标来源的参数热启动须显式迁移。完整BPTT／SHAC恢复仍保留原优化器、随机数和物理／延迟状态。

最小BPTT工程检查可运行：

```bash
JAX_PLATFORMS=cpu pixi run train experiment=learning/geometric env=hovering runtime.device=cpu env.task.duration=0.2 training.num_envs=2 training.policy_updates=2 algorithm.horizon_length=8 training.num_evals=2 training.checkpoint_eval_episodes=2 'network.hidden_sizes=[8,8]' run_id=geometric-entry-check
```

PPO改用`algorithm=ppo network=brax_ppo`并显式给出总交互步数与批量配置；SHAC改用`algorithm=shac`。单航点配方同时设置`method.output=waypoint method.physical_decoder.kind=waypoint env.execution.command=waypoint controller@env.execution.tracker=jax_waypoint_tracking`。网络结构、训练算法、物理输出和下游控制器是独立配置，兼容性在构建前检查。

BPTT轨迹头、SHAC航点头和PPO轨迹头已完成两并行、32决策步的小型训练及检查点重载评测。它们只验证更新／保存／加载链路，短回合RMSE未达标，不计入18格收敛矩阵。39项几何训练／物理输出／宿主组合回归通过。含RPC或当前原生MPC的组合继续用于宿主执行，这个JAX训练入口会明确拒绝；不要求为C++算法补导数。

## 规划轨迹进入神经跟踪器

冻结神经模块可显式声明`input: trajectory`，放在优化规划器之后。模块从公共位置／姿态／速度／体轴角速度与上游曲线重建检查点记录的`state_reference`观测，按原采样间隔取得未来位置，再输出其声明的物理类型。宿主自身可以使用深度或点云观测；这个跟踪器实际读取的字段及曲线采样偏移记录在`pipeline-contracts.json`中。

例如，在已有输出Trajectory的规划阶段后添加：

```yaml
- implementation: frozen_neural
  input: trajectory
  checkpoint: experiments/<trained-controller>/checkpoints/<snapshot>.pkl
  frequency_hz: 50
```

若检查点输出姿态／推力，则整链输出和执行命令均须为`attitude_thrust`。原检查点的机型、动力学、策略频率及动作解码仍须匹配；轨迹替换不意味着跨动力学迁移或导航质量已通过。默认10个参考点、0.1秒间隔需要至少0.9秒真实未来轨迹。时域不足返回`no_plan`，分数采样步长存在歧义时拒绝构建；Waypoint须先经过显式时间分配／轨迹生成模块。

23项相关检查通过，覆盖实际保存权重加载、外部参考改变网络动作、宿主观测隔离、原观测语义检查、未来时域缺失及配置式Waypoint→轨迹→神经命令链。此模式为宿主执行，不改变C++无导数的约定。

公开控制入口现同时支持`state`与`state_reference`宿主观测，原始传感器仍单独配置。真实C++轨迹服务→已训练BPTT控制器的1秒闭环完成10／50次调用、50次执行，无缺失命令；宿主仅13维状态，神经模块自行构造43维状态／参考输入。该曲线演示不遵循悬停目标，任务RMSE为1.093m，质量未通过；结果只证明这条组合可实际执行，不计入18格。首次入口因旧观测限制被拒绝的日志保留，修正后14项入口／原生控制回归通过。

## MotionCmd经过C++的完整回合对照

固定提交`56c1802`以同一已训练BPTT策略执行FigureEight：一组由冻结网络直接输出MotionCmd，另一组只在其后插入独立C++透传服务，再进入相同执行系统。两组均使用开发种子20000–20003、50Hz、每回合10秒和25–50ms动作延迟配置；C++侧实际处理2000次请求。两组均4/4完成，完成回合均值RMSE均为0.038500m；2000对逐帧物理命令、状态、观测、任务指标及全部保存轨迹字段逐值一致。配置差异仅为运行标识与插入的透传阶段，见[对照凭据](../../artifacts/verification/motion-echo-equivalence.json)。

该对照验证同一控制器经MotionCmd接口和gRPC后保留执行行为，没有训练新权重。透传服务是接口夹具，不是新增规划算法；4个开发回合不增加18格通过数。同步仿真保留配置的动作延迟，但没有把测得的RPC墙钟耗时额外注入动力学。此冻结版本未逐回合保存延迟数值，后续评测已补齐记录。
