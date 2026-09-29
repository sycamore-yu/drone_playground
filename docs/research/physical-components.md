# 物理模块组合的工程证据

公共接口为Trajectory、Waypoint和具名MotionCmd。模块按输入／输出合同连接；C++模块不要求参与可微训练链，仍须核验真实状态、测量、时钟、单位、有效期及失败行为。协议接入不表示某个尚未移植的研究算法已经实现。

## 神经几何输出

`networks.physical_outputs.PhysicalOutput`定义输出物理意义，而不以网络隐层维度代替它：

- Waypoint：每个点三个输出，按保存的米制尺度解码为世界坐标有序航点。锚点可为当前位置、任务目标或世界原点，容差单独保存。
- Trajectory：九个输出分别确定终点位置、速度和加速度；固定时长五次曲线从当前位置／速度、零参考加速度出发。它提供真正的未来时域，没有障碍或动态可行性保证。
- MotionCmd：保留既有检查点的具名动作解码。

数值解码支持JAX JIT、批量和求导；轨迹内部采样对输出的导数已与有限差分核对。`save_policy(..., physical_decoder=...)`记录全部解码参数及输出维度，`frozen_neural`核对它们、观测字段、物理模型和策略频率。当前公共学习入口尚未训练这些几何头，不能据此声称通用可微组合训练完成。

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

以下均用未训练的零参数神经网络验证接口，不能作收敛证据。逐次运行及报告摘要见[机器凭据](../verification/physical-components.json)。

| 组合 | 实际时长／调用次数 | 工程结果 |
|---|---|---|
| network→Waypoint→最小jerk→Trajectory→AttitudeMPC | 1s；10／10／50次 | 50次真实MPC执行，无缺失命令 |
| network→Trajectory→AttitudeMPC | 1s；10／50次 | 修复时钟边界后50次真实MPC执行，无缺失命令 |
| network→Waypoint→SUPER→Trajectory→AttitudeMPC | 3s；30／30／150次 | 44次真实MPC执行；启动／预测时域不足时明确回退 |
| network→Waypoint→EGO→Trajectory→AttitudeMPC | 3s；30／30／150次 | 15次真实MPC执行；启动／预测时域不足时明确回退 |

四例均未通过短回合任务的RMSE质量判据。SUPER／EGO结果验证了真实容器算法接受上游航点并返回完整曲线，尚不能说明这些组合适合导航或稳定控制；必须继续独立质量验证。最小jerk曲线采用明确启发式时长，有限曲线不足MPC预测时域时不能伪造未来参考。

首次神经Trajectory试验只有49/50次MPC执行。最小回归复现了`35 * 0.02`与`35 / 50`的浮点差异；执行器现与Trajectory采样共用1ns时钟容差，新的真实acados试验达到50/50。实际不足10ms的预测时域仍被拒绝。
