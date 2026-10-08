# 回放可视化接口

回放层只消费真实记录，不参与观测、奖励、控制或动力学。现役规划检查对象是
`SafeFlightCorridor`（SFC）和 `TrajectoryPreview`；可执行计划仍使用公共
`Trajectory` / `Waypoint`。每个对象保存自己的生成时间和有效期，不能把多种
检查数据压成一个总括有效期。

`ReplayLayers` 将传感器标定、真实决策归档和规划检查数据写入标准
`.mj_unroll`。Navigation 回放直接附加固定场景的 MJCF；动态障碍只更新运行时
mocap 位姿，不重新构造一份几何。SFC 与 preview 到期后分别隐藏，辅助几何不参与
碰撞。

原生 C++/ROS 方法通过 RPC 的可选 corridor / preview 字段传输相同对象；
Python/JAX 方法可以直接返回这些对象。是否需要 RPC bridge 取决于运行边界，而不
取决于算法名称：进程内方法不需要 bridge，独立 C++/ROS 进程才需要。

```python
from drone_playground.planning.corridors import (
    ConvexPolytope,
    SafeFlightCorridor,
    TrajectoryPreview,
)

corridor = SafeFlightCorridor(
    "candidate",
    (ConvexPolytope(halfspaces=planes),),
    generated_at=now,
    valid_until=now + 0.5,
)
preview = TrajectoryPreview(
    "backup",
    trajectory,
    generated_at=now,
    valid_until=now + 0.8,
)
```

MIGHTY 若以外部原生进程接入，可以复用 `tests/native` + RPC transport；
AllocNet 若以仓库内 JAX/Flax Policy 实现，则直接产生其声明的 Setpoint/Actuation，
不需要 bridge。只有选择了独立进程版 AllocateNet 时才需要 transport adapter。

历史已验收回放和决策证据不覆写；缺失的 SFC/preview 不从飞行轨迹猜测。
