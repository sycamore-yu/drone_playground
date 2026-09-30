# 回放可视化接口

回放显示三个通用对象：由真实标定生成的传感器视场、带接收时间和有效期的规划轨迹、算法实际输出的凸多面体安全飞行走廊（SFC）。世界坐标为右手z向上，长度用米；传感器外参相对机体FLU。显示层仅用于回放，不参与观测、奖励或动力学。

`export_rollout(..., visualization=ReplayLayers(...))`是统一导出接口。导航模型从实际传感器标定自动提供视场；新原生决策归档自动提供Trajectory及可选PlannerGeometry。模块链保留中间阶段的规划输出，不能只记录最后的Motion Cmd。Python方法直接提供相同对象；C++方法通过可选Protobuf字段传输，不依赖ROS。

凸多面体支持顶点或半空间`A x + b <= 0`。轨迹沿既有Trajectory合同采样，仅显示当时已收到且仍有效的计划；未规划、清空及过期状态不显示未来轨迹。SFC输出标注候选／备用角色及来源，不用实际飞行路径猜测缺失走廊。SUPER桥接从其原生MarkerArray取得真实走廊；EGO的B样条已转换为Trajectory，可直接显示。

静态视场边界附着在机体上；动态规划与SFC边线用MuJoCo空间线段及mocap端点存入标准mj_unroll。原RScope无需修改。绘制辅助物不参与碰撞；原无人机、障碍、时间、动作及任务指标保持原始数据。MID360绘制范围默认截到10m以便观察，真实100／200m量程完整记录；这是显示裁剪，不是传感量程变更。深度相机显示其完整10m视场。视场表示理想几何包络，不表示遮挡后的实际回波。

历史报告、权重和已验收回放不覆写。增强回放从冻结轨迹／决策复制生成，单列来源摘要。没有保存完整规划决策或SFC的旧回放无法还原对应图层；新SUPER工程记录单独验证采集，不增加质量通过格，也不替换原验收的原生运行。

验证覆盖外参及俯仰、360°雷达包络、计划切换与过期、凸多面体、决策归档、C++→Python协议、标准RScope读回和真实MuJoCo渲染。新方法如MIGHTY／AllocNet只需输出相同规划与走廊对象；这不代表其算法已移植或达到质量门槛。

## 使用

新 `eval` 回放自动带传感器视场；原生／模块链回放还读取同回合 `decision-trace`。`evaluation.record_planner_visualization=false`可关闭SUPER的原生SFC采集；完整Trajectory仍会记录。SFC采集可能改变异步规划调度，不能把新工程记录作为原质量运行的重放。SUPER候选／备用图层使用ROS回调收到的仿真时间和0.5秒显示有效期，不声称它们就是已提交轨迹的安全证书。

```python
from drone_playground.native.geometry import ConvexPolytope, PlannerGeometry, SafeFlightCorridor
geometry = PlannerGeometry(
    generated_at=now, valid_until=now+0.5,
    corridors=(SafeFlightCorridor('candidate', (ConvexPolytope(halfspaces=planes),)),))
```

`planes`为 `[M,4]`，约定 `A x+b<=0`；也可传入世界顶点 `[N,3]`。Python阶段在回复中提供`planner_geometry=geometry`；通用C++ SDK实现可选的`Algorithm::Geometry`。轨迹使用既有可执行Trajectory，额外预览使用`TrajectoryPreview`。MIGHTY／AllocNet只要适配这些物理数据就可复用显示，不需编写专用viewer。

历史可信本地文件可用工具复制增强，源文件不会被覆写：

```bash
JAX_PLATFORMS=cpu pixi run python scripts/tools/enhance_replay.py \
  --source /path/to/original.mj_unroll --output experiments/tmp/260930/new-enhanced \
  --calibration /path/to/actual-calibration.json \
  --decision-archive /path/to/same-episode/decision-trace
```

视场数据用`SensorView`统一表达，转换入口为`sensor_view(actual_calibration)`；不能拿默认D435参数替代深度飞行相机的实际俯仰／标定。轨迹图层以首个收到该计划的时刻显示，只绘制当前时刻至有效终点；模块链按阶段命名。每个带规划的文件只包含一个回合，避免把一个回合的计划画到其他初态上。

蓝色为理想传感视场、绿色为规划轨迹、橙色为SFC。MuJoCo默认显示辅助几何所在的组2，空间线段按mocap逐帧更新，RScope无需增加插件。`replay-visualization.json`只记录标定、图层和来源摘要；可直接播放的几何坐标在原生mj_unroll及其配套模型中。

## 本轮工程证据

[示例目录](../../experiments/tmp/260930/replay-visualization-20260930/README.md)提供深度、点云、SUPER和EGO的直接播放入口及图片。SUPER三个实例分别记录1165／1180／1267帧走廊，模型分别有270／324／318条动态边线；EGO每段63条轨迹边线。在4秒时，显示轨迹首点与归档的真实多项式采样误差均为0，六段可用标准RScope读取，详见[验证凭据](../verification/replay-visualization.json)。这六段工程采集不加入质量成功率。

选定结果的642段历史回放已生成视场增强副本：深度600、点云8、SUPER26、EGO8。位置、姿态、时间、观测、奖励和指标逐项读回相等；各自main_result目录的`enhanced-replays/`可直接访问。缺失规划／SFC的部分不补造，完整新图层见独立工程例子。完整回归556项＋5子测试通过，后续协议检查另行通过。
