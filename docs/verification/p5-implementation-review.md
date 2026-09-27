# P5-05/06 实现与工程验证

基线：76724ec（P5-04）。执行：p5-main 主会话，未派出 dsh 或子代理。
正式 8×8388608 训练及 36 格留出结果另由矩阵报告记录，本文件不替代正式结果。

## D.VA

- 官方锁定 01b2be4986a0851a952aa860afb4a5958e6676e2；Brax 网络/分布、JAX rollout、Optax 优化器。
- actor 的整个输入 stop_gradient，编码器参数可更新；动作→控制→动力学→奖励保持导数。
- critic 与 PPO 一样只取观测中的 20 维本体/目标/上一动作。传感器块不进入 critic。
- 自动重置前的 terminal_proprioception 保留末端价值导数；真实终止掩码，时间限制 bootstrap。
- critic 学习率按训练更新而非内部 16 次 Adam 步衰减；完整状态包括优化器、目标网络、归一化、环境、PRNG。
- 恢复校验全部训练/环境语义；运行目标路径与墙钟等非数值设置可改变。继承恢复步以前的开发选模，排除未来评测。
- 几何距离的零向量范数选择零次梯度，前向值不变。未截掉整个奖励/动力学梯度，也未用 NaN 替零掩盖失败。
- 正式共同网络采用 tanh_normal、关闭全输入 running normalization；稀疏传感器值由物理标定归一化。与此前 P5-04 工程配方的变化已保存在 git diff，八个正式单元一致。

|工程运行|交互|退出|结果|
|---|---:|---:|---|
|p5-static-depth-dva-seed0-v1-eng|未满预算|1|第 5 更新出现盒体内部距离导数 NaN，保留|
|p5-static-depth-dva-seed0-v2-eng|262144|0|真实 actor/critic 更新，3 次开发快照|
|p5-static-lidar-dva-seed0-v1-eng|262144|0|真实 actor/critic 更新，3 次开发快照|
|p5-static-depth-dva-seed0-resume-eng|续 131072→262144|0|完整状态与开发选模恢复|

CPU 数值恢复对照通过；GPU 跨进程的策略最大绝对差 0.000169534、critic 0.000664396。
环境及累计统计有分歧，因此只声明 GPU 近似恢复，不声明逐位等价。
原始量化见 [p5-dva-gpu-resume.json](p5-dva-gpu-resume.json)。

37 项导航/D.VA/原生接口测试通过，额外完整固定观测代理目标有限差分通过；
增加回放场景索引检查后的原生契约 9 项通过。全量 suite 148 项和 2 子测试通过，exit 0。
完整命令/日志身份见 [p5-tests.json](p5-tests.json)。

## 原生规划器

EGO bfda51284c8c1b476043255a8145ef925a3778a5、SUPER 2ad3419c127a617c6d7df6925e81a14175a9c096。
运行在现有 Noetic 容器 flightbench 的 /tmp/p5-native 独立构建目录；不修改其原有 FlightBench 工作区。
构建用上游 ROS1 模板；SUPER 只编译 fsm_node 运行目标，日志查看 GUI 不属于本项目依赖。
libdw/libelf 开发包提取在私有 sysroot，没有覆盖容器系统库。

- Pixi 主进程不导入 ROS；JSON 行协议连接 ROS worker。
- 每回合独立 master/map/FSM/轨迹进程；端口占用直接拒绝，结束回收其子进程。
- worker 脚本按内容摘要冻结，每次运行保存实际 launch/YAML。
- /clock 只使用仿真时间；真实 ROS 计算耗时单独计量。无有效或过期参考时保留当前位置。
- EGO 消费完整 120×90 32FC1 轴向米制深度与真实 optical pose/intrinsics。用预设三维航点，
  避免上游手动点击接口强制 z=1；原生 traj_server 的 time_forward=1 是 yaw 前视参数。
- SUPER 消费完整 24000 射线窗口中的有效 world XYZ + intensity。保留原生 ROG-Map 和规划/backup 逻辑。
- 学习策略使用深度 300 像素或 LiDAR 120 点、4 帧历史；规划器使用原始帧。该输入处理差别明确记录，
  不将跨传感器/跨方法比较声称为所有输入带宽完全一致。
- 两规划器最大速度 2 m/s、最大加速度 3 m/s，轨迹经同一 PD+加速度前馈外环转成姿态/推力，
  再用现有 Crazyflow attitude 控制与 first_principles 动力学。
- 静态/动态使用相同任务、几何、种子库、40 s、0.5 m、0.07 m 机体碰撞规则；未添加动态预测算法。
- 回放只记录控制需要的本体量；原始测量首帧/运动后采样随 native 目录保存。在线时序指标单独报告。

|工程运行|回合|成功|说明|
|---|---:|---:|---|
|p5-static-ego-v3-eng|3|3|easy/medium/hard，28 条原生轨迹|
|p5-dynamic-ego-v2-eng|3|2|hard 碰撞保留，40 条原生轨迹|
|p5-static-super-v3-eng|3|3|463 次原生轨迹发布|
|p5-dynamic-super-v1-eng|6|6|两种动态几何子类，729 次发布|
|p5-static-ego-parallel-eng|6|6|独立双 master，67 条原生轨迹|
|p5-static-super-parallel-eng|6|6|独立双 master，807 次发布|

EGO v1 的场景 JSON 序列化错误、v2 的硬编码目标高度、SUPER v1 的早期目标订阅竞态均保留。
SUPER v1 还跨过一次 worker 更新，只作为诊断，不进入任何正式矩阵。

## 代码审阅

规范：沿用可组合配置、外部 ROS 进程、纯环境状态、独立运行目录、已锁定依赖；
未创建第二套训练库、查看器或无人机动力学。原生代码与许可证保留在外部构建树。
审阅修复了场景 ListConfig 无法序列化、native worker 运行中变化、回放选中失败案例却丢失场景索引等问题。

规格：8 学习配方和 4 规划器配方齐备；实际更新、导数、终止、恢复、真实原生求解已有工程证据。
预算与正式 dev/heldout 完成度由队列和矩阵汇总脚本核验，尚未完成的结果不填成零或通过。
用户没有冻结成功率数值门槛，quality_passed 保持 null；工程通过、预算完成、策略质量分别呈现。

随后发现 P5-01 地面只参与感知而未进入碰撞；已将地面并入统一最近净空/球碰撞，
修复后 19 项导航回归通过。全量 suite 早于该改动加载代码，因此两份证据分列。
刚启动的 v1 队列中止，首个 PPO 最后已记录 2097152 交互，两个原生 dev 运行未完成；
这些目录的 `interruption.json` 明确排除正式统计。v2 从零开始统一使用修复后协议。
