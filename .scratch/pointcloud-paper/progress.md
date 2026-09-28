# 执行账本：.scratch/pointcloud-paper/implementation.md

2026-09-28 创建，主会话直接执行。
来源冻结：论文8页；[24] DiffPhysDrone@271936190b5c2a5e760230e718fa6b167718a5bf。
工作树pointcloud-paper，基线810f7d3，Navigation8已冻结。RTX4090空闲，磁盘3.5T可用。
独立性：现有refactor/native-planner-runtime工作树有进行中改动，本任务使用独立分支。
裁决：作者源码搜索无匹配，公开重建假设，不声明未知参数与作者一致。
裁决：保留50000更新完整目标；训练进行中的里程碑只报告实际预算与状态。
接口预检：物理/传感组件→PointCloudTask→训练与评测；PointCloudPolicy共享训练/冻结推理；
共享SceneBank几何用于训练损失、传感器、Navigation8碰撞与回放，避免场景不一致。
下一步：任务一测试先行。

基线：SHAC、深度、LiDAR共31通过、1失败。失败test_batch_environments_are_isolated_and_reset_clears_the_phase
仍假设ID0/3是不同随机几何；Navigation8的固定ID展开使两者都为S01。后续改用不同场景分组索引验证真实隔离。
任务一RED：新物理/传感模块六项明确因缺失模块/球体支持失败，见tmp/pointcloud/physics-red.log。
共享几何复核新增两条边界：轴对齐盒的平行射线有效命中、静态障碍时间导数安全。

任务一/二/训练核心：33项组件回归通过，含原深度与修正固定场景索引的LiDAR隔离测试。
新增8项物理/传感、5项网络/目标/场景、4项训练/恢复测试均经历RED→GREEN。
CPU保存恢复后的优化器、随机数、下一参数更新逐元素一致；编码器参数梯度有限非零。
GPU完整形状工程探测paper-pointcloud-throughput-seed0-v1真实3更新/15360交互，退出0，状态paused。
首次更新含编译28.858秒，稳定1.830秒/更新，独立开发损失22562.32→9.0534；仅作为工程结果。
按该吞吐50000更新约25.4小时。正式训练分可恢复阶段执行，目标预算仍为256000000交互。
下一动作：提交当前核心模块；启动正式首阶段1000更新，同时完成独立Navigation8评测与回放模块。

核心实现提交211e333，正式训练阶段paper-pointcloud-seed0-stage1-v1正在运行，session21551。
评测模块RED→实现：高速细杆、非有限/越界、参数冻结与完整分母、八场景身份四项通过。
回放测试发现rscope_io的入口校验固定动作4维，后续写入器已按实际轴数循环。
修正入口为[T,B,A>0]，保留真实三维加速度；不通过补零伪装四维动作。

独立评测复核：原500Hz递归积分使无碰撞步末状态与训练0.1秒离散映射出现0.01365速度差。
新增一致性测试先失败；改为按500Hz采样同一策略步起始状态的中间时间转移，步末精确保留训练映射。
训练代码与参数更新保持既有身份；评测明确记录10Hz离散状态转移、500Hz碰撞采样。

2026-09-28 10:15 UTC：首阶段真实完成1000更新/5120000交互，退出0、状态paused，完整目标50000更新保留。
独立静态开发损失2.3229413，选中update-0001000，参数摘要2fafd2c04579afc5bc6cab0026820496b64708a39d38dabe3551127d3ff3ad7b。
首阶段耗时1890.31秒，稳定1.83723秒/更新；这些数值属于首阶段，完整训练尚待后续运行。

最终方法测试42项通过（含物理、网络/目标、训练恢复、评测、协议、管线、原RScope记录）；
证据docs/verification/pointcloud-paper-method-tests.xml与tmp/pointcloud/method-final-verified.log。
评测新增全部回放逐条读回和XML重编译；三维动作、位姿及帧数有真实一致性检查。
代码审查发现的采样频率标签、名义协议修改、早期最佳权重/全训练预算混淆，均有显式校验和回归。
管线按首阶段评测→完整状态续训至50000→开发集选模→冻结Navigation8评测执行；失败保留并停止后续阶段。

独立审查追加：传感器source_rate_hz原先只改变校准标签，现校验同步采样频率等于策略频率；
Navigation8 nominal标签现绑定0.07米机体、0.5米到达、40秒、10Hz策略、500Hz碰撞采样与预注册24格。
tests/test_pointcloud_protocol_guards.py先10项失败，再连同两项真实配置构造共12项通过；
证据tmp/pointcloud/protocol-guards-{red,green}.log。该修复只加强配置校验，当前已启动训练的数值更新不变。
完整运行编排器的源码摘要应在本轮审查修复提交后冻结。

审查中的动作坐标/单位问题现已补丁处理：network_output_frame固定body、command_units固定m/s^2；
新增2项回归先失败，再连同前述协议检查共12项通过，证据command-contract-{red,green}.log。
独立模块配方文档已加入docs/research/pointcloud-paper-module-map.md。
首步终止回放和已有评测与训练身份绑定两个问题仍需当前评测/编排实现完成，冻结运行前再次核对。

上述两项现已补齐：首步数值失败回放增加真实t=0初态，帧数与转移数分开登记；
评测复用绑定所选检查点、参数摘要、来源训练目录和result.json摘要，恢复账本绑定阶段运行名。
single-step-replay-red.log两项失败→single-step-replay-green.log十二项通过；
evaluation-binding-red.log四项失败→review-binding-green.log二十项通过。
共享几何/任务/原模块独立回归已完成92项，耗时823.03秒，证据independent-regression.xml/log。

并发运行事实：72f8ffb冻结编排已于10:18 UTC启动，阶段Navigation8结果已完成0/24，24越界；
完整训练进程3956802于10:19 UTC从1000更新恢复，父编排3955029。
10:23 UTC核对冻结后的源码差异仅为scripts/run_pointcloud_pipeline.py及evaluation/pointcloud.py；
训练器、模型、网络、目标、传感器和配置字节保持72f8ffb版本。
当前需显式对账并接管原编排父进程，保持3956802训练不中断；旧编排内存中的源码摘要将在最终评测前触发拒绝。
保存原编排账本后采用同一完整训练的真实进程与状态恢复，记录评测/编排修复的来源边界。

2026-09-28 10:34 UTC 已完成明确对账和原训练接管。报告/回放与协调器修复提交2cd0d40；
55项最终方法测试通过，日志tmp/pointcloud/final-review-verified.log，XML在docs/verification。
旧协调器3955029定向退出，训练3956802的start_marker保持一致，交互数继续由7475200增至7577600。
恢复后的协调器使用原pipeline-v1账本和全部原运行身份；工具会话55888。
恢复凭据：experiments/paper-pointcloud-seed0-pipeline-v1/recovery-20260928T103424Z/receipt.json。
AST核对advance_checked/make_rollout/summarize_trace/build_commands保持一致；训练数值源码和配置字节保持72f8ffb。
新源码摘要760bd99f05ab6f84c059edfd5b3dbde6cb2921f7f3b2051ee094aa28f0339c24。
原首阶段24份回放有效，采用当时记录格式；后续新回放包含真实t=0初态和单独的转移数。
完整训练继续进行；最终结果需读取full-v1/result.json和navigation8-full-v1/eval/report.json实际产物。

10:38 UTC交付核验完成：3份轨迹归档摘要、24份回放及读回证明均匹配，全部终止高度超过6米；
55项最终方法测试证据匹配，完整训练同一进程继续至1600更新/8192000交互。
证据docs/verification/pointcloud-paper-stage1/delivery-verification.json；当前DP-005仍为进行中。
