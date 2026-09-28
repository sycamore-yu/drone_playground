# 当前进度

更新时间：2026-09-28 09:46（北京时间）。P1/P2已由用户验收；P3/P4及本轮可组合架构/LOTF交付完成。
当前分支：`implementation/composable-lotf`；基准`bd14468`。
本轮迁移前47项基线、五模块回归与LOTF上游对照已完成；最终79项测试、2个子测试通过，
命令、日志和统计见[最终检查](verification/composable-lotf-final-checks.json)。

## 当前工作

2026-09-28 场景复核新增决策：用户取消后续正式 benchmark 的随机密度生成场景。
原因是已观察到两类退化：Easy 圆柱森林可只有两根且不遮挡起终点直线；高密度实例又可能
缺少合理飞行通道。现已建立待人工确认的固定编号候选集 S01-S06、D01-D06，
目录 configs/scene/p5_fixed_catalog.json。所有候选均要求直接起终点航线被阻挡，同时有一条
最小机体净空不低于 0.35 m 的离线检查路径；该路径只用于可行性验收和 RScope 回放，不暴露给方法。
12 个独立检查回放位于 experiments/p5-fixed-scenes-review-v1/，动态障碍在回放中使用橙色，
静态障碍为灰色。NavRL/P2M 只作为障碍尺寸、形态和动态 clutter 设计参考，不复制其随机生成器。
用户确认场景前不启动基于新场景的训练；原 v2 结果保持历史证据，不覆盖。

第二轮人工检查后又增加 `p5-fixed-candidate-v3-density-boundary`：原 100 x 40 m 候选在
放大地图后障碍密度不足，而且两侧可沿空白带绕过。v3 固定每个 Easy/Medium/Hard 场景为
50/100/150 个场内障碍，另加四面共享物理边界墙；墙同时进入传感、碰撞和回放，不是隐藏终止条件。
12 个 v3 RScope 审阅文件位于 `experiments/p5-fixed-scenes-review-v3/`，最小已知检查通道净空
0.646 m，12/12 直接起终点线均被障碍阻断，所有 scene XML 均重新编译通过。外部场景来源与
SUPER PCD 语义见 `docs/research/p5-scene-reference-comparison.md`。仍等待用户视觉确认后再冻结新 benchmark。

2026-09-28 阶段性收尾：用户要求整理交接并协助处理问题。本轮按此要求完成状态核验与证据整理。
代码基线为 `4c91ed7486f6050d5cef6a3dfb6522822f440e57`，原会话标识 `p5-main`。
01:44 UTC 进程核验：P5 训练、补采和 ROS 规划进程均已退出；RTX 4090 利用率 0%、显存 146 MiB。
现存 RScope 查看器 PID 3216397，由用户编辑器启动，继续保留。

P5-05/06 工程实现完成；P5-07 八个正式训练单元全部完成，各 8388608 次，总计 67108864 次交互。
原学习队列于 20:29:46 UTC、原生规划器队列于 20:27:30 UTC 完成；两个 queue-state.json 均记录全部单元 completed。
最终独立开发 1152 回合、留出 4608 回合，分别覆盖 36 格。
01:46 UTC 重新执行 `python3 scripts/summarize_p5.py --revision v2 --require-complete`，
退出码 1；日志 `tmp/p5/handoff-integrity-20260928.log` 确认训练预算和评测分母齐全，交付完整性仍为 false。

P5-08 阻塞已定位：学习补采直接读取原评测 `manifest.json.config`，其中 `checkpoint=null`；
`NeuralPolicy.load` 因此在 `Path(None)` 报 TypeError。原评测 `eval/report.json.checkpoint`、
原 `command.txt` 和开发集 `best.json` 均可定位实际已保存权重。
配置解析函数 `evaluation/execution.py:resolve_evaluation_config` 从权重元数据复制训练配置，
评测记录因此保留了训练时为空的 checkpoint；补采函数误将它视为可直接执行的评测配置。
首个补采目标目录尚未创建；失败日志保存在 `experiments/p5-archive-repairs-v2/`。
学习补采失败后收尾脚本完整性检查退出，旧进度文档的运行状态已经过期。

尚缺静态深度 PPO、静态深度 D.VA、静态 MID360 PPO 的开发/留出逐帧归档：
共 6 个评测运行、18 格、1440 回合。四份静态原生规划器归档补采已完成。
场景隔离审计已通过，训练/开发/留出间的几何与运动指纹交集均为 0。
全 5760 条归档及 72 个代表格仍待最终读回；现有 replays.json 仅覆盖 30 格，图表作为历史部分结果保留。

当前留出结果：动态 MID360 PPO 22/384，其余七学习单元均 0/384；
四个 D.VA 单元均为 384/384 越界。低分机制仍待诊断，质量门槛保留 null。
报告所选完整归档结果为静态 EGO 324/384、SUPER 377/384，动态 EGO 199/384、SUPER 379/384。
静态规划器原始结果为 EGO 328/384、SUPER 375/384，两套身份与差异分别保存，统计按固定归档规则选择。

接续顺序：从原评测报告恢复冻结权重路径并核对摘要 → 补采六个缺失评测 →
重建汇总、图表、全量归档及回放核验 → 单独研究学习策略低分。
先保存原补采错误日志的副本，再运行可能覆盖同名日志的队列；沿用原权重、种子、协议与预算。
当前恢复对象是归档评测；原八训练及四规划器原始矩阵已经完成。
完整交接文档：`/tmp/P5-阶段性收尾与接续说明-20260928.md`。

### 2026-09-27 实施记录（历史状态，以以上核验为准）

P5 按 [感知导航规格与执行计划](../.scratch/p5-navigation/spec.md) 进入实现。
2026-09-27 16:47 UTC 主会话直接续做 P5-05—08（用户要求不启动 dsh）。
实现提交：`4456dd2`。两种 D.VA 各 262144 工程交互 exit 0；CPU 完整状态恢复和 GPU 跨进程恢复已运行，
GPU 策略参数最大差 1.695e-4，环境轨迹可分歧，仅标为近似恢复，见 `verification/p5-dva-gpu-resume.json`。
静态 EGO/SUPER 各 3/3，动态 EGO 2/3（hard 碰撞保留），动态 SUPER 6 回合完成；并发隔离工程运行通过。
16:55 UTC 自审发现 P5-01 的地面已渲染/感知，但碰撞只检查障碍物，参考点越界前的机体触地未判失败。
两条 v1 正式队列已中止（原 session 53875 / 52924），保留三份已启动运行及 `interruption.json`；
它们不进入正式统计，也不续训。新增地面净空/碰撞回归通过后，从头执行统一协议 v2。
v2 队列命令为 `scripts/run_p5_{learning,native}_matrix.py --revision v2`；
日志与队列状态进入 `experiments/p5-matrix-v2/`、`experiments/p5-native-matrix-v2/`。
先查原进程/状态，禁止重复启动或覆盖失败目录。
v2 已启动：学习 session 96099、规划器 session 27701（16:58 UTC，代码 `caf4c49`）。
17:47 UTC：静态 depth PPO/D.VA 均完成 8388608 交互及各 96 dev、384 heldout；
两个学习单元 heldout 到达均为 0。静态 EGO dev 86/96、SUPER dev 95/96，heldout 在运行。
当前学习单元为 static/lidar/PPO。最终表随运行完成自动重建：
`docs/verification/p5-results-v2/`；观察会话 21195，收尾会话 94038，
收尾日志 `tmp/p5/finalize-v2.log`，仅全部预算/评测通过完整性检查后生成最终图表与回放核验。
回放终止帧修复已提交 `7e4767f`，仅改变导出，实际 D.VA 六个独立评测格无填充帧；
早期静态 PPO/规划器旧回放保留，不改写原始文件。
18:08 UTC：第三个学习单元 static/lidar/PPO 已完成，static/lidar/D.VA 在运行。
核对规格 11 后补齐全回合逐帧归档，提交 `ffcbe8a`；新增 2 项归档测试通过。
早期缺归档评测按固定缺失规则在 `<原运行>-archive-v1` 重评，原结果保留，新训练交互为零。
原生补采 session 80408、学习补采 session 28348（等待全部训练完成后使用 GPU）；
日志 `tmp/p5/archive-{native,learning}-queue-v2.log`。最终矩阵按有完整归档的评测身份汇总，
必须额外核对 `archive-repair.json`、全部 5760 轨迹与新增退出状态。
收尾 session 94038 原先只等待 32 个初始运行，可能因补采尚未结束而退出非零；
应在补采完成后重新执行汇总、绘图、回放验证三命令，不能据此跳过完整性门槛。
18:36 UTC：四个静态训练单元均完成预算及原独立评测；动态 depth/PPO、EGO、SUPER 已运行。
原始评测累计 dev 576、heldout 2304；补采替代结果另由汇总选择，不能重复计入分母。
静态 SUPER 原 heldout 375/384（5 碰撞、2 越界、2 超时）；其补采 dev 95/96，
原 dev 的 1 次超时变成补采的 1 次碰撞，两份原始结果与差异都保留。
任意归档案例导出与 RScope 读回通过：easy/31 共 550 帧，位置误差 0。
静态 3 秒原生输入重建：深度 90×120、LiDAR 2803 有效点，误差均为 0。
仍需动态输入重建核验；全部八训练 manifest 出现后执行 `verify_p5_scene_splits.py --revision v2`，
该逐实例隔离审计现在也是汇总的强制门槛。最后完整命令顺序见 runbook。
18:45 UTC：动态原始输入重建核验完成；四个任务/传感器组合在 3 秒时刻误差均为 0，
动态 LiDAR 为 5984 个有效点。当前 288 条完整归档和 30 个代表回放已读回通过，
最终仍须全矩阵 5760 条归档及 72 个代表格完成验证。
19:12 UTC：第五个训练单元 dynamic/depth/PPO 完成 8388608 交互及独立评测；
heldout 0/384 到达、74 碰撞、310 越界。dynamic/depth/D.VA 已启动（第六单元）。
动态 EGO dev 49/96（47 碰撞），动态 SUPER dev 94/96（1 碰撞、1 超时），两者 heldout 在运行。
原始独立评测累计 dev 864、heldout 2688；静态规划器归档补采已进入 medium。
学习归档补采会等原八单元训练及全部独立评测结束，不占用其训练 GPU；未新增训练预算。
场景审计等待会话 32061 已启动，八个真实训练 manifest 齐备后自动核验；日志 `tmp/p5/scene-audit-v2.log`。
全量 CPU suite exit 0：148 项、2 子测试通过（`tmp/p5/full-suite.log`）；
该进程早于地面改动加载源码，补充修复后导航 19 项及原生接口/回放身份 9 项通过。
质量数值门槛未冻结，保留 null，不阻塞预算执行。
P5-05 已修正超时前状态 bootstrap、critic 学习率、完整恢复配置校验及盒体内部距离零点次梯度。
P5-06 锁定 EGO 与 SUPER 已在 flightbench 容器独立 `/tmp/p5-native` 构建运行目标；
独立 ROS master、JSON 进程桥、真实传感器、轨迹控制器及四实验配方工程闭环已通过。
当前会话 `p5-main`；正式八训练和 36 格留出评测尚未完成，质量门槛仍待用户明确。
用户确认八个训练单元各8388608次交互、推力与姿态动作、动态场景评测原生EGO/SUPER；
场景复用SANDO/MIGHTY几何与运动，导航统一40秒、0.5米到达、机体碰撞判失败。

P5-00 与 P5-01 已交付并本地提交：

| 任务 | 交付 | 证据 |
|---|---|---|
| P5-00 来源与协议 | 锁定依赖、许可与来源身份，记录已确认协议与待冻结参数 | [P5来源清单](verification/p5-source-inventory.json) |
| P5-01 场景与导航任务 | 解析几何场景、密度驱动生成、统一导航任务、独立评测与回放 | `tests/test_navigation.py` 17项通过 |

场景按 SANDO 的“占据面积密度”定义难度（0.05/0.10/0.20），元素尺寸沿用源值
（森林圆柱 1.0–1.5 米、立柱 0.4×0.4×4.0 米、横杆 0.4×4.0×0.4 米、立方体 0.8 米），
动态元素使用源 trefoil 方程与 GLOBAL_TIME_SCALE。走廊 20×10×5 米、导航距离 15 米、
起终点高度 2 米；机体碰撞取 Crazyflow 自带 0.07 米球。圆柱高度按 5 米天花板截断、
trefoil 垂直幅度按走廊缩小，这两项偏差与其余来源差异逐条记入来源清单。

两个真实闭环运行（`p5-nav-smoke-static-v3`、`p5-nav-smoke-dynamic-v1`）各实际更新
32768 交互，开发集每档难度 32 回合逐回合记录，每档导出 4 份 rscope 回放，退出码均为 0。
随机初始策略成功率 0，作为真实低分保留：本阶段交付的是工程链与事件协议，策略质量属 P5-07。

P5-02 与 P5-03 已交付：D435 理想深度链与 MID360 点云链，共享同一套解析图元射线。

| 任务 | 交付 | 证据 |
|---|---|---|
| P5-02 D435 深度链 | 相机模型、内外参、四帧历史、逆深度观测、25 Hz 节拍 | `tests/test_depth_sensor.py` 15项；与 `mujoco.mj_ray` 命中集合零分歧 |
| P5-03 MID360 与吞吐 | 复用 MuJoCo-LiDAR MID360 图案与窗口相位、120 点策略帧、世界系点云 | `tests/test_lidar_sensor.py` 12项；`docs/verification/p5-throughput.json` |

**后端探测结论**：本机无 GL 上下文，原生 MuJoCo 渲染不可用；MJX 3.14 批量深度渲染需要
未安装的 `warp-lang`，且 `mjx.create_render_context` 对全部 world 共用一个 model，
无法表达逐实例几何。因此按规格第 6 节第 4 步，深度与点云的训练后端统一冻结为
本项目对场景图元的解析批量射线，并以 `mujoco.mj_ray`、MuJoCo-LiDAR 自带
`MjLidarJax` 两条独立路径做几何 parity。

两种传感器暴露**完全相同数量**的策略输入（20 维本体 + 2400 维传感 = 2420），
策略输入维数一致。同一传感器的 PPO/D.VA 共享编码器配方；跨传感器仍有编码器结构、
参数量、视场和历史覆盖时长差异，不能称为网络完全一致或传感器单因素消融。

真实深度闭环运行 `p5-static-depth-ppo-seed0-v1-eng`：1048576 交互、
`training/sps=21795.7`、退出码 0。该运行暴露了一个奖励缺陷并已修正：
原失败代价 -20 小于最大进度收益 75，撞毁回报高于安全悬停；现冻结为
失败代价 = 2×最大进度收益（-150），并新增顺序断言（成功 > 安全超时 > 任何失败）。
深度与动态导航的低分结果按实际保留。

依赖变更：新增 `mujoco-lidar==0.3.5`（MIT）并写入 `pyproject.toml` 与 `pixi.lock`。
许可记录：`reference_repos/SUPER` 根目录没有 LICENSE；源文件头包含 LGPLv3-or-later 声明。
本轮使用未修改的外部构建与进程，具体来源和许可线索见来源清单与运行手册。

P5-04 已交付：共享 encoder 接口与 actor/critic 契约（`learning/perception.py`）。
两种传感器的 actor 头形状相同，编码器分别为 CNN 和点集编码器。
P5-05 正式配方中 actor 使用全部 2420 维，critic 仅取其中 20 维本体/目标/上一动作，无额外特权字段；
checkpoint 保存/重载逐元素复现动作。P5-04 时 `tests/test_perception_ppo.py` 7 项通过。
工程预算真实运行：深度 `p5-static-depth-ppo-seed0-v2-eng`（262144 交互，sps 3328）、
点云 `p5-static-lidar-ppo-seed0-v1-eng`，均退出码 0 并写出检查点与逐难度评测。

P5-05/06 工程实现已完成，当前继续 P5-07/08 正式矩阵与交付。

已按用户本轮意见写入 [可组合架构规格](../.scratch/composable-flight/spec.md)：
控制器合并跟踪与飞控内环；策略包含轨迹规划；动力学包含电机；训练配置收敛为四类。
LOTF范围明确为高保真前向＋简化反向＋原生BPTT，在线残差学习和策略交替更新列后续扩展。
按 [模块地图](../.scratch/composable-flight/map.md) 协同实现，最终交付悬停与八字的训练结果。
五个模块工作单均已解决。LOTF原始源码作为固定子模块保存在`third_party/learning_on_the_fly`，
悬停600万、八字2250万次训练交互均完成，各保存9个初始/中间/最终策略；开发和留出评测在独立进程执行。
实际计划与账本见 [实施计划](../.scratch/composable-flight/implementation.md)。
参考证据在 [开源编排比较](research/composable-platform-references.md)。

| 当前交付任务 | 开发集完整回合 | 留出集完整回合 | 留出全程位置RMSE | 留出最后一秒RMSE |
|---|---:|---:|---:|---:|
| LOTF悬停，3秒 | 32/32 | 128/128 | 0.424420米 | 0.076966米 |
| LOTF八字，5秒 | 32/32 | 128/128 | 0.184752米 | 0.164008米 |

悬停全程含随机初态的收敛过程。最终复核另跑两组各128留出，参数摘要和逐回合RMSE与原报告完全一致。
共同交付见[训练结果与模型](verification/composable-lotf-delivery.md)；
GPU续训的4.59e-6最大参数差及CPU逐元素一致的不同验证范围见[恢复核验](verification/composable-lotf-resume-check.json)。

当前可直接在RScope Viewer打开：

```text
experiments/lotf-hybrid-hover-seed0-v1/independent-heldout/rollouts/
experiments/lotf-hybrid-tracking-seed0-v1/independent-heldout/rollouts/
```

## 阶段结果

| 阶段 | 实际完成 | 质量结果 |
|---|---|---|
| P1/P2 | 已验收；四组跟踪策略与原证据保留 | 四组均128/128 |
| P3 | 两任务×三算法×四动力学，共24格；复用4组P2 | 21格达标，3格有效低分 |
| P4a 优化控制复现 | 每步真实优化；每方法128个原生扰动试次 | 采样MPC128/128；AttitudeMPC117/128，11碰撞 |
| P4b 竞速学习 | PPO/APG/SHAC各完成声明预算及独立32+128回合 | PPO/APG均128/128；SHAC0/128 |

总计29项实验完成、25项达到当前门槛。最终事实入口为
[实际结果表](verification/p3-p4-results.md)与[逐项校验JSON](verification/p3-p4-results.json)。
正式训练交互累计41,156,608次（含复用P2），独立开发/留出评测累计4,576试次。
SHAC竞速初始策略被开发集选中，属于训练质量未达标；最终训练权重、曲线和失败轨迹完整保留。

## 直接查看

在VS Code的RScope Viewer中点击以下目录内的 `.mj_unroll`：

```text
experiments/p4-racing-ppo-first_principles-seed0-v1/independent-heldout/rollouts/
experiments/p4-racing-apg-first_principles-seed0-v1/independent-heldout/rollouts/
experiments/p4-racing-shac-first_principles-seed0-v1/independent-heldout/rollouts/
experiments/p4-racing-attitude-mpc-heldout-v2/rollouts/shard-0/
experiments/p4-racing-sampling-mpc-heldout-v2/rollouts/shard-0/
```

每文件实际保存4–5个固定/最差试次，所有128试次在对应报告中。
查看训练结束的SHAC策略使用同运行 `rollouts/step-0000655360/`；留出目录显示开发集选中的策略。
TensorBoard沿用 `127.0.0.1:6006`；三算法的完成率、门进度、损失/梯度标签已实际读取。

## 故障与范围

P3随机样条PPO拖曳模型v1与降低学习率v2均发生非有限参数。诊断捕获到接近欧拉角奇异区域后
角速度达到8.119e26，溢出观测二阶矩与价值损失。v3恢复原学习参数，显式启用数值失败边界，
异常步计失败并重置，普通步保持原行为；最终4194304次交互、留出128/128、误差0.01311788米。
旧运行及捕获数据保留，协议差异见[数值诊断](verification/p3-numerical-diagnosis.md)。

竞速优化早期v1误读扰动配置层级，已停止并作为诊断保留。正式v2使用 `env.disturbances`，
合并时验证30000..30127每个种子恰好出现一次。P4比较作者固定样条上的闭环控制与过门，
自由最短时间规划、MID-360导航和多训练种子分别进入后续任务。

## 验证与交接

- [完整工程检查和来源](verification/p3-p4-engineering.md)
- [P3/P4交付记录](verification/p3-p4-delivery.md)、[五种方法实际VS Code交互](verification/p4-editor-verification.json)
- [P3原矩阵回放](verification/p3-replay-verification.json)、[修复单元回放](verification/p3-recovery-replay-verification.json)
- [P4全部14份回放](verification/p4-replay-verification.json)、[实际竞速画面](verification/p4-visual-check.json)
- [阶段地图](../.scratch/drone-platform/map.md)、[实施方案](design/p3-p4-implementation.md)、[运行手册](runbook.md)

`p3-matrix.json`保留初次队列23/24的历史；最终结果表包含v3恢复后的24/24。
47项测试通过；62个回放文件、308条保存轨迹逐帧还原误差为0，原文件摘要保持一致。
严格独占资源性能排行及峰值显存测量仍为后续待测项；本轮计时保留共享服务器的实际口径。
原DSH委派因配额失败，独立ChatGPT审查因浏览器启动失败，均未执行代码；实现、测试及收尾由当前主会话完成。
本轮结束后无LOTF正式训练等待完成；已保存结果可直接回放和重评。
后续按总体P5/P6推进感知导航、多训练种子及统一性能口径；LOTF在线适应属于另立范围。
SHAC历史低分按用户判断暂时保留为后续研究项，当前不扩大调参任务。
