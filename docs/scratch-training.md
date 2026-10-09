# Depth / LiDAR 从随机初始化训练

**历史实验记录（2026-10-09）。** 下文记录的运行路径、进度和命令对应 v1 训练状态；
当前程序只支持完整的 v2 状态与冻结策略。旧成功率只代表当时冻结评测，不能用于说明
当前代码已经重新收敛。新实验须重新创建输出目录和训练状态，不接受旧 v1 的 `resume`。

2026-10-09 新增的训练批次。三个算法均为 PPO / APG / SHAC，每种感知分别运行种子 0/1/2，
共 18 次独立训练，对应 12 个静态/动态 Navigation 验收单元。

每次运行的 `initial_checkpoint` 为 `null`：Actor、Critic、优化器与随机数从本次种子初始化。
检查点恢复只继续本次运行的状态；本批次没有导入 `research/` 权重。
结果不与此前初始化配方的成功种子混用，收敛门槛仍为 C1–C6。

## 当前状态与配方

- [从零训练验收表](../results/acceptance/scratch.md) · [完整证据](../results/acceptance/scratch.json)。
- [Depth 配方](../src/drone_playground/configs/experiment/navigation_depth_scratch.yaml)：CNN/GRU192、batch128。
- [LiDAR 配方](../src/drone_playground/configs/experiment/navigation_lidar_scratch.yaml)：PointNet/GRU192、batch64。
- 两者均保留 500 Hz Crazyflow、10 Hz 方法执行、32 步 rollout、完整设备采样与声明的策略预处理。
- 八场景轮换，每场景 20 次更新；高度误差权重 1.0。
  Depth 的净空 margin 1.5 m / 权重 8；LiDAR 的 margin 2 m / 碰撞权重 8 / 最低接近速度权重 1。
- 每 160 次更新执行八场景完整 checkpoint evaluation，连续三次主场景通过后独立冻结。
  每 80 次更新保存恢复状态；没有更新数或时长上限。

选定运行路径由下方清单固定，包含尚未启动的种子。早期失败配方另列，不混入选定种子。
运行中的状态只表示采样与更新已经开始，不能代替收敛验收。

## 复现

```bash
pixi run drone-playground -m mode=train experiment=navigation_depth_scratch_safe \
  learning=shac seed=0,1,2 \
  'output=results/scratch_depth_reproduce_${learning.algorithm}_s${seed}'
pixi run drone-playground -m mode=train experiment=navigation_depth_scratch_ppo_asymmetric \
  learning=ppo seed=0,1,2 'output=results/scratch_depth_ppo_asymmetric_reproduce_s${seed}'
pixi run drone-playground -m mode=train experiment=navigation_lidar_scratch_safe \
  learning=apg,shac seed=0,1,2 \
  'output=results/scratch_lidar_reproduce_${learning.algorithm}_s${seed}'
pixi run drone-playground -m mode=train experiment=navigation_lidar_scratch_safe \
  learning=ppo seed=0,1,2 '++learning.options.critic_uses_sensor=true' \
  'output=results/scratch_lidar_sensor_critic_reproduce_s${seed}'
```

当时的运行使用所记录的 `resume` 路径。当前版本不能恢复这批历史 v1 训练状态；
下文路径用于来源与成本审计，而不是可执行的当前恢复命令。
上方 Hydra grid 默认串行；当前执行按显存余量并行调度，竞争条件和墙钟时间须一起记录。

## 已记录的配方调整

APG-Depth 基础配方种子 0 在第 160 次更新时，S03 为 11/25、D03 为 24/25，
其余五个主场景为 25/25；第 320、480 次检查八场景均为 25/25。
第 640 次检查再次通过，选择该权重；独立冻结八场景全部 25/25。
该种子完成 2,621,440 次交互、640 次更新、2,988.22 s 活动墙钟时间。三种子的完整结果见后文。

早期 PPO-Depth 在第 160、320 次检查八场景均未成功，主要为约 2 s 内高度下越界；
LiDAR APG 的第 320 次 S01 轨迹主要在 z≈6 m 上越界，SHAC 在首次检查主要下越界。
这些是实际失败结果，不使用已训练参考权重替代。

| 早期配方 / 种子 0 | 状态 | 已完成更新 | 恢复入口 |
|---|---|---:|---|
| Depth PPO 基础 | 保存后中断 | 604 | `results/scratch_depth_ppo_s0/checkpoints/latest.training.zip` |
| LiDAR APG 基础 | 保存后中断 | 412 | `results/scratch_lidar_apg_s0/checkpoints/latest.training.zip` |
| LiDAR SHAC 基础 | 保存后中断 | 304 | `results/scratch_lidar_shac_s0/checkpoints/latest.training.zip` |

新增 `navigation_depth_scratch_safe` / `navigation_lidar_scratch_safe`，重新从本次种子的随机
权重开始。`failure_cost=160` 为首次非成功结束增加额外训练代价，加上原有 20 共 180；
PPO 初始 log_std 从 −1.5 改为 −3，以减小初始控制噪声。物理执行、Task 与验收门槛不变。
APG-Depth 继续采用基础配方；Depth PPO/SHAC 与全部 LiDAR 选用新增配方。

LiDAR 的随机 PointNet 首层初始化方差尺度从 1 改为 0.01，即标准差变为原来的 0.1；
后续层保持尺度 1。它是初始化规则，输入仍为米，不是在每次前向再缩放点云。
已有检查点的动作、记忆和辅助输出与变更前代码逐数组精确一致，核对日志位于
`tmp/ros-planner-refactor/lidar-inference-equivalence.log`。

增加失败代价的动机是避免长期负代价使快速失败获得相对较好的累计回报；
降低首层初始化尺度的动机是让米级点云与状态融合在初始化时具有合适的相对幅度。
这些仍是待验证的训练假设；多项参数共同变化，不能把效果归因于单项修改。
各种算法的完整损失配方有区别，本批次不声明仅替换优化算法的消融结果。

APG-Depth 的三种子复现：

```bash
pixi run drone-playground -m mode=train experiment=navigation_depth_scratch \
  learning=apg seed=0,1,2 'output=results/scratch_depth_apg_reproduce_s${seed}'
```

种子 2 已按首次连续通过 160/320/480 选择第 480 次模型，独立冻结为 198/200：
六个主场景全部 25/25，S06 为 25/25、D06 为 23/25。
该种子保存路径的成本为 480 次更新、1,966,080 次交互、4,715.27 s。
种子 1 已按 320/480/640 选择第 640 次模型，独立冻结为 198/200：
六个主场景全部 25/25，S06 为 25/25、D06 为 23/25；
保存路径成本为 640 次更新、2,621,440 次交互、5,510.78 s。
APG-Depth 三个种子独立冻结合计 596/600，六个主场景合计 450/450。
主机中断尝试未记录部分的成本边界见下方恢复说明。

## 旧仓库经验的当前核对

只读核查了旧仓库 `docs/notes/research/{pointcloud-pilot-diagnosis,pointcloud-next-candidate,learning-configs}.md`
和 `docs/notes/archive/training-pilot.md`，以及实际网络、损失和 PPO/SHAC 配方。
旧的 47/48 是开发集结果，不等同于本批次三种子 C1–C6 验收。

当前固定 S01 初态、真实 Mid-360 测量的零更新探针，见
[诊断数据](../results/scratch_diagnostics/perception-scales.json)：

| 条件 | 感知 RMS | 状态 RMS | GRU reset / update 预激活绝对值 >5 |
|---|---:|---:|---|
| 原随机初始化、米级点 | 10.879 | 0.475 | 52.1% / 55.2% |
| 首层小尺度初始化 | 1.088 | 0.475 | 0% / 0% |
| 原首层初始化、输入固定 ×0.1 | 1.088 | 0.475 | 0% / 0% |
| 新 APG-LiDAR 第 80 次 | 2.224 | 0.481 | 0% / 1.0% |
| 新 SHAC-LiDAR 第 80 次 | 2.342 | 0.483 | 0.5% / 3.1% |
| 成功 APG-Depth 第 640 次 | 0.126 | 0.470 | 0% / 0% |

小首层和固定输入缩放能在初始化时产生相同表征，但优化过程不同：
固定输入缩放也缩小对首层 kernel 参数的梯度，并使等效权重更新乘以 0.1；
首层小初始化不在后续更新中固定该尺度。不能因初始 RMS 相同就称二者训练等价。
Depth/LiDAR 第 640/80 次并非同训练阶段，以上单初态探针不证明算法或网络的质量高低。

旧 PPO/SHAC motion-aware 配方实际 `reward_scale=0.01`，故配置的 `failure_penalty=-1000`
实际进入 reward 为 −10；当前额外代价 160 加上基础 20，实际失败 reward 扣除 180。
两者还有目标速度、进度奖励和控制定义差异，不能直接搬移数值。
旧 recurrent LiDAR 采用 point-mass-lag、直接加速度；当前采用 Crazyflow 500 Hz、
`tanh(action)×6 m/s²` 和实际低层控制。`initial_log_std` 仅作用于 PPO；
APG/SHAC 是确定性 Actor，不把该字段解释为它们的探索强度。

新 safe PPO-Depth 的首次检查仍 0/25，S01 主要为约 14.35 s 的碰撞；
早期基础 PPO 在约 2.2 s 下越界。新 APG-LiDAR 首次 S01/S02 仍 0/25，
但轨迹已能到达 x≈62–99 m，主要是碰撞及末端高度/目标位置偏差；
SHAC-LiDAR 则仍有下越界。当前证据支持继续区分目标高度跟踪与边界提前惩罚，
尚未在没有后续实验结果时宣称已解决或把全部建议都实现。

后续固定初态探针见 [LiDAR 复查](../results/scratch_diagnostics/lidar-followup.json)：
APG 第 160 次感知/状态 RMS 为 1.835/0.502，SHAC 第 240 次为 1.999/0.522；
GRU update 门预激活绝对值超过 5 的比例分别为 0.5%/1.6%。
当前初态没有复现训练后特征失控；这不代表全部轨迹都没有饱和。
既有高度跟踪项在 z=0.6/5.9 m 的位置梯度约为 −0.480/+0.580，方向指向安全高度。
SHAC-LiDAR 第 320 次中断前主场景 S01/S02/S03/D01/D02/D03 为 25/25/20/25/8/25；
该轮尚未完成，不能计为 C5 通过。先继续已有配方，不增加边界损失。

[实际 PPO 对照](../results/scratch_diagnostics/ppo-failure-path.json)使用相同随机动作、
两个真实物理世界，其中一个从 z=0.6 m 以 −2 m/s 下落。
额外失败代价使失败 reward 和 GAE advantage 各下降约 160，
一次 actor 更新的最大参数差为 0.000600，证明该代价到达 actor 更新。
这项局部检查不证明任意状态下提前失败都比持续飞行差。
整数 `initial_log_std=-3` 的初始化类型问题已修正并通过实际 PPO 更新回归；
此前 safe YAML 使用 −3.0，不受该问题影响。

主机于 2026-10-09 07:06 UTC 重启，六个训练进程中断；
[保存状态与成本边界](../results/scratch_diagnostics/host-restart.json)记录检查点、
已知未保存更新和重启前的主存/交换空间压力。恢复不重新初始化参数，
中断的选模检查先补全，部分结果保留，种子编号不跳过。
保存状态之后未记录的交互和墙钟时间无法精确恢复；
本次中断尝试的总成本只能报告下界，不能把缺失部分填为零。
验证期间先并行三个 GPU 训练，后续并行度同时按主存与显存余量安排。

Depth-PPO 基础 safe 配方在 480/800 次的 S01 为 15/25、25/25，
但在 640/960 次再次回退；KL 最近多次为 0.035–0.060，裁剪比例约 25%–35%。
第 800 次实际 Critic 值为 −173.1 至 3.4、均值 −131.4，见
[值函数检查](../results/scratch_diagnostics/ppo-critic-scale.json)，
没有发现此前怀疑的明显数值尺度问题。训练后的 raw action 噪声标准差约 0.04，未持续扩大。

新增 `navigation_depth_scratch_ppo_stable`，只把 Actor `lr` 从 3e-4 降至 1e-4；
其余 safe 配置与实际物理/验收规则相同。选定 Depth-PPO 种子 0/1/2
改为 `results/scratch_stable_depth_ppo_s{seed}`，均重新随机初始化；
旧 safe 运行和恢复检查点保留，不能把旧优化器状态导入新学习率运行。
这一变化针对更新幅度，不声称已提高冻结质量；LiDAR-PPO 保留原学习率。

第 640/800/960 次的 SHAC-LiDAR 固定初态复查见
[后期诊断](../results/scratch_diagnostics/lidar-late-followup.json)：
感知 RMS 为 1.563/1.492/1.403，状态 RMS 为 0.565/0.577/0.586，
update 门大预激活比例为 6.3%/7.3%/7.3%，状态到动作 Jacobian 范数为 0.359/0.518/0.580。
该初态没有复现感知特征持续放大；保持现有 PointNet 输入单位和初始化。
第 640 次主场景全部 25/25，但 800 次 S03/D03 为 20/21，960 次 D03 为 20；
动态障碍表现仍有波动，尚未满足连续三次全主场景通过。

APG-LiDAR 种子 0 已按第 320/480/640 次连续通过选第 640 次模型，
独立冻结为 198/200：S01/S02/S03/D01/D03/S06/D06 各 25/25，D02 为 23/25。
保存路径成本为 640 次更新、1,310,720 次交互、7,746.68 s；
六个主场景的独立冻结总数为 148/150，不能据单种子宣布三种子单元完成。

LiDAR-PPO 基础 safe 配方在第 160/320/480/640/800 次八场景均为 0/25，
最近 KL 约 0.01–0.017，更新幅度已不同于早期 Depth-PPO。
按用户建议开始普通 PPO 的同观测独立 Critic 试验：只覆盖
`learning.options.critic_uses_sensor=true`，Actor、学习率、奖励、采样和 Task 保持该基线配置。
新选定路径为 `results/scratch_sensor_lidar_ppo_s{seed}`，三种子均随机初始化；
`results/scratch_safe_lidar_ppo_s0` 的失败结果和恢复检查点保留。
此试验不添加进度奖励、不使用仿真特权字段，也尚未证明质量提高。
该试验第 640 次主场景为 2/150，第 800 次上升至 63/150：
S01/S02/S03/D01/D02/D03 分别为 25/1/0/25/12/0；继续观察，尚未收敛。
2026-10-09 用户进一步授权测试 DiffAero AsymmetricPPO。
[训练专用特权 Critic](adr/0010-training-only-privileged-ppo-critic.md)已加入并通过五项定向回归，
完整 CPU 回归 239 项通过；Depth 种子 0 已启动，LiDAR 特权配方尚未启动，
不混同于本段的普通传感器 Critic 试验。

Depth-PPO 低学习率基线第 160/320/480/640/800/960/1120 次八场景均未成功，
近期 KL 约 0.012–0.022，降低更新幅度尚未带来冻结质量证据。
新独立试验沿用该配置，只覆盖 `learning.options.progress_reward_scale=0.5`，
每向目标接近 1 m 增加 0.5 reward；高度、净空、失败代价和 Critic 配置保留。
该阶段选择 `results/scratch_progress_depth_ppo_s{seed}` 三个随机初始化运行；
低学习率基线 `results/scratch_stable_depth_ppo_s0` 的检查点与失败分母保留。
与 LiDAR 的感知 Critic 试验分别记录，不把两项同时变化归为单项效果。

截至上述新一轮调参开始，18 次正式运行中已有 6 次通过：
Depth APG 三种子，以及 Depth SHAC / LiDAR APG / LiDAR SHAC 各种子 0。
Depth SHAC 种子 0 按 480/640/800 次连续通过选第 800 次，独立冻结 199/200，
唯一失败位于 D03（24/25）。LiDAR SHAC 种子 0 按 1600/1760/1920 次连续通过选第 1920 次，
独立冻结 200/200。两者的事件、完整评测和权重均保留在选定目录。

Depth 进度奖励对照在 160/320/480 次检查均为 0/200，主动停止并以真实最新分数 0 记录为 pruned。
选定 Depth-PPO 路径更新为 `results/scratch_asymmetric_depth_ppo_s{seed}`；
新候选相对低学习率基线只增加特权 Critic，重新随机初始化，尚未证明改善质量。
当前并行运行 LiDAR PPO 对照及 Depth SHAC / LiDAR APG / LiDAR SHAC 的种子 1；
特权 Depth PPO 候选优先于其余种子 2 排队。CPU 验证期间最多五路 GPU 训练，验证结束后有足够 RAM 和显存时增到六路。
GPU 整卡利用率已达 100%；并行量由实际资源决定，不能把进程数等同于吞吐量。
optim-agent 0.1.1 的 ask/tell 使用 `.optim-agent-runs/ppo-depth.db` 和 `ppo-lidar.db`，
候选的真实命令、逐次主场景分母和状态保存在同目录；独立工具环境位于 `tmp/optim-agent-venv/`。
[执行计划](research/scratch-convergence-plan.md)记录优化目标、停止条件和未完成事项。

## 选定清单

全矩阵报告保留此前已完成的 Tracking / Racing 六个单元；
其中 Racing 使用本项目的 APG 预训练。新增的十二个 Navigation 单元全部采用本批次从零训练。
清单包含尚未启动的种子，避免仅选择成功结果。

```json
{
  "root": ".",
  "seeds": [
    0,
    1,
    2
  ],
  "runs": [
    "results/acceptance_tracking_ppo_s0",
    "results/acceptance_tracking_ppo_s1",
    "results/acceptance_tracking_ppo_s2",
    "results/acceptance_tracking_apg_s0",
    "results/acceptance_tracking_apg_s1",
    "results/acceptance_tracking_apg_s2",
    "results/acceptance_tracking_shac_s0",
    "results/acceptance_tracking_shac_s1",
    "results/acceptance_tracking_shac_s2",
    "results/acceptance_racing_initialized_ppo_s0",
    "results/acceptance_racing_initialized_ppo_s1",
    "results/acceptance_racing_initialized_ppo_s2",
    "results/acceptance_racing_initialized_apg_s0",
    "results/acceptance_racing_initialized_apg_s1",
    "results/acceptance_racing_initialized_apg_s2",
    "results/acceptance_racing_initialized_shac_s0",
    "results/acceptance_racing_initialized_shac_s1",
    "results/acceptance_racing_initialized_shac_s2",
    "results/scratch_depth_apg_s0",
    "results/scratch_depth_apg_s1",
    "results/scratch_depth_apg_s2",
    "results/scratch_asymmetric_depth_ppo_s0",
    "results/scratch_asymmetric_depth_ppo_s1",
    "results/scratch_asymmetric_depth_ppo_s2",
    "results/scratch_safe_depth_shac_s0",
    "results/scratch_safe_depth_shac_s1",
    "results/scratch_safe_depth_shac_s2",
    "results/scratch_sensor_lidar_ppo_s0",
    "results/scratch_sensor_lidar_ppo_s1",
    "results/scratch_sensor_lidar_ppo_s2",
    "results/scratch_safe_lidar_apg_s0",
    "results/scratch_safe_lidar_apg_s1",
    "results/scratch_safe_lidar_apg_s2",
    "results/scratch_safe_lidar_shac_s0",
    "results/scratch_safe_lidar_shac_s1",
    "results/scratch_safe_lidar_shac_s2"
  ]
}
```
