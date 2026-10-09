# Depth / LiDAR 从随机初始化训练

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
  learning=ppo,shac seed=0,1,2 \
  'output=results/scratch_depth_reproduce_${learning.algorithm}_s${seed}'
pixi run drone-playground -m mode=train experiment=navigation_lidar_scratch_safe \
  learning=ppo,apg,shac seed=0,1,2 \
  'output=results/scratch_lidar_reproduce_${learning.algorithm}_s${seed}'
```

继续已有运行时保持其保存的配置，增加 `resume=<运行目录>/checkpoints/latest.training.zip`。
上方 Hydra grid 默认串行；当前执行按显存余量并行调度，竞争条件和墙钟时间须一起记录。

## 已记录的配方调整

APG-Depth 基础配方种子 0 在第 160 次更新时，S03 为 11/25、D03 为 24/25，
其余五个主场景为 25/25；第 320、480 次检查八场景均为 25/25。
第 640 次检查再次通过，选择该权重；独立冻结八场景全部 25/25。
该种子完成 2,621,440 次交互、640 次更新、2,988.22 s 活动墙钟时间。其余种子仍独立验收。

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
    "results/scratch_safe_depth_ppo_s0",
    "results/scratch_safe_depth_ppo_s1",
    "results/scratch_safe_depth_ppo_s2",
    "results/scratch_safe_depth_shac_s0",
    "results/scratch_safe_depth_shac_s1",
    "results/scratch_safe_depth_shac_s2",
    "results/scratch_safe_lidar_ppo_s0",
    "results/scratch_safe_lidar_ppo_s1",
    "results/scratch_safe_lidar_ppo_s2",
    "results/scratch_safe_lidar_apg_s0",
    "results/scratch_safe_lidar_apg_s1",
    "results/scratch_safe_lidar_apg_s2",
    "results/scratch_safe_lidar_shac_s0",
    "results/scratch_safe_lidar_shac_s1",
    "results/scratch_safe_lidar_shac_s2"
  ]
}
```
