# 实验配方与调整记录

所有配方使用同一 Crazyflow first-principles 前向、500 Hz 物理、具名 Actor、Task 判定。
`checkpoint_eval` 用于诊断和选模；最终 `benchmark` 使用独立种子分区。完成状态以结果文件为准。

## 当前选定配方

全部使用种子 0/1/2。精确运行目录由 [验收清单](acceptance.md) 明确列出，
完成状态见 [逐种子报告](../results/acceptance/current.md)。下表的初始化路径均相对仓库根目录。

| 条件 | 算法 | Experiment | 初始化归档 |
|---|---|---|---|
| Tracking | PPO/APG/SHAC | `tracking_gpu` | 随机初始化 |
| Racing | PPO/APG/SHAC | `racing_initialized` | `results/apg_racing_diagnostic_s0/checkpoints/update-00000400.policy.zip` |
| Depth Navigation | PPO/APG/SHAC | `navigation_depth_initialized` | `results/apg_depth_mixed_diagnostic_s0/checkpoints/update-00000400.policy.zip` |
| LiDAR Navigation | APG | `navigation_lidar_initialized` | `results/apg_lidar_clearance_diagnostic_s0/checkpoints/update-00000160.policy.zip` |
| LiDAR Navigation | SHAC | `navigation_lidar_finetune` | 同上 |
| LiDAR Navigation | PPO | `navigation_lidar_interleaved` | 同上 |

在同一任务与感知条件下，Actor、物理、初始化方案和冻结评价规则一致；
LiDAR 的算法间 Actor 学习率及训练场景轮换频率不同，因此不是只替换优化算法的消融实验。
初始化及配方搜索成本在下文单列，不将微调成本解释为从零训练成本。

Tracking 与 Racing 的三种子复现命令：

```bash
pixi run drone-playground -m mode=train experiment=tracking_gpu \
  learning=ppo,apg,shac seed=0,1,2 'output=results/tracking_reproduce_${learning.algorithm}_s${seed}'
pixi run drone-playground -m mode=train experiment=racing_initialized \
  learning=ppo,apg,shac seed=0,1,2 \
  initial_checkpoint=results/apg_racing_diagnostic_s0/checkpoints/update-00000400.policy.zip \
  'output=results/racing_reproduce_${learning.algorithm}_s${seed}'
```

Navigation 的对应命令与每次配方调整的证据见下文。复现时使用新输出目录；
继续既有运行时增加 `resume=<该目录>/checkpoints/latest.training.zip` 并保持其保存的配置。

## 初始诊断

- APG Tracking seed 0：300 次更新后，连续三次 checkpoint evaluation 及独立 100 回合通过，
  冻结位置 RMSE 0.0225 m；见 `results/apg_tracking_diagnostic_s0`。
- PPO Tracking seed 0：400 次更新后通过，冻结 100/100，RMSE 0.2186 m。
- SHAC Tracking seed 0：400 次更新后通过，冻结 100/100，RMSE 0.1665 m。
- APG Racing seed 0：400 次更新后通过，冻结 100/100 合法完赛。
- PPO Racing 从零训练：多次完整 checkpoint evaluation 均未合法完赛；记录显示能近似跟踪
  路线，但没有穿过第一道门。SHAC 原始 horizon 32 配方出现早期碰撞及较大路径梯度。
  失败和中断状态保留在对应 `*_diagnostic_s0` 目录。
- `racing_long_horizon` 将 horizon 改为 64、lr 改为 1e-4、时间梯度衰减率改为 0.4。
  它是单独的 SHAC 诊断，不替换原配方或失败记录。

上述诊断使用 Python 3.13/JAX 0.9.2 的只读依赖环境；导入的项目实现来自本仓库，
Crazyflow 来自固定官方源码。正式复验使用本项目 Pixi 的 Python 3.12/JAX 0.11.2。

## 锁定环境验收配方

`tracking_gpu` 保留初始网络与目标，batch=1024；每 100 次更新进行完整选模评测。
三算法、三个种子各自训练，连续三次通过后冻结；不设更新数或时长上限。

`racing_initialized` 使用同一份已保存的 APG Racing 参数作为三个算法的共同初始化，
由 `initial_checkpoint` 显式指定。它保留 MLP [256,128]，batch=1024、horizon=64、
lr=1e-4、时间梯度衰减率=0.4、PPO 初始 log_std=-3。每个种子重新初始化优化器、
Critic、随机数和环境，执行实际算法更新，重新满足 C5，最后独立冻结评价。
初始化模型的 hash、来源和预训练成本必须与后续训练结果一起报告；不能称为从零训练，
也不能只以微调时间代表端到端学习成本。PPO/SHAC 的 Critic 和目标仍各自独立。

已完成的锁定环境 Tracking 三种子均为 100/100：PPO 的成功回合 RMSE 为
0.1594/0.1767/0.1734 m，APG 为 0.0224/0.0226/0.0235 m，
SHAC 为 0.1377/0.2246/0.2106 m。Racing 九次初始化训练的冻结结果均为
100/100 合法完赛，门序均为 `1→2→3→4→2`。逐种子的原始指标与成本见
[验收报告](../results/acceptance/current.json)。

## Navigation 诊断与采样调整

初始 APG Depth 配方只在 S01 名义起点附近采样，首次完整选模评价八个场景均未成功。
导入参考归档的 CNN/GRU 参数后，直接执行仍会越过高度边界；缩放动作头也没有解决。
这些导入仅用于初始化诊断，不构成参考模型在新物理中的复现成功。

`apg_depth_altitude_diagnostic_s0` 增加显式高度误差代价
`altitude_weight * dt * (z-goal_z)^2`，权重 1.0，保留净加速度控制、原传感器和 Actor。
100 次真实更新后的 S01/S02 失败轨迹已能保持约 3 m 高度并飞行到约 x=46/41 m，
但全部因碰撞失败；S03 仍在早期碰撞，八个场景均未达到 C4。
独立控制诊断确认 10/50 Hz 下的零加速度和固定水平加速度输入没有系统性高度坠落。

Depth 高度适配所用初始化可从已声明的研究权重重新生成：

```bash
JAX_PLATFORMS=cpu pixi run python tools/convert_depth_reference.py \
  --action-gain 0.16666666666666666 --output results/reference_initialization/depth_reproduced.policy.zip
```

此命令生成的全部 Actor 张量已逐项与本次训练实际使用的
`gain_0.167.policy.zip` 精确比对一致；它只重现权重转换，不重现随后训练结果。

后续 `navigation_mixed_depth` / `navigation_mixed_lidar` 使用八场景轮换、20 次更新一组，
仅训练时在合法空间随机起点；冻结评测仍使用规格的名义起点及扰动。
Depth batch=128，LiDAR batch=32，horizon=32，lr=3e-4，时间梯度衰减率
0.916290731874155。独立诊断命令增加高度权重 1.0、净空 margin=1.5 m；
Depth 的 clearance_weight=8.0，LiDAR 的 collision_weight=8.0。
Depth 从上述第 100 次更新权重初始化，LiDAR 从随机参数初始化。
每 400 次更新做一次完整选模评价、每 100 次保存恢复状态；不设训练预算硬上限。
记录分别位于 `results/apg_depth_mixed_diagnostic_s0` 和
`results/apg_lidar_mixed_diagnostic_s0`。这两次仍是配方诊断，尚不能声明 C1—C5 通过。

Depth 混合训练第 100 次更新的小批独立诊断在 S01、S03 分别达到 4/4；
记录为 `results/depth_mixed_update100_diagnostic`。四回合不足 C4，报告仍为未通过。

LiDAR 的补充初始化诊断使用具名参考归档 `e1750add…`，转换脚本为
`tools/convert_lidar_reference.py`。参考输入缩放 0.1 合并进第一层 kernel，
point projection 的 bias 合并进相加的 state projection bias，动作 kernel 除以 6。
当前动作头无 bias，故没有复制原偏置 `[-0.002639, 0.001029, 0.018862]`；
当前 tanh 动作和 Crazyflow 控制也不同于原质点前向。这是明确的近似初始化。
其未经新训练的 S01/S03 小批评价均为 0/4，没有零样本成功声明。
`apg_lidar_transfer_mixed_diagnostic_s0` 在同一八场景采样、完整 Mid-360 下重新训练，
每 200 次更新做完整选模评测；旧的随机初始化实验继续独立保留。

Depth 混合训练第 400 次更新的完整选模评价在八个场景均为 25/25。
随后保存恢复状态，结束本次配方诊断，作为三个算法共同初始化来源。记录训练成本为
1,638,400 次交互、400 次更新、2,223.43 s 活动墙钟时间；此前高度适配和历史初始化
仍需另计。随机初始化 LiDAR 的第 400 次更新评价八场景均为 0/25；结束该诊断时
为 421 次更新、431,104 次交互，失败与恢复状态完整保留，继续验证迁移初始化配方。

`navigation_depth_initialized` 以 Depth 第 400 次更新权重为共同初始化，
保留上述 Actor、传感器、混合场景与物理执行条件，lr 改为 1e-5、PPO 初始
log_std=-3，每 25 次更新做完整检查点评价。三个算法各自以种子 0/1/2
重新初始化优化器、Critic、环境与随机数，执行实际更新。连续三次主场景通过后，
自动按 C5 选模并做独立冻结评测。结果目录为
`results/acceptance_depth_initialized_{ppo,apg,shac}_s{0,1,2}`；
九次正式运行均在第 25/50/75 次更新连续通过，选择第 75 次更新的权重。
独立冻结评测的八个场景全部为 25/25；三种子与共同 Actor/物理配置经验收收集器核对，
Depth 静态和动态共六个单元通过。各种子均执行 307,200 次新交互，
活动墙钟时间为 679.41–1,353.87 s，受同时运行作业及编译缓存影响，
不能把这些时间差直接归因于训练算法。共同初始化及此前诊断成本另外追溯。

LiDAR 迁移训练第 200 次更新的完整检查点评价：S01/S02/S03 均为 25/25，
D01/D02/D03 分别为 17/25、14/25、14/25，S06/D06 为 23/25、8/25。
动态失败均为碰撞，主场景多发生在起飞后约 3.5–4.1 s，另有中途碰撞；
高度越界已不再是本轮主要失败。该结果未达到动态 C4，继续同配方训练。

该配方继续训练后的完整评价如下（每格分母均为 25）：

| 更新 | S01/S02/S03 | D01 | D02 | D03 | S06 | D06 |
|---|---|---:|---:|---:|---:|---:|
| 200 | 25/25/25 | 17 | 14 | 14 | 23 | 8 |
| 400 | 25/25/25 | 22 | 18 | 17 | 23 | 10 |
| 600 | 25/25/25 | 23 | 25 | 16 | 24 | 23 |

D03 未随其他场景改善，因而保存该配方完整结果与恢复状态，停止时为
631 次更新、646,144 次交互、2,899.70 s 活动墙钟时间。
`navigation_lidar_clearance` 从第 600 次更新的权重开始新的独立诊断：
batch 32→64、lr 3e-4→1e-4、净空 margin 1.5→2.0 m、
`minimum_approach_speed` 0→1 m/s，其余具名 Liu 损失和梯度规则保持原值。
最低接近速度只用于损失权重，使近障时的惩罚不会因估计距离暂时增大而归零；
它不修改观测、控制或实际障碍运动。碰撞球和成功门槛不变。
这是配方调整，多个参数共同变化，不能将效果归因于某一个参数。
每 160 次更新完成一轮八场景训练后进行完整选模评价。

```bash
pixi run drone-playground mode=train experiment=navigation_lidar_clearance learning=apg \
  initial_checkpoint=results/apg_lidar_transfer_mixed_diagnostic_s0/checkpoints/update-00000600.policy.zip \
  output=results/lidar_clearance_reproduce_s0
```

第 160 次更新的完整评价六个主场景均为 25/25，S06 为 25/25，D06 为 0/25。
D06 的 25 次失败均为碰撞，部分发生在 x≈54 m，其他发生在 x≈78 m；
该扩展场景的退化完整保留，不改变其几何或结果。按原规格，扩展场景逐项报告，
不参与主场景 C4/C5 达标判定。结束该诊断时为 211 次更新、432,128 次交互、
1,354.44 s 活动墙钟时间；初始化消费的是第 160 次更新权重，完整来源成本单独计入。

`navigation_lidar_initialized` 保留该配方的 batch=64、完整 Mid-360、
margin=2.0、最低接近速度权重=1.0，lr 改为 1e-5、PPO 初始 log_std=-3，
每 25 次更新完整检查。三个算法各自运行种子 0/1/2，重新创建 Critic、优化器和 RNG，
连续三次主场景通过后选择第三份权重，进行独立冻结评测，全部扩展场景同时报告。
实际结果目录为 `results/acceptance_lidar_initialized_{ppo,apg,shac}_s{0,1,2}`。

该配方的 APG 三种子全部完成 C5 与冻结评测：三个种子的 D03 分别为
24/25、24/25、23/25，种子 1 的 S03 为 24/25，其余主场景均为 25/25。
三个种子的 S06 均为 25/25，D06 分别为 2/25、4/25、11/25。

PPO、SHAC 在种子 0 分别于第 175、75 次更新通过，但种子 1 未能保持连续通过。
PPO 种子 1 的 D03 在第 25/50/75/100/125 次检查分别为 24/17/16/14/1（分母 25）；
SHAC 种子 1 在第 75 次的 D03 为 22/25，第 150 次的 D02 为 20/25，
第 175 次 D02/D03 为 18/25、22/25。训练轮换后的退化说明当前 Actor 更新幅度
对这两个算法不够稳定，因而保留原始成功、失败和恢复状态，另开独立配方。

`navigation_lidar_finetune` 继承上述配置，只将 Actor 学习率从 1e-5 降到 1e-6。
Critic 学习率保持 1e-3，Actor、初始化归档、传感器、物理、场景采样、损失和
所有验收条件均保持原值。PPO、SHAC 各重新执行三个种子，不将旧配方的成功种子
混入新配方。该阶段写入 `results/acceptance_lidar_finetune_{ppo,shac}_s{0,1,2}`。
SHAC 已完成该配方的三种子验收；PPO 的进一步调整见后文。APG 选择已完成的
`navigation_lidar_initialized` 三种子。

旧 PPO/SHAC 配方的实际额外试验成本如下，不属于新配方的初始化成本，也未删除：

| 算法/种子 | 状态 | 更新 | 交互 | 活动墙钟秒 |
|---|---|---:|---:|---:|
| PPO/0 | 已通过但未纳入新配方 | 175 | 358400 | 1836.91 |
| PPO/1 | 保存后中断 | 142 | 290816 | 1129.40 |
| SHAC/0 | 已通过但未纳入新配方 | 75 | 153600 | 1129.37 |
| SHAC/1 | 保存后中断 | 175 | 358400 | 1879.86 |
| SHAC/2 | Hydra 调度下一种子后中断 | 1 | 2048 | 19.93 |

每个中断目录的 `checkpoints/latest.training.zip` 是精确恢复入口。
活动墙钟包含各自编译和评测；这些并行作业的时间不能相加解释为单卡独占耗时。

```bash
pixi run drone-playground -m mode=train experiment=navigation_lidar_finetune \
  learning=shac seed=0,1,2 \
  initial_checkpoint=results/apg_lidar_clearance_diagnostic_s0/checkpoints/update-00000160.policy.zip \
  'output=results/lidar_finetune_reproduce_${learning.algorithm}_s${seed}'
```

最初以三个种子分别启动并行进程，计划各自依次执行 PPO、SHAC；
PPO 诊断结束后中断该调度，SHAC 改为独立进程启动。
上方命令复现选定的 SHAC 三种子，默认顺序运行，资源竞争条件不同。

新配方 PPO 种子 2 的首次 D03 检查为 19/25，六次失败均为约 x=51 m、
t=21 s 的碰撞。为隔离更新的影响，使用原始第 160 次初始化权重重新评测同一组
25 个初态：独立入口指定 `benchmark.scenes=[D03]`、`seed=2`、
`benchmark.seed_base=1000105`，对应原检查的实际种子 1020105。
结果为 24/25；两个 CSV 的种子/world index，以及轨迹的初始时间、位置、速度、
姿态逐项精确一致。该对照确认权重更新影响了这组飞行表现，不能解释为不同初态。
证据位于 `results/lidar_initialization_seed2_matched_diagnostic`。
另一次默认分区的初始化诊断为 25/25，位于
`results/lidar_initialization_seed2_diagnostic`，其初态不同，不作为配对比较。

仅降低学习率的 PPO 三种子在第 50 次检查的 D03 为 13/18/18，第 75 次为
17/3/4（分母均为 25），没有达到稳定收敛。分别在 81/75/78 次更新保存后中断，
活动墙钟为 974.14/973.54/956.81 s；完整失败历史和恢复归档仍保留。
为检验长时间连续训练单场景的影响，`navigation_lidar_interleaved` 只将
`scene_updates` 从 20 改为 1，保持低学习率配方的其余配置。
这会更频繁地截断并重置训练回合；每次更新仍有 32 步真实物理采样，
冻结评测的初态、场景时钟和回合长度完全保持原值。

该配方种子 2 的首次主场景检查通过：D02 为 24/25，其他五个主场景为 25/25，
S06/D06 为 25/25、0/25。随后第 50、75 次检查连续通过，选择第 75 次更新。
冻结六个主场景全部为 25/25，S06/D06 为 25/25、15/25。
该种子执行 153600 次新交互，活动墙钟 736.56 s。另两个种子也已完成独立验收：
种子 0 的 D03 为 24/25，其余主场景为 25/25；种子 1 的六个主场景均为 25/25。
三个种子的 S06 均为 25/25，D06 按种子 0/1/2 分别为 0/25、1/25、15/25。
PPO 的新清单明确选择该配方全部三个种子：
`results/acceptance_lidar_interleaved_ppo_s0`、
`results/acceptance_lidar_interleaved_ppo_s1`、`results/lidar_interleaved_ppo_s2`。
种子 2 保留其初始实验目录名；目录名不参与验收判断。

SHAC 低学习率配方的三个种子冻结六个主场景全部为 25/25，S06 均为 25/25，
D06 分别为 4/25、2/25、3/25。最终选定的九次 LiDAR 运行全部在第 25/50/75
次更新连续通过，选择第 75 次权重；每次执行 153600 次新交互。
活动墙钟为 499.31–1447.46 s，包含编译与评测，且并行竞争条件不同。
共同 Actor 与物理配置、三种子一致配方及所有冻结 CSV 经验收收集器核对，
LiDAR 静态和动态六个单元通过，至此完整矩阵为 18/18。

```bash
pixi run drone-playground -m mode=train experiment=navigation_lidar_interleaved \
  learning=ppo seed=0,1,2 \
  initial_checkpoint=results/apg_lidar_clearance_diagnostic_s0/checkpoints/update-00000160.policy.zip \
  'output=results/lidar_interleaved_reproduce_ppo_s${seed}'
```

```bash
pixi run drone-playground -m mode=train experiment=navigation_lidar_initialized \
  learning=apg seed=0,1,2 \
  initial_checkpoint=results/apg_lidar_clearance_diagnostic_s0/checkpoints/update-00000160.policy.zip \
  'output=results/lidar_reproduce_${learning.algorithm}_s${seed}'
```

```bash
pixi run drone-playground -m mode=train experiment=navigation_depth_initialized \
  learning=ppo,apg,shac seed=0,1,2 \
  initial_checkpoint=results/apg_depth_mixed_diagnostic_s0/checkpoints/update-00000400.policy.zip \
  'output=results/depth_reproduce_${learning.algorithm}_s${seed}'
```

Depth 与最初的 LiDAR 配方各以一个进程顺序运行某算法的三个种子，算法之间并发。
相应单条复现命令按 Hydra 默认顺序执行全部九次运行；资源竞争条件因此不同。

## 吞吐调整

Scene 射线批次由 128 增至 8192；保留每条射线的时间、几何、遮挡和返回值语义。
RTX 4090 上 32 世界、完整 Mid-360 20,000 点/帧测量由 86.3 ms 降到 5.91 ms；
38 项几何/传感器回归通过。原生执行合并控制/物理/采样 JIT，并仅在规划决策时序列化测量。
结果指标保存整卡显存与利用率；并行实验存在资源竞争，比较成本时需报告这个条件。
