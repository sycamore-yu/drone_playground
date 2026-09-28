# P5 正式矩阵（v2）

状态：尚未完整。8 个训练单元已完成 8 个；独立开发 1152/1152 回合，留出 4608/4608 回合。

以下数值来自原始逐回合记录。缺失格保留“未完成”，不填零、不剔除失败。质量门槛尚未冻结，quality_passed=null。
单训练种子为 0；置信区间仅描述场景回合抽样，不估计跨训练种子稳定性。
共同协议：动作 50 Hz、物理 500 Hz、时限 40 秒、到达半径 0.5 米、机体球半径 0.07 米；同一步碰撞优先于到达。

## 训练预算

| Task | Sensor | Method | 实际交互 | 最佳开发步 | 状态 |
|---|---|---|---:|---:|---|
| static | depth | ppo | 8388608 | 2097152 | completed |
| static | depth | dva | 8388608 | 1048576 | completed |
| static | lidar | ppo | 8388608 | 1048576 | completed |
| static | lidar | dva | 8388608 | 1048576 | completed |
| dynamic | depth | ppo | 8388608 | 1048576 | completed |
| dynamic | depth | dva | 8388608 | 1048576 | completed |
| dynamic | lidar | ppo | 8388608 | 3145728 | completed |
| dynamic | lidar | dva | 8388608 | 1048576 | completed |

## 留出结果总览

每行包含三个难度各 128 回合。成功条件时间仅统计到达回合，完整分母仍为 384。

| Task | Sensor | Method | 到达/N | 碰撞 | 越界 | 数值失败 | 超时 |
|---|---|---|---:|---:|---:|---:|---:|
| static | depth | ppo | 0/384 | 7 | 377 | 0 | 0 |
| static | depth | dva | 0/384 | 0 | 384 | 0 | 0 |
| static | lidar | ppo | 0/384 | 25 | 359 | 0 | 0 |
| static | lidar | dva | 0/384 | 0 | 384 | 0 | 0 |
| static | depth | ego | 324/384 | 57 | 0 | 0 | 3 |
| static | lidar | super | 377/384 | 5 | 2 | 0 | 0 |
| dynamic | depth | ppo | 0/384 | 74 | 310 | 0 | 0 |
| dynamic | depth | dva | 0/384 | 0 | 384 | 0 | 0 |
| dynamic | lidar | ppo | 22/384 | 320 | 42 | 0 | 0 |
| dynamic | lidar | dva | 0/384 | 0 | 384 | 0 | 0 |
| dynamic | depth | ego | 199/384 | 185 | 0 | 0 | 0 |
| dynamic | lidar | super | 379/384 | 4 | 0 | 0 | 1 |

## 最终独立开发 36 格

| Task | Sensor | Method | Difficulty | N | 成功率 | 碰撞率 | 受限时间/s |
|---|---|---|---|---:|---:|---:|---:|
| static | depth | ppo | easy | 32 | 0.000 | 0.031 | 40.00 |
| static | depth | ppo | medium | 32 | 0.000 | 0.062 | 40.00 |
| static | depth | ppo | hard | 32 | 0.000 | 0.000 | 40.00 |
| static | depth | dva | easy | 32 | 0.000 | 0.000 | 40.00 |
| static | depth | dva | medium | 32 | 0.000 | 0.000 | 40.00 |
| static | depth | dva | hard | 32 | 0.000 | 0.000 | 40.00 |
| static | lidar | ppo | easy | 32 | 0.000 | 0.125 | 40.00 |
| static | lidar | ppo | medium | 32 | 0.000 | 0.031 | 40.00 |
| static | lidar | ppo | hard | 32 | 0.000 | 0.125 | 40.00 |
| static | lidar | dva | easy | 32 | 0.000 | 0.000 | 40.00 |
| static | lidar | dva | medium | 32 | 0.000 | 0.000 | 40.00 |
| static | lidar | dva | hard | 32 | 0.000 | 0.000 | 40.00 |
| static | depth | ego | easy | 32 | 1.000 | 0.000 | 11.19 |
| static | depth | ego | medium | 32 | 0.844 | 0.156 | 16.02 |
| static | depth | ego | hard | 32 | 0.781 | 0.219 | 17.79 |
| static | lidar | super | easy | 32 | 1.000 | 0.000 | 9.28 |
| static | lidar | super | medium | 32 | 1.000 | 0.000 | 9.39 |
| static | lidar | super | hard | 32 | 0.969 | 0.031 | 10.37 |
| dynamic | depth | ppo | easy | 32 | 0.000 | 0.062 | 40.00 |
| dynamic | depth | ppo | medium | 32 | 0.000 | 0.219 | 40.00 |
| dynamic | depth | ppo | hard | 32 | 0.000 | 0.438 | 40.00 |
| dynamic | depth | dva | easy | 32 | 0.000 | 0.000 | 40.00 |
| dynamic | depth | dva | medium | 32 | 0.000 | 0.000 | 40.00 |
| dynamic | depth | dva | hard | 32 | 0.000 | 0.000 | 40.00 |
| dynamic | lidar | ppo | easy | 32 | 0.125 | 0.781 | 35.54 |
| dynamic | lidar | ppo | medium | 32 | 0.000 | 0.969 | 40.00 |
| dynamic | lidar | ppo | hard | 32 | 0.000 | 0.938 | 40.00 |
| dynamic | lidar | dva | easy | 32 | 0.000 | 0.000 | 40.00 |
| dynamic | lidar | dva | medium | 32 | 0.000 | 0.000 | 40.00 |
| dynamic | lidar | dva | hard | 32 | 0.000 | 0.000 | 40.00 |
| dynamic | depth | ego | easy | 32 | 0.500 | 0.500 | 25.84 |
| dynamic | depth | ego | medium | 32 | 0.469 | 0.531 | 26.99 |
| dynamic | depth | ego | hard | 32 | 0.562 | 0.438 | 24.29 |
| dynamic | lidar | super | easy | 32 | 1.000 | 0.000 | 9.28 |
| dynamic | lidar | super | medium | 32 | 0.969 | 0.031 | 10.67 |
| dynamic | lidar | super | hard | 32 | 0.969 | 0.000 | 11.48 |

## 正式留出 36 格

| Task | Sensor | Method | Difficulty | N | 成功率 | 碰撞率 | 受限时间/s |
|---|---|---|---|---:|---:|---:|---:|
| static | depth | ppo | easy | 128 | 0.000 | 0.047 | 40.00 |
| static | depth | ppo | medium | 128 | 0.000 | 0.008 | 40.00 |
| static | depth | ppo | hard | 128 | 0.000 | 0.000 | 40.00 |
| static | depth | dva | easy | 128 | 0.000 | 0.000 | 40.00 |
| static | depth | dva | medium | 128 | 0.000 | 0.000 | 40.00 |
| static | depth | dva | hard | 128 | 0.000 | 0.000 | 40.00 |
| static | lidar | ppo | easy | 128 | 0.000 | 0.031 | 40.00 |
| static | lidar | ppo | medium | 128 | 0.000 | 0.055 | 40.00 |
| static | lidar | ppo | hard | 128 | 0.000 | 0.109 | 40.00 |
| static | lidar | dva | easy | 128 | 0.000 | 0.000 | 40.00 |
| static | lidar | dva | medium | 128 | 0.000 | 0.000 | 40.00 |
| static | lidar | dva | hard | 128 | 0.000 | 0.000 | 40.00 |
| static | depth | ego | easy | 128 | 0.898 | 0.102 | 14.35 |
| static | depth | ego | medium | 128 | 0.867 | 0.125 | 15.24 |
| static | depth | ego | hard | 128 | 0.766 | 0.219 | 18.34 |
| static | lidar | super | easy | 128 | 1.000 | 0.000 | 9.33 |
| static | lidar | super | medium | 128 | 0.992 | 0.000 | 9.60 |
| static | lidar | super | hard | 128 | 0.953 | 0.039 | 11.03 |
| dynamic | depth | ppo | easy | 128 | 0.000 | 0.109 | 40.00 |
| dynamic | depth | ppo | medium | 128 | 0.000 | 0.234 | 40.00 |
| dynamic | depth | ppo | hard | 128 | 0.000 | 0.234 | 40.00 |
| dynamic | depth | dva | easy | 128 | 0.000 | 0.000 | 40.00 |
| dynamic | depth | dva | medium | 128 | 0.000 | 0.000 | 40.00 |
| dynamic | depth | dva | hard | 128 | 0.000 | 0.000 | 40.00 |
| dynamic | lidar | ppo | easy | 128 | 0.148 | 0.688 | 34.71 |
| dynamic | lidar | ppo | medium | 128 | 0.023 | 0.852 | 39.16 |
| dynamic | lidar | ppo | hard | 128 | 0.000 | 0.961 | 40.00 |
| dynamic | lidar | dva | easy | 128 | 0.000 | 0.000 | 40.00 |
| dynamic | lidar | dva | medium | 128 | 0.000 | 0.000 | 40.00 |
| dynamic | lidar | dva | hard | 128 | 0.000 | 0.000 | 40.00 |
| dynamic | depth | ego | easy | 128 | 0.578 | 0.422 | 23.65 |
| dynamic | depth | ego | medium | 128 | 0.508 | 0.492 | 25.62 |
| dynamic | depth | ego | hard | 128 | 0.469 | 0.531 | 26.86 |
| dynamic | lidar | super | easy | 128 | 1.000 | 0.000 | 9.34 |
| dynamic | lidar | super | medium | 128 | 0.984 | 0.016 | 10.08 |
| dynamic | lidar | super | hard | 128 | 0.977 | 0.016 | 11.42 |

## 比较边界

- 学习组四帧压缩观测：D435 每帧 300 点；MID360 每帧 120 点。原生规划器读取完整 120×90 深度或每帧 24000 条 MID360 射线的有效点。
- 两模态策略观测均为 2420 维，但编码器、视场和采样率不同；四帧覆盖的历史时长也不同，不宣称网络或信息带宽相同。
- 同传感器 PPO/D.VA 共享输入及执行链；EGO/SUPER 是完整方法比较，不能将差值归因于单独的规划算法。
- 算法各自采用冻结的折扣和更新配方，实际值见 training.csv 和运行 manifest；这不是只替换损失函数的受控消融。
- 噪声头保留既有参数化：init_noise_std=0.367879 先取 log，再由 Brax softplus+0.001 转换，初始高斯尺度约 0.31426135（tanh 前），不是字面 0.367879；八单元一致，不改变本轮已冻结训练。
- 全部执行 Crazyflow first_principles/cf2x_L250；MuJoCo 用于几何核验与回放，ROS 仅存在于原生规划器外部工作进程。
- D435/MID360 为锁定标定的理想几何测量，未复刻真实硬件全部误差；各方法共同使用真值机体状态，不包含 VIO/FAST-LIO 状态估计。
- 碰撞按 500 Hz 物理子步离散判定，不声明任意速度下的连续碰撞检测保证。
- 未加动态预测器；动态任务评测原生重规划表现。质量合格与工程完成分列。
- v1 在地面碰撞遗漏被发现后中止并保留，未并入 v2。
- 67,108,864 仅为 v2 正式训练交互；工程运行与已中止 v1 另计。v1 PPO 最后记录至少 2,097,152 交互，不能把该额外开销抹去。
- 场景生成器、种子和参数预先固定；完整清单在运行初始化时落盘。scene-splits.json 按不含种子/编号的几何与运动参数指纹检查三个集合交集，并核对实际评测库摘要。训练清单为事后确定性重建，不宣称所有清单都在训练前物化保存。
- 早期 v2 评测仅导出代表轨迹；缺少全回合逐帧归档的单元按固定规则重评，使用后缀 archive-v1 的完整证据。选择规则只看归档缺失，不看得分；原结果及前后差异保留在 archive-repair.json，新增训练交互为零。

## 证据

training.csv：预算与选模；units.csv：各方法汇总（成功条件时间按全部成功回合加权）；cells.csv：逐格指标、原生 RPC 延迟和回放位置；episodes.csv：全分母逐回合；summary.json：报告 SHA256、代码身份、场景身份与完整性问题。
subtypes.csv 按难度内的场景子类型另作分层，保留各自分母和全部失败；这些行不增加训练单元或正式评测总回合数。

图表通过 scripts/plot_p5.py 重建；RScope 读取校验通过 scripts/verify_p5_replays.py 重建。
规划器 rpc_case_p95_max_s 是各回合 RPC 延迟第 95 百分位的最大值，包含通信和等待，不等于纯求解耗时。
规划器接收仿真时钟并异步计算；墙钟计时来自本次共享主机运行，不构成独占资源性能排名或硬实时保证。

## 尚缺证据

- p5-formal-static-depth-ppo-seed0-v2-dev/easy: all-case trajectory archive incomplete
- p5-formal-static-depth-ppo-seed0-v2-dev/medium: all-case trajectory archive incomplete
- p5-formal-static-depth-ppo-seed0-v2-dev/hard: all-case trajectory archive incomplete
- p5-formal-static-depth-ppo-seed0-v2-heldout/easy: all-case trajectory archive incomplete
- p5-formal-static-depth-ppo-seed0-v2-heldout/medium: all-case trajectory archive incomplete
- p5-formal-static-depth-ppo-seed0-v2-heldout/hard: all-case trajectory archive incomplete
- p5-formal-static-depth-dva-seed0-v2-dev/easy: all-case trajectory archive incomplete
- p5-formal-static-depth-dva-seed0-v2-dev/medium: all-case trajectory archive incomplete
- p5-formal-static-depth-dva-seed0-v2-dev/hard: all-case trajectory archive incomplete
- p5-formal-static-depth-dva-seed0-v2-heldout/easy: all-case trajectory archive incomplete
- p5-formal-static-depth-dva-seed0-v2-heldout/medium: all-case trajectory archive incomplete
- p5-formal-static-depth-dva-seed0-v2-heldout/hard: all-case trajectory archive incomplete
- p5-formal-static-lidar-ppo-seed0-v2-dev/easy: all-case trajectory archive incomplete
- p5-formal-static-lidar-ppo-seed0-v2-dev/medium: all-case trajectory archive incomplete
- p5-formal-static-lidar-ppo-seed0-v2-dev/hard: all-case trajectory archive incomplete
- p5-formal-static-lidar-ppo-seed0-v2-heldout/easy: all-case trajectory archive incomplete
- p5-formal-static-lidar-ppo-seed0-v2-heldout/medium: all-case trajectory archive incomplete
- p5-formal-static-lidar-ppo-seed0-v2-heldout/hard: all-case trajectory archive incomplete
