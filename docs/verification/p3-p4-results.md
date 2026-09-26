# P3/P4 实际结果

由 `scripts/summarize_p3_p4.py` 从逐回合记录和检查点校验值重建。

实验完成 29/29；达到当前任务门槛 25/29。

## P3：同动力学训练与评测

训练种子0；开发32回合选模；留出128回合独立重评。

| 任务 | 动力学 | 方法 | 训练交互 | 留出完成 | 位置均方根误差（米） | 备注 |
|---|---|---|---:|---:|---:|---|
| 八字 | `so_rpy` | PPO | 2,097,152 | 128/128 | 0.026879 | 达标 |
| 八字 | `so_rpy` | APG | 655,360 | 128/128 | 0.025742 | 达标 |
| 八字 | `so_rpy` | SHAC | 655,360 | 128/128 | 0.026354 | 达标 |
| 随机样条 | `so_rpy` | PPO | 4,194,304 | 128/128 | 0.009916 | 达标 |
| 随机样条 | `so_rpy` | APG | 655,360 | 128/128 | 0.007616 | 达标 |
| 随机样条 | `so_rpy` | SHAC | 655,360 | 0/128 | 0.513423 | 初始策略被开发集选中，质量未达标 |
| 八字 | `so_rpy_rotor` | PPO | 2,097,152 | 128/128 | 0.050759 | 达标 |
| 八字 | `so_rpy_rotor` | APG | 655,360 | 128/128 | 0.038828 | 达标 |
| 八字 | `so_rpy_rotor` | SHAC | 655,360 | 128/128 | 0.040800 | 达标 |
| 随机样条 | `so_rpy_rotor` | PPO | 4,194,304 | 128/128 | 0.010069 | 达标 |
| 随机样条 | `so_rpy_rotor` | APG | 655,360 | 128/128 | 0.007799 | 达标 |
| 随机样条 | `so_rpy_rotor` | SHAC | 655,360 | 128/128 | 0.023944 | 达标 |
| 八字 | `so_rpy_rotor_drag` | PPO | 2,097,152 | 128/128 | 0.065943 | 达标 |
| 八字 | `so_rpy_rotor_drag` | APG | 655,360 | 128/128 | 0.052028 | 达标 |
| 八字 | `so_rpy_rotor_drag` | SHAC | 655,360 | 128/128 | 0.058713 | 达标 |
| 随机样条 | `so_rpy_rotor_drag` | PPO | 4,194,304 | 128/128 | 0.013118 | 达标；数值失效边界修复v3，原学习参数；v1/v2失败保留 |
| 随机样条 | `so_rpy_rotor_drag` | APG | 655,360 | 128/128 | 0.011388 | 达标 |
| 随机样条 | `so_rpy_rotor_drag` | SHAC | 655,360 | 128/128 | 0.020356 | 达标 |
| 八字 | `first_principles` | PPO | 2,097,152 | 0/128 | 0.544722 | 有效低分 |
| 八字 | `first_principles` | APG | 655,360 | 128/128 | 0.199799 | 达标 |
| 八字 | `first_principles` | SHAC | 655,360 | 128/128 | 0.248511 | 达标 |
| 随机样条 | `first_principles` | PPO | 4,194,304 | 128/128 | 0.012657 | 达标 |
| 随机样条 | `first_principles` | APG | 655,360 | 128/128 | 0.010060 | 达标 |
| 随机样条 | `first_principles` | SHAC | 655,360 | 0/128 | 0.097696 | 初始策略被开发集选中，质量未达标 |

失败回合的误差仅覆盖存活步数，结合完成率解释。PPO/APG/SHAC的网络与训练预算各有冻结配方。

## P4：LSY Level0 竞速轨迹跟踪

固定作者门序 `[1,2,3,4,2]`、原生扰动和碰撞判定，实际穿越5次指定门。

| 方法 | 训练交互 | 留出完成 | 平均完成时间（秒） | 位置误差（米） | 状态 |
|---|---:|---:|---:|---:|---|
| PPO | 4,194,304 | 128/128 | 17.0986 | 0.014389 | 达标 |
| APG | 655,360 | 128/128 | 17.0567 | 0.004991 | 达标 |
| SHAC | 655,360 | 0/128 | — | 1.761449 | 初始策略被开发集选中，训练质量未达标 |
| attitude_mpc | 在线求解 | 117/128 | 17.1092 | 0.062754 | 达标 |
| sampling_mpc | 在线求解 | 128/128 | 17.0139 | 0.033627 | 达标 |

控制器耗时包含共享服务器竞争，真实耗时未注入仿真。完整每步求解状态和延迟保存在各分片记录。

## 检查点与回放

- `p2-figure8-ppo-seed0-v2`：`experiments/p2-figure8-ppo-seed0-v2/checkpoints/step-0001310720.pkl`；回放在同运行的 `independent-heldout/rollouts/`。
- `p2-figure8-apg-seed0-v1`：`experiments/p2-figure8-apg-seed0-v1/checkpoints/step-0000655360.pkl`；回放在同运行的 `independent-heldout/rollouts/`。
- `p3-figure8-shac-so_rpy-seed0-v1`：`experiments/p3-figure8-shac-so_rpy-seed0-v1/checkpoints/step-0000655360.pkl`；回放在同运行的 `independent-heldout/rollouts/`。
- `p2-random-ppo-seed0-v1`：`experiments/p2-random-ppo-seed0-v1/checkpoints/step-0003670016.pkl`；回放在同运行的 `independent-heldout/rollouts/`。
- `p2-random-apg-seed0-v1`：`experiments/p2-random-apg-seed0-v1/checkpoints/step-0000655360.pkl`；回放在同运行的 `independent-heldout/rollouts/`。
- `p3-random-shac-so_rpy-seed0-v1`：`experiments/p3-random-shac-so_rpy-seed0-v1/checkpoints/step-0000000000.pkl`；回放在同运行的 `independent-heldout/rollouts/`。
  训练结束策略另见同运行 `checkpoints/step-0000655360.pkl`，对应回放 `rollouts/step-0000655360/`；保留开发集原选模结果。
- `p3-figure8-ppo-so_rpy_rotor-seed0-v1`：`experiments/p3-figure8-ppo-so_rpy_rotor-seed0-v1/checkpoints/step-0001048576.pkl`；回放在同运行的 `independent-heldout/rollouts/`。
- `p3-figure8-apg-so_rpy_rotor-seed0-v1`：`experiments/p3-figure8-apg-so_rpy_rotor-seed0-v1/checkpoints/step-0000655360.pkl`；回放在同运行的 `independent-heldout/rollouts/`。
- `p3-figure8-shac-so_rpy_rotor-seed0-v1`：`experiments/p3-figure8-shac-so_rpy_rotor-seed0-v1/checkpoints/step-0000655360.pkl`；回放在同运行的 `independent-heldout/rollouts/`。
- `p3-random-ppo-so_rpy_rotor-seed0-v1`：`experiments/p3-random-ppo-so_rpy_rotor-seed0-v1/checkpoints/step-0003670016.pkl`；回放在同运行的 `independent-heldout/rollouts/`。
- `p3-random-apg-so_rpy_rotor-seed0-v1`：`experiments/p3-random-apg-so_rpy_rotor-seed0-v1/checkpoints/step-0000655360.pkl`；回放在同运行的 `independent-heldout/rollouts/`。
- `p3-random-shac-so_rpy_rotor-seed0-v1`：`experiments/p3-random-shac-so_rpy_rotor-seed0-v1/checkpoints/step-0000573440.pkl`；回放在同运行的 `independent-heldout/rollouts/`。
- `p3-figure8-ppo-so_rpy_rotor_drag-seed0-v1`：`experiments/p3-figure8-ppo-so_rpy_rotor_drag-seed0-v1/checkpoints/step-0001048576.pkl`；回放在同运行的 `independent-heldout/rollouts/`。
- `p3-figure8-apg-so_rpy_rotor_drag-seed0-v1`：`experiments/p3-figure8-apg-so_rpy_rotor_drag-seed0-v1/checkpoints/step-0000655360.pkl`；回放在同运行的 `independent-heldout/rollouts/`。
- `p3-figure8-shac-so_rpy_rotor_drag-seed0-v1`：`experiments/p3-figure8-shac-so_rpy_rotor_drag-seed0-v1/checkpoints/step-0000491520.pkl`；回放在同运行的 `independent-heldout/rollouts/`。
- `p3-random-ppo-so_rpy_rotor_drag-seed0-v3`：`experiments/p3-random-ppo-so_rpy_rotor_drag-seed0-v3/checkpoints/step-0004194304.pkl`；回放在同运行的 `independent-heldout/rollouts/`。
- `p3-random-apg-so_rpy_rotor_drag-seed0-v1`：`experiments/p3-random-apg-so_rpy_rotor_drag-seed0-v1/checkpoints/step-0000655360.pkl`；回放在同运行的 `independent-heldout/rollouts/`。
- `p3-random-shac-so_rpy_rotor_drag-seed0-v1`：`experiments/p3-random-shac-so_rpy_rotor_drag-seed0-v1/checkpoints/step-0000655360.pkl`；回放在同运行的 `independent-heldout/rollouts/`。
- `p3-figure8-ppo-first_principles-seed0-v1`：`experiments/p3-figure8-ppo-first_principles-seed0-v1/checkpoints/step-0002097152.pkl`；回放在同运行的 `independent-heldout/rollouts/`。
- `p3-figure8-apg-first_principles-seed0-v1`：`experiments/p3-figure8-apg-first_principles-seed0-v1/checkpoints/step-0000655360.pkl`；回放在同运行的 `independent-heldout/rollouts/`。
- `p3-figure8-shac-first_principles-seed0-v1`：`experiments/p3-figure8-shac-first_principles-seed0-v1/checkpoints/step-0000573440.pkl`；回放在同运行的 `independent-heldout/rollouts/`。
- `p3-random-ppo-first_principles-seed0-v1`：`experiments/p3-random-ppo-first_principles-seed0-v1/checkpoints/step-0003670016.pkl`；回放在同运行的 `independent-heldout/rollouts/`。
- `p3-random-apg-first_principles-seed0-v1`：`experiments/p3-random-apg-first_principles-seed0-v1/checkpoints/step-0000655360.pkl`；回放在同运行的 `independent-heldout/rollouts/`。
- `p3-random-shac-first_principles-seed0-v1`：`experiments/p3-random-shac-first_principles-seed0-v1/checkpoints/step-0000000000.pkl`；回放在同运行的 `independent-heldout/rollouts/`。
  训练结束策略另见同运行 `checkpoints/step-0000655360.pkl`，对应回放 `rollouts/step-0000655360/`；保留开发集原选模结果。
- `p4-racing-ppo-first_principles-seed0-v1`：`experiments/p4-racing-ppo-first_principles-seed0-v1/checkpoints/step-0004194304.pkl`；回放在同运行的 `independent-heldout/rollouts/`。
- `p4-racing-apg-first_principles-seed0-v1`：`experiments/p4-racing-apg-first_principles-seed0-v1/checkpoints/step-0000655360.pkl`；回放在同运行的 `independent-heldout/rollouts/`。
- `p4-racing-shac-first_principles-seed0-v1`：`experiments/p4-racing-shac-first_principles-seed0-v1/checkpoints/step-0000000000.pkl`；回放在同运行的 `independent-heldout/rollouts/`。
  训练结束策略另见同运行 `checkpoints/step-0000655360.pkl`，对应回放 `rollouts/step-0000655360/`；保留开发集原选模结果。
- `p4-racing-attitude-mpc-heldout-v2`：回放在 `experiments/p4-racing-attitude-mpc-heldout-v2/rollouts/shard-0/` 至 `shard-3/`。
- `p4-racing-sampling-mpc-heldout-v2`：回放在 `experiments/p4-racing-sampling-mpc-heldout-v2/rollouts/shard-0/` 至 `shard-3/`。

查看回放使用已交付的 RScope Viewer。报告包含所有32/128试次，回放文件保存固定前4及最差试次的子集。
