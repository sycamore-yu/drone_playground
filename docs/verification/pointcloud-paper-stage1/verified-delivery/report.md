# 点云论文重建与Navigation8交付核验

阶段训练与冻结测试已核验，完整预算继续进行。

实际训练：1000/50000 次更新，5120000/256000000 次交互。
开发集选择的权重来自第 1000 次更新；参数摘要 `2fafd2c04579afc5bc6cab0026820496b64708a39d38dabe3551127d3ff3ad7b`。
训练运行：`/home/tong/tongworkspace/simulation_dev/.worktrees/drone_playground/pointcloud-paper/experiments/paper-pointcloud-seed0-stage1-v1`。测试运行：`/home/tong/tongworkspace/simulation_dev/.worktrees/drone_playground/pointcloud-paper/experiments/paper-pointcloud-navigation8-stage1-v1`。

| 终止结果 | 数量 |
|---|---:|
| 到达 | 0 |
| 碰撞 | 0 |
| 越界 | 24 |
| 数值失败 | 0 |
| 超时 | 0 |

分母为8张固定场景×3个命令速度，共24个确定性试次。每个失败均计入统计。
已核对3份数值轨迹、24份回放及原评测进程生成的读回证明。

| 场景 | 4米/秒 | 6米/秒 | 8米/秒 |
|---|---|---|---|
| S01 | 越界 | 越界 | 越界 |
| S02 | 越界 | 越界 | 越界 |
| S03 | 越界 | 越界 | 越界 |
| S06 | 越界 | 越界 | 越界 |
| D01 | 越界 | 越界 | 越界 |
| D02 | 越界 | 越界 | 越界 |
| D03 | 越界 | 越界 | 越界 |
| D06 | 越界 | 越界 | 越界 |

对应组合：论文式5400点MID-360测量＋PointNet/GRU＋三维加速度＋一阶滞后质点＋指数衰减状态导数＋160步时间反传。

实际模块参数见 `module-slots.json`；公开参数与重建假设见 `reconstruction-assumptions.json`。
当前传感器采用分离位姿梯度的规则；已训练的是点云编码器及策略参数，控制和物理链提供训练导数。
本结果属于论文公开信息重建的Navigation8迁移评测；作者场景、实机、跨模态及薄障碍实验各自具有独立条件。
