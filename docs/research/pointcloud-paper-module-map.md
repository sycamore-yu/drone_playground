# 点云可微飞行：论文复刻的模块配方

论文：Learning to Fly from Point Clouds via Differentiable Simulation，IROS 2026。
来源：[用户提供的论文](https://rasevents.org/uploads/documents/pdfviewer/a9/f4/233762-1123.pdf)。
实现位于分支 `research/pointcloud-paper-navigation8`；配置身份为
`paper-public-information-reconstruction-v1`。完整假设与训练/评测协议见
[规格](../../.scratch/pointcloud-paper/spec.md)，实际运行进展见
[执行账本](../../.scratch/pointcloud-paper/progress.md)。

## 一次选择完整方法

本配方为：论文式均匀 MID-360 点云测量＋PointNet 类逐点编码器＋GRU 循环策略＋
三维加速度命令＋一阶滞后质点动力学＋论文式状态雅可比指数衰减＋长时域时间反传。
学习编码器参数和控制/物理链具有训练导数；点云测量对位姿的导数采用分离观测规则。
论文对射线生成反向传播的描述不足，当前传感器导数边界作为显式重建假设保存。

| 现有槽位 | 新增配置选择 | 实际执行内容 | 来源范围 |
|---|---|---|---|
| 策略 `policy` | `paper_pointcloud` | 循环网络输出三维加速度，经机体到世界坐标变换形成命令 | 论文三维加速度；坐标实现详见规格 |
| 控制器 `controller` | `acceleration_passthrough` | 向质点模型传递重力补偿后的世界系加速度 | 质点配方的执行预设 |
| 动力学 `dynamics` | `paper_point_mass` | 位置、速度、加速度状态；一阶执行滞后 | 论文结构；时间常数及积分形式为声明的补充设置 |
| 反向子槽位 `dynamics.backward` | `exponential` | 对质点平移状态 `(p,v,a)` 输入导数乘 `exp(-decay_rate*dt)`，动作导数保留；姿态映射导数分离 | 按论文式（9）重建；与 DiffPhysDrone 的部分通路缩放分别登记 |
| 场景 `scene` | `paper_primitives` | 每次更新独立采样静态圆柱、长方体、球体与地面 | 论文图元类型；数量与分布为补充设置 |
| 观测 `observation` | `paper_pointcloud` | 机体系点集、有效性掩码及速度/目标速度/姿态/半径 | 论文输入类别；10维排列等细节为补充设置 |
| 测量子槽位 `observation.sensor` | `UniformMid360Lidar` | 水平180×竖直30、共5400条规则角度射线；10Hz同步测量 | 论文2度采样；仰角起点、距离限制与无回波处理为补充设置 |
| 网络 `network` | `paper_pointnet_gru` | 逐点64→128→1024、最大池化→192；状态192相加；GRU192→3维输出 | 论文第三节C；激活斜率等细节为补充设置 |
| 算法 `algorithm` | `pointcloud_bptt` | 160步时间展开与重计算节省显存；AdamW固定学习率0.0001 | 论文训练机制；优化器默认参数明确列在配置中 |
| 目标 `objective` | `paper_pointcloud` | 速度Huber、接近速度加权净空、加速度与加加速度目标 | 论文式（4）—（8）；未披露系数逐项暴露 |
| 任务 `task` | `paper_avoidance` | 10Hz目标速度驱动避障；训练每段16秒 | 论文0.1秒与160步；目标生成细节为补充设置 |
| 运行 `training` | `paper_pointcloud` | 32环境、种子0、目标50000更新、完整状态保存恢复 | 32环境来自正文；50000根据图5横轴和上游默认推定 |

## 训练与 Navigation8 的关系

`experiment=paper_pointcloud` 训练只使用独立静态图元。
`experiment=paper_pointcloud_navigation8` 加载冻结权重，切换场景槽位为 `paper_navigation8`，
保留策略、网络、传感器、控制器与动力学配置。后者加载当前权威 `configs/scene/navigation8.json`，
并核对用户验收记录中的目录摘要。

测试包含 S01/S02/S03/S06、D01/D02/D03/D06；每场景分别使用4、6、8米/秒命令速度，
共24个确定性名义试次。所有碰撞、越界、数值失败和超时计入分母。
到达半径0.5米、机体半径0.07米、40秒时限沿用 Navigation8；策略10Hz延续训练的循环记忆节拍。
碰撞检查以500Hz采样同一离散步的中间状态，完整步末状态与训练的0.1秒转移一致。
这项结果标记为“Navigation8 场景迁移测试”，实际动力学身份为 PointMassLag。

开发集来自另一组固定静态图元，用于选择检查点；Navigation8 测试结果只用于冻结后报告。
训练完成的预算与被选中检查点的训练年龄分别记录，允许完整训练后由开发集选中较早的参数。

## 运行入口

在本论文工作树内，使用已有项目环境并显式指向当前源码：

```bash
export PYTHONPATH="$PWD/src"
export SCIPY_ARRAY_API=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
PYTHON=/home/tong/tongworkspace/simulation_dev/mujoco/drone_playground/.pixi/envs/default/bin/python

# 查看完整解析配方
"$PYTHON" -m drone_playground.app --cfg job experiment=paper_pointcloud

# 新的完整训练使用唯一运行目录
"$PYTHON" -m drone_playground.app experiment=paper_pointcloud run_id=paper-pointcloud-new-run

# 已有本轮任务的阶段接续与冻结评测由单一编排进程负责
"$PYTHON" scripts/run_pointcloud_pipeline.py
```

执行编排入口前读取 `experiments/paper-pointcloud-seed0-pipeline-v1/state.json` 并核对活动进程。
已有运行使用其原会话或已保存训练状态接续；阶段首轮为1000次更新，完整目标为50000次。
权重、优化器、随机数和计数由同一训练状态文件恢复，阶段结果和完整预算分别显示。

## 复现程度

目前作者完整源码未取得。本文明确实现了论文公开结构，同时把传感器梯度、随机场景分布、
未披露损失权重、姿态表示与动力学参数作为公开补充假设；精确数值复现仍需这些作者细节。
规格、解析配置、代码身份、训练曲线和独立测试记录共同构成本轮可复查的方法重建证据。
