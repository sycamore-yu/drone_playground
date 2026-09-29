# 点云可微飞行方法

来源为 Yi Liu 等的《Learning to Fly from Point Clouds via Differentiable Simulation》，IROS 2026，会议页码5831—5838。[论文](https://rasevents.org/uploads/documents/pdfviewer/a9/f4/233762-1123.pdf)在2026-09-28的本地复核摘要为 `e2f72a7be6653889dbd0cd88c78a8579660a95ff93d344fd00a1c83e67246dd4`。本地实现身份为 `paper-public-information-reconstruction-v1`，采用论文公开信息及明确记录的补充假设。

## 模块配方

论文式均匀 MID-360 点云测量、PointNet 类逐点编码、GRU 时序策略、三维加速度输出、一阶滞后质点动力学，以及带时间衰减的状态雅可比反传共同构成完整方法。LOTF 是另一个独立方法。

| 部分 | 论文公开内容 | 本地实现与声明 |
|---|---|---|
| 点云 | 水平360度、竖直约60度、2度间隔，共180×30条射线 | `UniformMid360Lidar` 保留最近有效交点和机体系坐标 |
| 编码 | 逐点64/128/1024、LeakyReLU、全局最大池化、192维投影 | `PointCloudPolicy.encode` |
| 时序融合 | 本体状态投影192维，与感知相加，GRU192，线性加速度头 | `PointCloudPolicy.act` |
| 动力学 | 可微质点和一阶执行响应 | `PointMassLag` |
| 更新 | AdamW、固定学习率0.0001、32环境、0.1秒、160步 | `pointcloud_bptt`，完整状态可保存与恢复 |
| 目标 | 速度Huber、接近速度加权净空、加速度与加加速度正则 | `PaperObjective` 按论文显示公式计算 |
| 时间导数 | 动力学雅可比乘指数衰减 | `(p,v,a)` 状态导数采用 `exp(-alpha*dt)` |

## 补充假设

一阶时间常数为1/12秒，重力为9.80665；平移积分使用项目固定的离散方程，衰减系数为 `-ln(0.4)`。参考机制来自 [DiffPhysDrone](https://github.com/HenryHuYu/DiffPhysDrone/tree/271936190b5c2a5e760230e718fa6b167718a5bf) 的固定提交。姿态由推力方向和速度偏航构造，姿态映射状态导数分离。

点云对位姿的状态导数分离，网络参数及平移控制链保留训练梯度。论文没有明确给出射线求交的反向实现，该边界作为重建假设记录。状态输入为当前速度3维、目标速度3维、姿态3维及半径1维；动作在机体系输出后变换到世界系。

扫描仰角从−7.2度开始，量程0.1—100米；未命中点由有效掩码排除池化。LeakyReLU 斜率0.01，点云与状态投影使用线性层，GRU采用Flax实现。AdamW 的参数为 `(0.9,0.999)`、`eps=1e-8` 和 `weight_decay=0.01`。

原重建的损失权重为速度1、碰撞1.5、加速度0.01、加加速度0.001；碰撞平滑项采用 `beta1=4/3`、`beta2=32`，加加速度均值与方差权重1和0.1。Huber 阈值1，速度采用30步因果平均，接近速度权重的梯度分离。论文式（5）的显示公式与配套文字存在表达差异，本地计算采用显示公式。

训练每次重新生成静态图元：球、直立圆柱和长方体各30个，空间48×18×6米，速度范围2—6米/秒。图元数量、生成范围、上述未披露参数和具体输入排列均为本地重建选择。50000次目标更新依据论文图5的横轴制定，对应2.56亿次环境转移。

## 三类独立实验

原始重建通过 `method=paper/pointcloud_flight env=paper/pointcloud_flight` 训练。开发集由独立静态图元构成，用于选择参数。历史第一版迁移评测是八张场景×4/6/8米每秒、40秒时限，共24回合；原冻结工作树仍按其源配置完成50000次更新和随后评测。

正式矩阵的点云导航使用第30000次更新冻结参数，通过 `env=paper/pointcloud_navigation_v2` 在4/6/8/20米每秒、300秒时限下评测，共32回合。静态和动态各16回合，完整失败分母见[正式结果](../verification/final-acceptance/README.md)。这一历史参数选择与原50000次目标运行分开报告。

正式矩阵的悬停、跟踪、竞速使用 `paper/control/hovering`、`paper/control/tracking` 和 `paper/control/racing`。它们保留完整点云编码与循环结构，使用 `paper_pointnet_gru_conditioned` 的0.02点坐标尺度，并执行各自1024次控制迁移更新。三项控制结果分别属于具名任务适配。

后续 `learning/pointcloud_navigation` 为显式导航域适配，采用导航场景训练、独立损失、500Hz积分和25—50毫秒随机传输延迟。其结果移入表外研究归档，正式历史矩阵保持原选择。可选 `training.point_noise_std_m` 控制训练点坐标扰动，默认0；扰动代码具备回归验证，扰动训练的效果仍待独立实验。

## 配置与验证

```bash
pixi run train method=paper/pointcloud_flight env=paper/pointcloud_flight --cfg job
pixi run train method=paper/pointcloud_flight env=paper/pointcloud_flight \
  runtime.device=gpu run_id=pointcloud-paper-new
```

来源清单固定代码和假设文件，参数元数据保存真实模块配置。单步与多步前向、导数边界、有效点掩码、点排列不变性、循环状态、完整恢复和原始事件分别由测试覆盖。新训练使用独立运行标识；原冻结运行按[分支生命周期](branch-lifecycle.md)保持其源码与环境。
