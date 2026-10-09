# 无人机训练方法：论文与实现参考

本研究记录用于确定网络、学习目标、梯度规则和参考超参数。实际训练组合及完成条件以 [功能规格](../spec.md) 为准；实验权重和报告见 [参考模型索引](../../research/checkpoints/manifest.json)。

## 1. Zhang et al. 2025：Depth CNN/GRU

第一方来源：[Learning vision-based agile flight via differentiable physics](https://doi.org/10.1038/s42256-025-01048-0)、[DiffPhysDrone](https://github.com/HenryHuYu/DiffPhysDrone)。

| 内容 | 论文与具名方法 |
|---|---|
| 传感与状态 | Depth + 本体状态、目标速度；论文采用逆深度并池化到 16×12 的网络输入 |
| 网络 | 卷积 `32/64/128` → 192 维特征、状态融合、GRU192 |
| 输出 | 加速度控制与辅助速度估计 |
| 目标 | 速度匹配、接近速度加权避障、加速度及 jerk；速度估计辅助目标由具体配方声明 |
| 优化 | 可微物理反传；状态敏感度时间指数衰减，减轻长时梯度放大 |
| 本平台 | 以 Crazyflow 作为前向物理，网络、损失和梯度规则按具名配方接入；D435i Sensor 与 CNN 输入预处理分开配置 |

可用于训练的网络预处理参考：轴向深度 `64×48` → `3 / clip(depth, 0.3, 10) − 0.6` → `4×4 max-pool`，输出 `12×16`；这是一种**具名研究预处理**，其 `10 m` 截止和 `20°` 安装仰角应单独配置。状态特征可包含机体系速度、目标速度、竖直方向与机体半径，共 10 维。上述数值部分来自经过实验的实现配方，作者原始条件以论文为准。

验证接口：图像深度和无效值、状态排列与坐标、双头输出、GRU reset、损失分项、Crazyflow 状态导数映射与指数衰减。

## 2. Liu et al. 2026：LiDAR PointNet/GRU

第一方来源：[Learning to Fly from Point Clouds via Differentiable Simulation（IROS 2026）](https://rasevents.org/uploads/documents/pdfviewer/a9/f4/233762-1123.pdf)。

| 内容 | 论文与具名方法 |
|---|---|
| 传感 | 论文仿真以 `360°×约60°`、`2°` 网格生成 `180×30` 射线；Depth `48×64` 也可反投影成点云 |
| 网络 | PointNet 逐点 `64/128/1024`、LeakyReLU、global max-pool、192 维映射、GRU192 |
| 状态 | 速度、目标速度、姿态、机体半径等，192 维状态融合 |
| 输出 | 3D 加速度命令 |
| 目标 | 速度 Huber、接近速度与净空、加速度、jerk |
| 优化 | AdamW `lr=1e-4`、`batch=32`、`dt=0.1 s`、`rollout=160`；BPTT 与时间衰减 Jacobian |
| 本平台 | 标准 LiDAR 使用 Mid-360 的非重复扫描；`uniform_lidar_paper` 保留论文角度网格；使用 Crazyflow 前向物理 |

论文报告的梯度 checkpointing 用于降低点云长时 BPTT 的激活显存压力。点云编码对排列保持不变，未命中点不参与最大池化，网络输入保留实际时间和空间标定。

参考实现配方中还采用过：LeakyReLU 斜率 `0.01`、状态特征 10 维、GRU192、AdamW weight decay `0.01`，以及以 `exp(−αΔt)` 调整状态雅可比。梯度超参数和障碍成本权重属于具体实验配方；以实际声明的值构造和复核训练目标。

验证接口：射线距离、mask 和池化、点云排列、循环记忆、动作坐标、避障/速度成本分项、状态梯度处理及真实闭环。

### 数值复现实验参数

| 项目 | 参考值 | 来源属性 |
|---|---|---|
| PointNet 激活与记忆 | LeakyReLU negative slope 0.01，192维 GRU | 具名方法的实现配方 |
| PointNet 训练辅助参数 | AdamW betas=(0.9,0.999)、eps=1e−8、weight_decay=0.01 | 实验设置，可单独修改 |
| 点云避障损失 | 速度1、碰撞1.5、加速度0.01、jerk0.001；平滑项 β1=4/3、β2=32 | 已实验过的公式重建参数 |
| 时域损失 | 30步因果速度平滑、Huber δ=1、jerk均值/方差权重1/0.1 | 具名实现的实验配方 |
| 论文式训练几何 | 球/圆柱/长方体各30个，48×18×6 m，参考速度2–6 m/s | 具名重建的随机几何分布 |
| 旧配方的执行滞后 | τ=1/12s，指数衰减 α=−ln(0.4) | 在质点模型上实验过的参数；Crazyflow 中需重新定义导数映射 |

上表将论文报告的数值与实验重建补充的参数明确区分。对可微损失，公式及实际保留的状态敏感度是实验身份的一部分。

## 3. 本平台统一 Actor 网络基线

在同一 Task/Sensor 下，**PPO、APG/BPTT、SHAC 使用相同 Actor 主干**；优化器与 Critic 按具体训练算法构造。初始配置为：

| 任务／观测 | 初始 Actor 主干 | 主要出处 |
|---|---|---|
| Tracking、Racing / State | MLP **[256,128]**，LayerNorm + ELU | [DiffAero MLP 配置](https://github.com/flyingbitac/diffaero/blob/main/cfg/network/mlp.yaml)、[网络实现](https://github.com/flyingbitac/diffaero/blob/main/utils/nn.py) |
| Navigation / Depth | CNN **32/64/128** + 特征192 + **GRU192** | [Zhang 2025](https://doi.org/10.1038/s42256-025-01048-0) |
| Navigation / LiDAR | PointNet **64/128/1024** + 特征192 + **GRU192** | [Liu 2026](https://rasevents.org/uploads/documents/pdfviewer/a9/f4/233762-1123.pdf) |

这套统一网络由本平台制定：DiffAero 的 RCNN（残差 CNN 8/16/8/8 + GRU512）和 VisFly 的可配置提取器与 Zhang 的 CNN/GRU 并非同一结构；两者的 State MLP 宽度也不同。PPO／SHAC 复用 Zhang、Liu 的 Actor，并不是原论文已经报告的 PPO/SHAC 结果。

**收敛调优参考**：当初始网络及参数未能达到 Spec C1—C5 时，优先核查实际执行与梯度，再考虑参考 [DiffAero](https://github.com/flyingbitac/diffaero)、[VisFly](https://github.com/SJTU-ViSYS-team/VisFly)、[VisFly-Lab/APG](https://github.com/Fanxing-LI/APG)、Zhang、Liu 的网络宽度、GRU 结构、特征融合、优化器参数和训练损失。修改作为新的 Experiment 配方保存；用于 PPO/APG/SHAC 受控比较时需在相同新 Actor 与物理条件下重新训练和评测。始终保留没有收敛的运行记录。

## 3. 状态训练与感知 PPO 参考

| 项目 | 官方实现 | 对本平台的用途 |
|---|---|---|
| **DiffAero** | [flyingbitac/diffaero](https://github.com/flyingbitac/diffaero) | PPO、APG/BPTT、SHAC 更新；状态 MLP、短展开和 actor/critic |
| **VisFly** | [SJTU-ViSYS-team/VisFly](https://github.com/SJTU-ViSYS-team/VisFly) | 视觉无人机网络、感知 PPO 与训练工程 |
| **VisFly-Lab / APG** | [Fanxing-LI/APG](https://github.com/Fanxing-LI/APG) | 状态输入、解析梯度、时间展开与训练配方 |

已有训练配方还采用过 PPO MLP `[64,64]`、APG MLP `[32,32]`，可作为未收敛时的对照候选；**本平台初始 State Actor 使用上表的统一 `[256,128]`**。PPO 和 SHAC 配备各自的 Critic，感知 PPO/SHAC 与 APG 使用相同的 Zhang/Liu Actor 主干和动作合同。

SHAC 的 target critic 更新须按具体公式解释 `alpha`：例如 `target = alpha*old + (1-alpha)*new` 时，新权重系数为 `1-alpha`。实际 actor 末状态梯度、价值 bootstrap 和 done/truncation 分别测试。

### 训练实现的参考配置

| 来源 | 参考网络及参数 | 参考用途 |
|---|---|---|
| DiffAero SHAC | MLP [256,128]、LayerNorm/ELU；actor/critic lr约0.001/0.003，γ=0.99、λ=0.95 | 网络结构与短时域训练的候选设置 |
| VisFly 感知 PPO | Depth特征128，state/goal网络[128,64]，actor/value [64,64]；lr 5e−5，horizon256，clip0.2 | 感知 PPO 训练器与观测编码参考 |
| VisFly-Lab APG | 状态特征192，actor/critic [192,96]，lr0.001，horizon96，100 agents | 解析策略梯度和时间展开参考 |

这些是上游或具名适配的**参考超参数**。最终训练以解析的 Experiment 配置及独立 benchmark 为准，采用其他数值时通过实验记录说明。

## 4. 实验身份与数据来源

具名方法配方记录网络、传感采样、目标与梯度规则；物理动力学独立记录为 Crazyflow 的具体型号。原论文采用简化点质量与一阶执行滞后，本平台的 Crazyflow 适配在论文条件之外改变了前向动力学，因此应标注**method adaptation**。

已有参数和冻结评测保存在 [参考模型](../../research/checkpoints/README.md)。相同数值可作为新的训练初始化候选；跨网络/物理条件加载时须显式检查观测与动作合同。最终训练结果按 [Learning 验收](../spec.md) 测量。
