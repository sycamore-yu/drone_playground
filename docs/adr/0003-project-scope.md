# ADR-0003：首发研究范围与验收

**Status:** Accepted for scope; its former C/S/L numbering was superseded by [ADR-0015](0015-eight-acceptance-gates.md) on 2026-10-10.

## Context

平台首先服务于单机四旋翼的算法研究，需要可重复的感知导航、状态控制和原生规划器实验。

## Decision

- **Task**：Tracking、Racing、Navigation；固定 Navigation8 与 LSY Level0 场景，Navigation 同时包含静态和动态案例。
- **Learning**：PPO、APG/BPTT、SHAC，覆盖 Tracking、Racing 以及静态/动态 Depth 和 LiDAR Navigation，共 **18 个首发验收单元**。
- **Sensor**：标准虚拟 D435i 和 Mid-360 采用厂商公布的扫描与成像参数，Zhang/Liu 各自的网络预处理由具名方法配置。
- **Native Planner**：EGO-Planner＋Depth、SUPER＋LiDAR 接入静态与动态 Navigation。
- **Quality**：正式验收采用 S1–S4 / L1–L4；原 C1–C6 任务质量和 S6 原生规划器实测被完整保留在 L3 / S3，见 [ADR-0015](0015-eight-acceptance-gates.md) 和 [spec.md](../spec.md)。
- **Compute**：优先充分利用 GPU 训练，记录实际效率，不预置硬性训练资源上限。
- **Actor 网络**：三种算法按任务与传感器共享主干：Tracking/Racing 使用 DiffAero 风格 MLP `[256,128]`（LayerNorm + ELU）；Depth 使用 Zhang 2025 CNN/GRU192；LiDAR 使用 Liu 2026 PointNet/GRU192。PPO/SHAC 的 Critic 和训练目标按算法分别实现。
- **训练调整**：初始配置以统一网络为基线。未达到 L3 时，可依据 DiffAero、VisFly、VisFly-Lab 及 Zhang/Liu 的方法和参数调整网络及训练配置；保存独立实验版本，受控对照保持相同 Actor 网络与执行条件。

## Consequences

同一任务／感知条件下统一 Actor 主干与物理输出，以便对照 PPO、APG/BPTT 和 SHAC。达标判断按 Spec L3 的完整任务门槛，调参使用轻量筛查/完整 checkpoint validation，最终 benchmark 保持独立。场景来源与论文配方分别留存，不允许用于本轮的旧权重初始化。
