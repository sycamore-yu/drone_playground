# 当前进度

更新时间：2026-09-28。现役重组工作树：`simulation_dev/.worktrees/drone_playground/architecture-v3`，分支 `refactor/composable-architecture-v3`。配置版本为 3。

## 本次架构重组

用户授权的结构重组已实现：method/env 装配、统一 train/eval/play、共享执行转移与 JAX/外部方法闭环、网络独立、环境子域、dynamics/预测/导数角色分离，以及 navigation 资产和协议。LOTF 特殊子模块由固定提交、打包补丁、许可证和项目自管缓存取代。

PPO 的悬停、跟踪、竞速、静态导航和动态导航均完成实际小预算训练、参数更新、检查点保存及独立冻结评测。两种 MPC 在恢复原始 `first_principles + cf21B_500` 条件后各完成一圈。SUPER/EGO 分别完成静态/动态三难度闭环，失败及超时保留。点云方法完成逐元素数值对照、旧完整状态迁入续训、5400点输入短训练及24格冻结测试。

最终完整回归274项测试、5项子测试通过，退出码0，耗时1287.93秒。测试、运行选择、质量与诊断运行分别记录在 [本次验证](verification/architecture-v3/README.md)，机器可读依据为 [evidence.json](verification/architecture-v3/evidence.json)。原始失败日志和各阶段修复证据一并保留。

完整实际目录见 [project-tree.md](project-tree.md)，入口见 [runbook.md](runbook.md)，职责见 [architecture.md](architecture.md)。实施账本位于 `.scratch/architecture-v3/`。基础重组提交为48cb031，最终收尾包括冻结权重局部覆盖、上游补丁路径、完整文档与验收归档；最终提交身份以本分支Git历史为准。

## 任务质量与后续范围

本次 PPO 仅每任务256交互，点云仅2次更新，均用于迁移工程验收。策略质量尚需其完整训练和独立评测。点云本轮24格为24次碰撞。原生导航的2m/s名义速度与96m位移/40秒时限存在预算冲突；新配方应明确校准这组条件，并保留原始失败结果。

LOONG、AERO-MPPI 与 AC-MPC 的身份已经确认，来源表标记为后续 JAX 实现。当前 SamplingMPC 保留其原有精英均值采样算法身份。

## 原运行与历史事项

原主工作树保持 `refactor/native-planner-runtime@1b93534` 及其已有未提交 `docs/status.md`。重组未覆盖原运行文件。

原点云工作树保持 `research/pointcloud-paper-navigation8@dbb660c`。原训练 PID3956802、协调器PID3980090 沿既有路径继续执行50000更新目标。最新现场核对写入本次 evidence.json；该原运行的后续结束状态按其自身 state.json 判断。

P2质量问题、P5历史归档补采及点云完整训练仍在 [backlog.md](backlog.md) 单独维护。旧状态全文保存在 [历史状态快照](research/archive/status-before-architecture-v3.md)。
