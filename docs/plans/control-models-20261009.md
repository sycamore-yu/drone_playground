# 控制、反向模型与运行记录实施

**Status:** Completed in isolated worktree · 2026-10-09

批准范围：ADR-0001、0006—0009 及控制模型设计，增加两种旧 MPC 和 SUPER 理想跟踪。删除失效的 ADR-0005，Trajectory 不增加 JAX 要求，SE3 暂缓。

## 实施顺序

1. 控制接口与网络输出解耦，Controller 外部注入，共享动作解码与 Actor 记忆初始化。
2. 加入 SO3、LSY Attitude MPC、Sampling MPC、理想跟踪；构造阶段选择 JAX/宿主执行。
3. 加入 PointMassLag、LOTF 数学模型和训练导数连接，保留未建模输出的原生导数。
4. 环境实例保存命令、测量延迟状态；Crazyflow 原生 reset/step pipeline 随机化参数与载荷。
5. 配方独立选择观测、网络、损失、控制和验收协议；任务、赛道与记录使用同一事实来源。
6. 权重与选模成绩同目录，独立评测归入 eval，迁移工具保持权重字节和历史证据。
7. 运行针对性回归、短训练、控制闭环、Ruff 和安装包检查，记录实际结果。

## 执行边界

基线提交为 `a563386`。主目录有四个正在运行的训练，本轮在 `feat/control-models-20261009` 工作树中执行，不停止其它任务，不热改主目录源码，不迁移活动结果。主目录现有的 validation.md 改动保留。

历史权重不重解释动作语义；更改完整训练状态需要显式迁移或新运行。冻结策略的旧规格在读取边界识别，恢复的规格必须可核对。

## 验证

本次续接只执行针对性验证，没有重新运行全量回归、GPU 收敛矩阵或 ROS S6 飞行。

| 检查 | 结果 | 日志 |
|---|---|---|
| 模型、控制器、延迟、恢复和 CLI 选模 | 19 passed | `tmp/implementation/continue-targeted-green.log` |
| 迁移及外部证据保护 | 4 passed，含上述检查中的两项，共 21 个独立测试 | `tmp/implementation/continue-migration-green.log` |
| Ruff 与格式 | 全部通过，56 files already formatted | `tmp/implementation/continue-lint-final.log` |
| 实际 acados 短闭环 | 0.1 s 悬停回合 SUCCESS，进程退出 0 | `tmp/implementation/continue-attitude-mpc.log` |
| sdist 及从 sdist 构建 wheel | 成功 | `tmp/implementation/continue-build.log` |
| 安装后的模块与配置 | 9 个新增模块导入、66 种配置解析、2 份来源许可证检查通过 | `tmp/implementation/continue-package-smoke.log` |

acados 冒烟沿用 `learning` 协议，只执行了 1 回合，报告的 `passed: false` 表示未满足
100 回合的正式门槛；其 `outcomes` 为 `SUCCESS: 1`，不能把短闭环写成正式验收通过。
日志仍包含可选 Warp 未安装、CasADi 版本提示和 Optax 弃用提示，没有阻断上述执行。

续接修复了两处实际问题：归一化动作使用中心/尺度形式解码，避免对称加速度指令的
浮点抵消误差；宿主调度从整数物理步计算时间，避免 float32 时间造成时钟偏差。
没有放宽对应测试阈值。迁移工具另增加目录及报告符号链接检查，避免复制迁移写穿到
外部原始证据；两个新测试先复现失败，再验证拒绝且源文件不变。

早先完整测试的失败日志保留在 `tmp/implementation/full-tests-1.log`，不将其改写为
全量通过。三项历史失败均已在本次针对性测试中通过。基线环境/运行检查为 18 passed。

## 交付状态

使用说明见 [control-models.md](../control-models.md)。源码、配置、测试、结果读取/迁移
工具及对应 ADR 已同步。Trajectory 仍为宿主 NumPy 数据；SE3 未加入。

本次没有合并、提交或推送，也未移动主目录的历史结果。主目录有独立的未提交训练
改动；合并前需要保留并核对这些变化。运行中的作业继续使用原目录代码。
