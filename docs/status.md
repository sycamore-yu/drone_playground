# 当前进度

更新时间：2026-09-25。项目阶段：P1/P2 实施中，功能分支 `implementation/p1-p2`。

## 当前结论

- 用户已接受的八项推荐与 Crazyflie/Brax/rscope 选择已写入当前规格。
- 独立 Pixi/Python 3.13.15 已安装并锁定，依赖检查通过。
- 已实现八字/样条纯函数任务、原生 Brax 训练入口、独立评测、检查点与运行记录。
- rscope 原生导出/读取与 TensorBoard 事件测试通过，正在执行整套测试及完整飞行。
- 已继承的 15 项 Brax 短程工程证据、rscope 往返和采样 MPC 探针见验证目录。
- 正式成功策略、LSY 竞速/MPC 完整运行、导航、Windows 远程查看均待阶段实施。

## 当前工作与下一步

当前工作：完成 P1 的整链验证，随后运行冻结预算的 PPO/APG 八字与随机样条训练。
当前训练进程：正式训练待整链测试完成；测试日志在 `tmp/p1p2/`。
执行会话：当前 Chat On Steroids 主会话；运行记录模块的同一辅助会话已恢复用于独立只读审查。
下一项：完整原生控制飞行、真实短程 PPO 更新、TensorBoard 服务及 rscope 窗口验证。
具体裁决和执行步骤见 `.scratch/drone-platform/p1-p2-execution.md`。

## 本轮验证

37 个 Markdown 文件的编码/空白检查、37 个本地文档链接、11 个任务依赖及 12 个模块/配置目录
检查通过；依赖图无环，任务状态均保持待开始。65 份历史文件逐一校验，15 个 Brax 短程结果和
2 项辅助验证保持原数据。此次只检查迁移后的证据完整性，原训练未重复执行。
原 Crazyflow 的 3 份项目设计文档已归档到这里；依赖变更补丁保存用于还原历史测试环境。
详细记录见 [项目建立检查](verification/project-setup-checks.json)。

## 入口

- [阶段表](../.scratch/drone-platform/map.md)
- [规格](../.scratch/drone-platform/spec.md)
- [任务 01](../.scratch/drone-platform/issues/01-visible-flight.md)
- [架构](architecture.md)
- [评测和策略质量](evaluation.md)
- [运行手册](runbook.md)
- [继承的接入报告](verification/2026-09-25-brax-integration.md)

任务状态以 `.scratch/drone-platform/issues/` 为准。每次实质进展、训练开始/结束或交接更新本页。
本页的文件更新时间代表记录更新时间；运行是否有进展还要核对实际日志、步数和检查点。
