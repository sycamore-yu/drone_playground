# 当前进度

更新时间：2026-09-25。项目阶段：P0，设计与目录交付已检查。

## 当前结论

- 用户已接受的八项推荐与 Crazyflie/Brax/rscope 选择已写入当前规格。
- 当前平台训练实现尚未启动；模块目录承载职责说明。
- 已继承的 15 项 Brax 短程工程证据、rscope 往返和采样 MPC 探针见验证目录。
- 正式成功策略、LSY 竞速/MPC 完整运行、导航、Windows 远程查看均待阶段实施。

## 当前工作与下一步

当前工作：文档、目录、任务依赖和历史证据迁移已检查，形成初始本地提交。
当前训练进程：本轮未启动。执行会话：本次 Chat On Steroids 调用会话。
下一项：任务 01，建立一条真实完整飞行→运行记录→远程 rscope→TensorBoard 的可见路径。

## 本轮验证

37 个 Markdown 文件的编码/空白检查、36 个本地文档链接、11 个任务依赖及 12 个模块/配置目录
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
