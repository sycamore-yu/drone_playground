# 开发与维护

`main` 是现役入口。任务开始时读取 `docs/status.md` 和 `docs/backlog.md`，核对工作树、未提交改动及运行进程，然后直接读取相关模块。已有任务继续使用原会话和恢复状态。

## 验证

正式测试位于 `tests/`，临时探针和调试输出位于 `tmp/`。修改共享执行、配置、传感器、检查点或回放时，覆盖相关调用链并执行完整回归。算法变更先提供可失败的反例，再验证实际梯度、参数更新和保存／恢复一致性。

```bash
pixi run lint
JAX_PLATFORMS=cpu pixi run test
```

完整 MPC 测试依赖 acados；设置方式见[操作手册](runbook.md)。测试通过、预算完成和策略收敛分别给出证据。保存失败的测试名及原因，避免用短探针的成功覆盖任务质量问题。

直接调用`.pixi/envs/default/bin/python`而不经过`pixi run`时，须在启动解释器前设置`SCIPY_ARRAY_API=1`；Pixi已配置该激活变量。否则先导入SciPy的测试集合可能导致Crazyflow中的JAX姿态计算报Tracer转换错误。

## CPU 与 GPU

正式训练默认选择 GPU。单卡任务按显存和预算排队，重型训练与大批量评测分别调度。CPU 用于单元测试、短时诊断及原生规划器宿主执行。设备选择显式写入解析配置。

2026-09-29的8环境、64步 BPTT 探针显示，越过前两次编译后，GPU 更新约0.76秒，CPU约2.2秒。此前为了与原点云训练及大批量GPU评测并发，曾安排CPU长训练；后续采用GPU排队。小批量编译耗时与稳定更新耗时分别测量，完整数值及样本范围见 `../artifacts/verification/cpu-gpu-timing.json`。

## 数据与文档

每个真实运行独占`results/runs/<task>/<method>/<run_id>/`；Task／Method 属于目录语义，`run_id`只标识一次执行，重训使用新标识。运行记录完整配置、来源、预算、种子、实际设备、参数摘要和终止事件。`training.warm_start`保存参数来源；完整恢复保存优化器、随机数和环境状态的来源。

最终／当前最好结果不复制成另一棵产物目录，而由`results/selected/<目标>.json`引用已有 run、report 和 checkpoint。第一版入口为`selected/v1-18-cells.json`。预览、诊断、迁移材料和历史复制包放`results/scratch/`，不能据目录名提升验收状态。读取运行统一使用`artifacts.layout`按稳定`run_id`定位。

完整 RScope/MuJoCo replay 默认关闭；需要可视化证据时显式设置`evaluation.record_replays=true`。数值报告、失败回合和必要 trace 与 replay 开关独立保存。清理前核对运行进程和`selected/`引用关系，将唯一成功权重、正式回放和唯一未提交源码备份到项目外并验证摘要。

稳定术语放 `CONTEXT.md`，组件职责放 `architecture.md`，命令放 `runbook.md`，当前状态放 `status.md`，尚未完成事项放 `backlog.md`。已经结束的计划、控制台转录和阶段复盘归入Git历史与独立备份。

## 公开交付

发布源码前检查依赖锁、许可证和来源、文档链接、主机绝对路径、凭据模式、构建和测试。权重、回放、环境和依赖缓存遵守忽略规则，作为独立产物包管理。

本地提交只包含明确核对的文件。源码快照、结果包和完整Git历史分别检查；历史提交中的本机信息需在发布历史前单独审计。远端推送、仓库可见性和发布操作由用户明确授权。
