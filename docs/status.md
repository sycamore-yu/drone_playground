# 现役状态

更新日期：2026-10-02。唯一开发仓库为 `main`；当前为研究预览。

第一版18格中，14格按现行质量标准通过（12格原标准、深度2格修订标准），EGO两格由用户接受交付，共16/18格满足当前交付要求；剩余点云静态／动态两格继续开发与独立验收。权威进度见 [release-progress](../artifacts/verification/release-progress.json)。

## 已完成

- PPO／SHAC／BPTT 的跟踪与竞速 6/6 格已完成多种子确认。
- AttitudeMPC／SamplingMPC 的跟踪与竞速 4/4 格已通过；AttitudeMPC 竞速使用单列的38ms因果延迟适配。
- 深度可微飞行静态／动态两格按修订的主要场景标准通过；原完整结果和原判定继续保留。
- SUPER 静态／动态均为96/100；心跳互斥修复与回放证据已归档。
- EGO-Planner 静态／动态由用户接受交付并停止调试，不作为正式质量通过格。
- LOTF 已退役为独立 Task／Method／Evaluation，只保留 high-fidelity 与 simplified 两种动力学。
- Training 与 Evaluation 复用同一 Task；现役运行角色仅为 `train`／`eval`，训练内 checkpoint evaluation 与正式 Benchmark 不再使用独立开发集语义。
- 动力学随机化、测量／动作噪声、初态、任务指令、场景分布、外力与鲁棒性 sweep 已按独立概念接入。
- 结构与配置迁移收尾已完成：真实训练／冻结评测／播放、生成场景梯度和 Sampling MPC 定向检查通过；选定结果与原回放引用已修复，原质量结论及权重不变。凭据见 [迁移收尾检查](../artifacts/verification/structure-config-refactor-20261002.json)。

## 进行中

- 点云1.5m安全余量配方当前开发结果：静态26/32、动态31/32；主要场景中D03为7/8，尚未达到逐场景≥90%的开发门槛。
- 在D03开发门槛通过前，不启动训练种子50／51／52及正式 benchmark 确认；下一步只围绕该失败继续诊断或训练。
- 本轮按范围执行定向验证，未运行大规模重构后的全量回归；当前结果不作为全套测试通过或新训练质量通过的声明。

## 约束

- 固定 Navigation8 的评测只证明独立初态／扰动下的表现，不声称未见几何泛化。
- 不使用 benchmark 结果重新选择训练配方；学习方法按训练种子逐个判断，不平均掉失败种子。
- 远端推送、公开发布和源码快照发布仍需单独授权。

当前架构见 [architecture](architecture.md)，正式评测语义见 [evaluation](architecture.md)，剩余工作见本页进行中事项。历史研究过程在 `docs/notes/`，机器证据在 `artifacts/verification/`。
