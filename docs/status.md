# 当前进度

更新时间：2026-09-27。原训练时间戳为2026-09-26 UTC。P1/P2已由用户验收；P3/P4及本轮可组合架构/LOTF交付完成。
当前分支：`implementation/composable-lotf`；基准`bd14468`。
本轮迁移前47项基线、五模块回归与LOTF上游对照已完成；最终79项测试、2个子测试通过，
命令、日志和统计见[最终检查](verification/composable-lotf-final-checks.json)。

## 当前工作

已按用户本轮意见写入 [可组合架构规格](../.scratch/composable-flight/spec.md)：
控制器合并跟踪与飞控内环；策略包含轨迹规划；动力学包含电机；训练配置收敛为四类。
LOTF范围明确为高保真前向＋简化反向＋原生BPTT，在线残差学习和策略交替更新列后续扩展。
按 [模块地图](../.scratch/composable-flight/map.md) 协同实现，最终交付悬停与八字的训练结果。
五个模块工作单均已解决。LOTF原始源码作为固定子模块保存在`third_party/learning_on_the_fly`，
悬停600万、八字2250万次训练交互均完成，各保存9个初始/中间/最终策略；开发和留出评测在独立进程执行。
实际计划与账本见 [实施计划](../.scratch/composable-flight/implementation.md)。
参考证据在 [开源编排比较](research/composable-platform-references.md)。

| 当前交付任务 | 开发集完整回合 | 留出集完整回合 | 留出全程位置RMSE | 留出最后一秒RMSE |
|---|---:|---:|---:|---:|
| LOTF悬停，3秒 | 32/32 | 128/128 | 0.424420米 | 0.076966米 |
| LOTF八字，5秒 | 32/32 | 128/128 | 0.184752米 | 0.164008米 |

悬停全程含随机初态的收敛过程。最终复核另跑两组各128留出，参数摘要和逐回合RMSE与原报告完全一致。
共同交付见[训练结果与模型](verification/composable-lotf-delivery.md)；
GPU续训的4.59e-6最大参数差及CPU逐元素一致的不同验证范围见[恢复核验](verification/composable-lotf-resume-check.json)。

当前可直接在RScope Viewer打开：

```text
experiments/lotf-hybrid-hover-seed0-v1/independent-heldout/rollouts/
experiments/lotf-hybrid-tracking-seed0-v1/independent-heldout/rollouts/
```

## 阶段结果

| 阶段 | 实际完成 | 质量结果 |
|---|---|---|
| P1/P2 | 已验收；四组跟踪策略与原证据保留 | 四组均128/128 |
| P3 | 两任务×三算法×四动力学，共24格；复用4组P2 | 21格达标，3格有效低分 |
| P4a 优化控制复现 | 每步真实优化；每方法128个原生扰动试次 | 采样MPC128/128；AttitudeMPC117/128，11碰撞 |
| P4b 竞速学习 | PPO/APG/SHAC各完成声明预算及独立32+128回合 | PPO/APG均128/128；SHAC0/128 |

总计29项实验完成、25项达到当前门槛。最终事实入口为
[实际结果表](verification/p3-p4-results.md)与[逐项校验JSON](verification/p3-p4-results.json)。
正式训练交互累计41,156,608次（含复用P2），独立开发/留出评测累计4,576试次。
SHAC竞速初始策略被开发集选中，属于训练质量未达标；最终训练权重、曲线和失败轨迹完整保留。

## 直接查看

在VS Code的RScope Viewer中点击以下目录内的 `.mj_unroll`：

```text
experiments/p4-racing-ppo-first_principles-seed0-v1/independent-heldout/rollouts/
experiments/p4-racing-apg-first_principles-seed0-v1/independent-heldout/rollouts/
experiments/p4-racing-shac-first_principles-seed0-v1/independent-heldout/rollouts/
experiments/p4-racing-attitude-mpc-heldout-v2/rollouts/shard-0/
experiments/p4-racing-sampling-mpc-heldout-v2/rollouts/shard-0/
```

每文件实际保存4–5个固定/最差试次，所有128试次在对应报告中。
查看训练结束的SHAC策略使用同运行 `rollouts/step-0000655360/`；留出目录显示开发集选中的策略。
TensorBoard沿用 `127.0.0.1:6006`；三算法的完成率、门进度、损失/梯度标签已实际读取。

## 故障与范围

P3随机样条PPO拖曳模型v1与降低学习率v2均发生非有限参数。诊断捕获到接近欧拉角奇异区域后
角速度达到8.119e26，溢出观测二阶矩与价值损失。v3恢复原学习参数，显式启用数值失败边界，
异常步计失败并重置，普通步保持原行为；最终4194304次交互、留出128/128、误差0.01311788米。
旧运行及捕获数据保留，协议差异见[数值诊断](verification/p3-numerical-diagnosis.md)。

竞速优化早期v1误读扰动配置层级，已停止并作为诊断保留。正式v2使用 `env.disturbances`，
合并时验证30000..30127每个种子恰好出现一次。P4比较作者固定样条上的闭环控制与过门，
自由最短时间规划、MID-360导航和多训练种子分别进入后续任务。

## 验证与交接

- [完整工程检查和来源](verification/p3-p4-engineering.md)
- [P3/P4交付记录](verification/p3-p4-delivery.md)、[五种方法实际VS Code交互](verification/p4-editor-verification.json)
- [P3原矩阵回放](verification/p3-replay-verification.json)、[修复单元回放](verification/p3-recovery-replay-verification.json)
- [P4全部14份回放](verification/p4-replay-verification.json)、[实际竞速画面](verification/p4-visual-check.json)
- [阶段地图](../.scratch/drone-platform/map.md)、[实施方案](design/p3-p4-implementation.md)、[运行手册](runbook.md)

`p3-matrix.json`保留初次队列23/24的历史；最终结果表包含v3恢复后的24/24。
47项测试通过；62个回放文件、308条保存轨迹逐帧还原误差为0，原文件摘要保持一致。
严格独占资源性能排行及峰值显存测量仍为后续待测项；本轮计时保留共享服务器的实际口径。
原DSH委派因配额失败，独立ChatGPT审查因浏览器启动失败，均未执行代码；实现、测试及收尾由当前主会话完成。
本轮结束后无LOTF正式训练等待完成；已保存结果可直接回放和重评。
后续按总体P5/P6推进感知导航、多训练种子及统一性能口径；LOTF在线适应属于另立范围。
SHAC历史低分按用户判断暂时保留为后续研究项，当前不扩大调参任务。
