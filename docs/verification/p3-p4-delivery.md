# P3/P4 交付记录

2026-09-26。项目：`/home/tong/tongworkspace/simulation_dev/mujoco/drone_playground`；
分支：`implementation/p3-p4`。P1/P2已由用户验收；本轮完成P3/P4工程实现和单训练种子首轮实验。

## 实际范围与结果

P3包含PPO/APG/SHAC × 八字/随机样条 × 四种动力学，共24格；4个P2单元只读复用。
24格均完成声明预算及独立评测，21格达到当前任务门槛。P4包含三种竞速策略训练及两种真实优化控制。

| 方法 | 训练交互 | 留出完赛 | 碰撞 | 成功回合平均时间（秒） | 全体回合位置RMSE均值（米） |
|---|---:|---:|---:|---:|---:|
| PPO | 4,194,304 | 128/128 | 0 | 17.09859 | 0.014389 |
| APG | 655,360 | 128/128 | 0 | 17.05672 | 0.004991 |
| SHAC（开发集选中初始策略） | 655,360 | 0/128 | 0 | 无成功回合 | 1.761449 |
| LSY AttitudeMPC | 在线求解 | 117/128 | 11 | 17.10923 | 0.062754 |
| 采样MPC | 在线求解 | 128/128 | 0 | 17.01391 | 0.033627 |

29项正式实验完成，25项达到当前门槛；累计41,156,608次训练交互（含复用P2），
4,576个独立开发/留出评测试次。数值异常单元按具名v3恢复，两个失败版本保留在结果清单中。

竞速比较采用作者18.75秒预设参考样条、4个实体门和`[1,2,3,4,2]`指定门序，
原生动作/外力扰动、碰撞、过门方向与判定框，50Hz控制、最长30秒。
128试次使用独立扰动种子，赛道几何保持固定；三种学习方法共用43维状态/未来参考点观测和姿态动作。
自由最短时间规划、多赛道、视觉导航和多训练种子属于后续研究范围。

四项质量低分完整保留：P3基础拟合模型/随机SHAC、第一性原理/八字PPO、第一性原理/随机SHAC，
以及P4竞速SHAC。两项随机跟踪SHAC和竞速SHAC的开发集最优仍为初始策略；
最终训练权重、开发评估与`rollouts/step-0000655360/`另外保存，供核对真实训练后的行为。
失败回合的误差覆盖其有效步数，与完成率共同解释。

表格从实际文件重新核验并生成：

```bash
pixi run python scripts/summarize_p3_p4.py
```

该命令核验预算、检查点文件/参数摘要、独立评测身份、每个种子恰好一次、逐回合完成数及误差汇总。
控制器合并记录同时核验4个原始分片报告及其文件摘要。结果见`p3-p4-results.json`和`p3-p4-results.md`。

## 实现与来源

现有五模块继续承担任务、学习、控制、评测和记录。Brax原生PPO/APG直接调用，
SHAC扩展复用Brax网络和推理接口；短窗口目标、终端价值状态梯度、真终止/时间截断及TD-λ有独立测试。
SHAC完整续训状态包括网络、目标网络、优化器、环境、归一化和随机数，恢复结果与连续更新逐元素一致。

LSY固定提交`b1f5b36adb8e08e8e2adea85de790bd0e0a1d118`；必要源码、原文件摘要、
许可和最小兼容差异在`tasks/lsy_upstream/`与`controllers/lsy_upstream/`。
acados固定0.5.1，保留原25步/0.5秒时域、代价矩阵、SQP/HPIPM及原推力界限语义。
采样器保留精英算术均值、候选分布、暖启动和推力估计器；任务参考和障碍输入明确记录。

## 验证与远程观察

最终实际命令使用项目Python执行 `-m pytest tests -q`：47通过，224.89秒，退出0；
日志为`tmp/p3p4/final-tests-after-recovery.log`。
覆盖真实参数更新、梯度、续训、原生门事件/扰动/接触、选模、分母、重复种子、参数损坏及数值失效边界。

已交付RScope读取器逐帧核验62个文件、308条实际保存轨迹，原文件摘要保持一致；
最大位置还原误差为0，显式参考坐标逐项一致。报告为`p3-replay-verification.json`、
`p4-replay-verification.json`、`p3-recovery-replay-verification.json`。

本轮实际调用已交付RScope Viewer 0.1.0的读取器核对文件、模型、指标和逐帧坐标，
并用MuJoCo渲染保存的竞速PPO与采样MPC状态，检查门模型、纹理及参考/实际轨迹。
截图与对应帧号见`p4-visual-check.json`；这是保存轨迹的图形核验。
VS Code窗口交互沿用此前扩展交付的验证，本轮未重新完成五种方法的编辑器端交互测试。

TensorBoard继续使用6006端口；已实际读取三种竞速算法的指标标签和门进度字段。
轨迹文件只保存固定前4及最差试次的子集，完整32/128试次仍在逐回合报告中。

## 数值错误与历史修正

随机样条PPO在含阻力的拟合模型中出现角速度8.119×10^26，单个值保持有限，
观测二阶统计和价值损失已溢出。原版本和降低学习率版本均保留失败证据。
v3恢复原PPO学习参数，显式启用`finite-square-v1`数值失效边界：失效积分计失败，
保留最后有效物理状态用于终止记录，下一动作前重置；健康转移保持原实现。
该协议差异具名记录，v3完整训练后留出128/128，RMSE0.01311788米。
完整诊断见`p3-numerical-diagnosis.md`及同名JSON。

初期竞速适配误读扰动配置层级；种子行为测试识别并修正为`env.disturbances`。
无扰动控制器v1与探针保留为诊断；正式控制器为v2，三个学习方法均在修正后训练。

## 计时边界与后续测量

结果保存真实作业耗时、Brax原生计时、SHAC编译/更新分项和控制器逐步求解延迟。
采样控制实际执行设备为CUDA，AttitudeMPC为CPU；方法预算及并行度分别记录。
控制器分片并发时存在资源竞争，测得时延未注入物理步进。
独占硬件、统一计时边界和峰值显存的严格性能排行仍为P6待测项。

## 直接查看

在VS Code打开以下目录中的`.mj_unroll`：

```text
experiments/p4-racing-ppo-first_principles-seed0-v1/independent-heldout/rollouts/
experiments/p4-racing-apg-first_principles-seed0-v1/independent-heldout/rollouts/
experiments/p4-racing-shac-first_principles-seed0-v1/rollouts/step-0000655360/
experiments/p4-racing-attitude-mpc-heldout-v2/rollouts/shard-0/
experiments/p4-racing-sampling-mpc-heldout-v2/rollouts/shard-0/
```

先检查APG的过门和实际/参考轨迹，再对照AttitudeMPC中最差试次的碰撞过程。
