# Drone Playground

基于 Crazyflow 的无人机学习与优化控制研究平台。首版使用 Crazyflie，训练统一使用
Brax，轨迹查看统一使用 rscope。P1/P2 已获用户验收；P3/P4 完成24格跟踪实验、
三算法竞速训练与两种真实优化控制的独立评测，29项实验中25项达到当前门槛。
PPO/APG竞速成功；SHAC竞速的低分及全部训练证据保留。

## 从这里开始

- [当前进度与下一步](docs/status.md)
- [已接受的范围与规格](.scratch/drone-platform/spec.md)
- [阶段进度表](.scratch/drone-platform/map.md)
- [模块设计](docs/architecture.md)
- [评测与交付标准](docs/evaluation.md)
- [运行与远程查看](docs/runbook.md)
- [智能体执行与交接](docs/agents/workflow.md)
- [来源及复用清单](docs/research/references.md)
- [P3/P4 完整结果与检查点](docs/verification/p3-p4-results.md)

## 项目位置

本项目与 `../crazyflow/` 并列，各有独立 Git 历史。Crazyflow 提供仿真、动力学和控制器；
这里保存任务适配、训练扩展、优化控制接入、评测及记录。

## 首版交付

轨迹任务采用 Crazyflow 八字和 LSY 随机样条；竞速采用 LSY 原生门序与判定。
静态、动态导航均在首版范围内，状态和虚拟 MID-360 优先。PPO、APG/BPTT、SHAC 使用
同一 JAX 学习体系；优化方法先接已有采样 MPC 与 LSY AttitudeMPC。

算法是否接通、完整实验是否跑完、策略是否达标分别记录。已继承的 Brax 短程测试只证明接入能力。
项目交付还包含可加载的成功策略、独立评测结果和可远程重放的完整轨迹。

## 当前可用的内容

`pixi run train`、`evaluate`、`demo`、`replay`、`metrics`、`status` 已有真实执行入口。
操作见 [运行手册](docs/runbook.md)，检查点重载结果见
[独立评测](docs/verification/p2-independent-evaluation.md)，观察链证据见
[P1 观察验收](docs/verification/p1-observation-checks.md)。

当前结果覆盖四种动力学、状态与参考轨迹观测、训练种子0，以及原生扰动下的固定赛道。
三种学习方法和两种优化控制均已接通。感知导航、多训练种子和导航规划器扩展进入P5/P6，
可复算结果与数值故障修正范围见完整报告。
