# Drone Playground

基于 Crazyflow 的无人机学习与优化控制研究平台。首版使用 Crazyflie，训练统一使用
Brax，轨迹查看统一使用 rscope。当前已完成项目设计与目录建立，正式训练入口将在任务 01 交付。

## 从这里开始

- [当前进度与下一步](docs/status.md)
- [已接受的范围与规格](.scratch/drone-platform/spec.md)
- [阶段进度表](.scratch/drone-platform/map.md)
- [模块设计](docs/architecture.md)
- [评测与交付标准](docs/evaluation.md)
- [运行与远程查看](docs/runbook.md)
- [智能体执行与交接](docs/agents/workflow.md)
- [来源及复用清单](docs/research/references.md)

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

阅读上述规格、阶段任务和验证报告；原有接入实验见
[历史验证](docs/verification/2026-09-25-brax-integration.md)。
目录中的模块说明定义后续实现位置；训练、评测和远程查看命令的实际可用状态以
[运行手册](docs/runbook.md) 为准。
