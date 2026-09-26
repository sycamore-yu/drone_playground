# Drone Playground

基于 Crazyflow 的无人机学习与优化控制研究平台。Hydra组合策略、控制器、动力学、
场景、观测及任务，训练使用Brax/JAX和同一体系的LOTF原生BPTT适配，轨迹查看使用rscope。
Crazyflow配方使用Crazyflie，LOTF配方保留作者原机型。P1/P2 已获用户验收；P3/P4 完成24格跟踪实验、
三算法竞速训练与两种真实优化控制的独立评测，29项实验中25项达到当前门槛。
PPO/APG竞速成功；SHAC竞速的低分及全部训练证据保留。

## 从这里开始

- [当前进度与下一步](docs/status.md)
- [已接受的范围与规格](.scratch/drone-platform/spec.md)
- [阶段进度表](.scratch/drone-platform/map.md)
- [模块设计](docs/architecture.md)
- [可组合架构与LOTF训练规格](.scratch/composable-flight/spec.md)
- [组合架构的模块工作地图](.scratch/composable-flight/map.md)
- [评测与交付标准](docs/evaluation.md)
- [运行与远程查看](docs/runbook.md)
- [智能体执行与交接](docs/agents/workflow.md)
- [来源及复用清单](docs/research/references.md)
- [P3/P4 完整结果与检查点](docs/verification/p3-p4-results.md)
- [LOTF悬停、八字及模块化交付](docs/verification/composable-lotf-delivery.md)

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

`pixi run experiment`、`train`、`evaluate`、`simulate`、`demo`、`replay`、`metrics`、`status` 已有真实执行入口。
操作见 [运行手册](docs/runbook.md)，检查点重载结果见
[独立评测](docs/verification/p2-independent-evaluation.md)，观察链证据见
[P1 观察验收](docs/verification/p1-observation-checks.md)。

当前结果覆盖四种动力学、状态与参考轨迹观测、训练种子0，以及原生扰动下的固定赛道。
三种学习方法和两种优化控制均已接通。感知导航、多训练种子和导航规划器扩展进入P5/P6，
可复算结果与数值故障修正范围见完整报告。

可组合架构已经迁移现有任务、控制器、网络、动力学和评测。LOTF高保真前向＋解析反向＋BPTT
已完成悬停600万、八字2250万交互，独立留出均128/128完整回合；悬停最后一秒误差0.07697米，
八字全程误差0.18475米。查看原记录即可复核，使用新运行名才会启动新的训练。

```bash
pixi run experiment --cfg job experiment=lotf_hybrid_hover
pixi run train experiment=lotf_hybrid_hover run_id=my-new-hover
pixi run train experiment=lotf_hybrid_tracking run_id=my-new-tracking
```

LOTF模块直接复用固定GPLv3子模块，源码和来源见`THIRD_PARTY_NOTICES.md`。在线适应、视觉和实机部署另列后续范围。
