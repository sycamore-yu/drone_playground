# 现役状态

## 2026-10-04：架构重构已实现并完成定向验证

已批准的环境构造、统一物理执行、配置入口、数值函数、训练编排、持久化、回放和 ROS 分工已落实。六组件、Brax State 与各后端原生动力学保持不变。公共执行资源由环境初始化代码建立并释放；Task 保留任务特有初始化和规则。时钟改为 `env.freq` / `env.physics_freq`，公共 `ActionTransition` 管理子步、动作到达、碰撞累计和首次终止冻结。

`composition.py` 和旧运行入口已删除。CLI/Python 共用 `configuration.py` 的配置准备与 `app.py` 的执行分派。共享数值函数、任务奖励、导航回放、RScope 发布及 ROS launch 分别归入各自模块；三个循环训练入口共用记录/快照调度。具体职责见 [architecture](architecture.md)。

验证记录见 [架构重构验收](../artifacts/verification/architecture-20261004/README.md)：248 个独立定向用例的最新结果全部通过；九类环境的 388 个数组与迁移前基线在 `rtol=atol=3e-6` 内一致。实际完成 GPU BPTT 更新、PPO/BPTT/SHAC/DVA 与循环策略短程训练、BPTT/SHAC 恢复一致性、RScope 读回及原生 C++/ROS 接入。SUPER 短闭环执行 17 次原生控制，EGO 执行 46 次。wheel 在仓库外导入并完成 JIT reset/step。

16,124 个受保护文件的 SHA-256 全部一致，无删除或覆盖；历史权重、报告、资产和黄金夹具保持不变。没有运行全量质量基准或声称策略收敛。核心 Ruff `F,I` 与改动 Python 格式检查通过；全规则 Ruff 仍有 565 项报告，主要为文档字符串及其他规范项，详见验收日志，不能声明全量 lint 通过。本次没有 Git 提交、推送或发布，既有未提交内容保持在工作区。

## 2026-10-03：外部接入整合

外部运行代码已统一到 `integrations/`：`ros1/` 是当前 EGO/SUPER 适配器，`rpc/` 是内部协议实现，`service.NativeServicePlanner` 保留通用外部服务接入。部署文件在 `docker/ros1/`；C++ 互通夹具在 `tests/native/`，直接实现 gRPC 生成接口。根目录 `native/`、包根目录 `rpc/` 和自定义 SDK 基类已删除。

方法、Pipeline、Controller 和记录统一使用 `Decision`；纯轨迹在 Controller 内采样一次。EGO/SUPER 实际短闭环、C++ 互通和公共 `eval.py` 跟踪/导航入口已通过定向验证。没有重建容器、修改上游算法或运行全量回归。

## 2026-10-03：冻结权重正式重跑

原 v1 清单中的 22 份独立学习权重已在当前代码上完成 2,600 回合正式评测。PPO、SHAC、BPTT 的跟踪与竞速六格均为三个种子各 100/100，跟踪 RMSE 均低于 0.25 m。Depth 静态三种子通过；动态 seed 30 的 D03 为 22/25，未达 23/25 门槛，另外两个种子通过。

PointCloud 仅有 seed 0 的开发权重：静态三主场景各 25/25，D03 为 13/25，动态未达标，且缺少另外两个确认种子。非学习方法只验收真实组合执行，不以成功率阻塞。当前不能声明所有学习格均通过正式质量要求。

矩阵编排与本次报告保存在 `tmp/matrix18/`，不进入公开 scripts API。Depth 的旧 seed=60000 对照另行保存，未用它替换现行 seed=80000 的正式结果。原权重和 v1 清单未覆盖。

## 2026-10-03：release 结构清理

发布树只保留包内 `src/drone_playground/{assets,benchmarks,configs}` 一份运行资源；
根目录同名 symlink、setuptools `build/`、重复 demo launcher 和空源码目录已清理。
第三方源码锁与通用补丁归入 `third_party/`，ROS 运行环境归入
`docker/ros1/`。Navigation 直接加载 `catalog.xml` 和各场景 MJCF，不再使用资产 SHA 门禁。
历史 JSON→MJCF 审查归档在 `artifacts/verification/navigation-mjcf/`，不属于运行依赖。

现役文档已同步到 Reference/Setpoint、RPC v2、MJCF 与上述目录职责。历史
`docs/notes/archive/` 和既有运行产物不按新路径重写。

## 2026-10-03：无 registry 的环境重构

本轮已完成六组件环境、统一 Dynamics.step、Reference/Setpoint、MJCF 资产和回放、公共训练与冻结评测、完整恢复、RPC v2、MPC/ROS 短闭环及安装包验证。没有运行全量回归。交付清单和实际验证日志见 [14项核对](../artifacts/verification/direct-composition-v4/README.md)。

该轮验证当时确认 632 个受保护文件及四个原始基准文件保持不变；这是
`direct-composition-v4` 交付时点的证据。随后本次 release 清理只更新当前发布树
的资源表示与路径，不回写旧运行、权重或原质量结论。v3 检查点通过显式复制迁移后
用于 v4 运行。

更新日期：2026-10-03。唯一开发仓库为 `main`；当前为研究预览。

历史第一版18格的交付判定保存在 [release-progress](../artifacts/verification/release-progress.json)，不代替上方当前冻结权重重跑结论。只有学习方法要求质量门槛；Depth 动态和 PointCloud 的当前阻塞按上方报告处理。

## 已完成

- PPO／SHAC／BPTT 的跟踪与竞速 6/6 格已完成多种子确认。
- AttitudeMPC／SamplingMPC 的跟踪与竞速 4/4 格已通过；AttitudeMPC 竞速使用单列的38ms因果延迟适配。
- 深度可微飞行保留历史修订判定；当前正式重跑的静态三种子通过，动态 seed 30 未达标。
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
