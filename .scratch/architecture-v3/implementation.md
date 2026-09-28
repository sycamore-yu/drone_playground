# 可组合架构第三版实施计划

执行者在当前会话按 executing-plans 逐项实施，使用 test-driven-development 验证新增行为。

目标：将已有全部方法迁入可组合架构，交付真实可运行的训练、评测和展示入口。
架构：方法决策、环境任务、执行转移、运行编排、学习更新和评测统计分别拥有复杂性；同一实现保持唯一来源。
技术：JAX、Brax、Hydra、Crazyflow、MuJoCo/RScope、既有 ROS/acados。
规格：[spec.md](spec.md)。执行账本：[progress.md](progress.md)。

## 全局约束

优先 JAX；使用 dynamics 和 navigation 命名；原点云工作树和训练保持独立；预算/质量分别记录。
临时产物在 tmp/，正式测试在 tests/，每次验证运行在独立 experiments/ 目录。
主工作树修改和原运行均不作为本次提交内容；本地重组分支独立提交。

## 审查焦点

检查点序列化类型路径变化；传感器和观测配置冲突；重复执行控制；超时与终止混淆；原生规划器失败时生命周期清理。

## 任务 1：来源快照及失败测试

- [ ] 合并点云固定提交，解决唯一文档冲突，保留原工作树身份；记录完整基线测试。
- [ ] tests/test_architecture_v3.py 覆盖公开 method/env 装配、纯优化拒绝训练、核心环境列表和命令接口。
- [ ] 运行新测试确认旧架构缺少预期能力。
- [ ] 保存迁移前的配方和点云数值对照数据到 tmp/architecture-v3。

## 任务 2：配置及物理模块迁移

- [ ] 重组 methods、networks、environments、execution、models、learning/algorithms、learning/objectives、visualization。
- [ ] 更新全部源码、配置、测试和现役工具的引用；旧公开路径移除。
- [ ] 升级配置版本与 env/method 配方，源配置独立选择 sensor，导数选择归 algorithm.gradient。
- [ ] navigation 资产与协议分开，检查摘要；LOTF 来源改为固定版本依赖。
- [ ] 回归新装配测试及已有关联测试后提交。

## 任务 3：环境、执行与闭环运行

- [ ] execution/transition.py 集中控制映射和物理推进，以实际模型验证单次执行及前向/导数等价。
- [ ] runtime/jax_runner.py 与 runtime/host_runner.py 集中闭环，评测与展示共用。
- [ ] methods/optimal_control 和原生规划器适配遵循方法契约，优化目标和预测模型归方法。
- [ ] 环境任务拥有结束条件，训练目标通过学习适配接入。
- [ ] 新增独立 PPO 悬停任务及测试，其余任务保持原有协议。
- [ ] 运行控制、事件、梯度和不同方法替换测试后提交。

## 任务 4：入口、产物与点云迁入

- [ ] scripts/train.py、eval.py、play.py 调用同一装配路由。
- [ ] 配置冻结、显式版本迁移、参数重载和完整续训分别验证。
- [ ] 运行目录防覆盖、回放只读、正式评测完整分母纳入入口测试。
- [ ] 点云方法参数/前向/梯度对照及短续训验证；原50000更新运行单独保持。
- [ ] 回归后提交。

## 任务 5：真实验收与交付

- [ ] 完整 pytest、静态导入与格式检查。
- [ ] 核心任务 PPO 小预算训练及独立评测；两原生规划器静态/动态代表场景；两 MPC 实际运行。
- [ ] 点云新配置真实训练/冻结评测/恢复；导出回放并读回。
- [ ] 更新 README、CONTEXT、architecture、runbook、status、backlog、来源和验证文档。
- [ ] 整体差异自审、完整实际目录、git diff --check 与最终提交。
