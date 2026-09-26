# 01：组合配置与共同运行入口

Type: task
Status: resolved
Blocked by: none
Engineering: passed
Experiment: completed
Quality: passed
Owner: current-chat

## 模块职责

依据 [规格](../spec.md) 设计Hydra配置和构造入口。外部配置保留策略、控制器、动力学及环境，
训练保留算法、网络、目标和运行设置。控制时钟、状态传递和契约校验由组合入口处理。

## 实现证据

验证合法组合、输入输出类型不匹配、原生组合内控制器重复调用、状态投影和预设覆盖；
解析后配置随运行保存。迁移现有调用和命令说明，旧数据按明确版本读取。

## 共同交付

与02–05模块连接后完成两任务真实训练结果；本工作单不构成单独的用户验收阶段。

## Comments

2026-09-26：用户确认规格，要求完成整体模块化和LOTF悬停/八字训练；授权直接执行，计划见[implementation.md](../implementation.md)。

## Answer

Hydra分组和共同入口已完成，训练/评测/仿真/重放均有实际执行证据。
任务频率、动作接口、代理导数、模型用途和预算冲突在组合处校验；旧JSON配方一次迁移，
持久检查点通过只读格式解释进入同一构造链。历史训练字段不影响冻结评测。
两任务结果及命令见[共同交付](../../../docs/verification/composable-lotf-delivery.md)，
回归覆盖见`tests/test_composition.py`、`tests/test_execution_config.py`及`tests/test_lotf_review.py`。
