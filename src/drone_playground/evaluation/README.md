# 评测模块

`evaluator.py` 路由冻结神经方法、MPC 和原生规划器；各任务文件负责独立试次、事件聚合及其原有指标。闭环调用顺序由 `runtime/` 负责，末次终止转移和全部失败分母均保留。

`protocols.py` 校验具名评测条件与场景摘要。`native_planners.py` 和 `optimization.py` 管理各自资源、记录与统计，并复用同一个外部方法运行循环。点云的八场景、三速度冻结测试由 `pointcloud.py` 执行。

公开检查点入口默认沿保存的环境和时序；`env=...` 显式替换整套兼容环境，`env.execution.dynamics.forward=...` 只覆盖具名动力学字段。运行配置、原始事件、参数摘要及预算/质量结果单独保存。
