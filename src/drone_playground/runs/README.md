# 运行记录模块

`record.py`、`console.py` 和 `rscope_io.py` 已连接运行标识、任务、代码、配置、日志、指标和轨迹。
复用 rscope 原生导出/读取，TensorBoard 只展示标量。记录状态和低频进展更新，支持真实会话恢复。
实时发布是可选显示环节：切换查看对象不改变已保存的训练结果，显示错误单独记录。
完整记录契约见项目 `docs/agents/workflow.md`。
