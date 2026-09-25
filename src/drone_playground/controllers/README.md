# 优化控制模块

`demo.py` 已调用 Crazyflow 原生 Mellinger 状态控制，完成完整八字飞行。采样 MPC 和
LSY AttitudeMPC 的正式比较在任务 06 接入，保留作者计算和执行含义。
优先沿用 compute_control 和原有回调。实际输出类型、时序及预测模型进入方法配置。
新增导航规划方法按任务需要决定，模块中不预建未来算法的空实现。
