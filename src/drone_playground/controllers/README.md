# 优化控制模块

待任务 01、06 使用。复用 Crazyflow sampling.py 与 LSY AttitudeMPC，保留作者计算和执行含义。
优先沿用 compute_control 和原有回调。实际输出类型、时序及预测模型进入方法配置。
新增导航规划方法按任务需要决定，模块中不预建未来算法的空实现。
