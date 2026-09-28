# 控制器模块

同一个controller预设拥有需要的跟踪、飞控内环和命令转换；动力学保存机体与电机状态。
`crazyflow.py`接通姿态命令，`lsy_mpc.py`运行原生acados，`sampling.py`运行真实候选预测；
`lotf.py`复用作者Betaflight风格内环。LOTF每个物理子步只调用一次控制器。

`factory.py`从同一实验配置构造优化控制器。策略可提供固定或随机轨迹；命令类型、预测模型、
控制频率随运行记录。两种MPC已有P4正式结果，模块化迁移又核对了真实求解和完整过门。
`demo.py`继续提供Mellinger固定示例；新方法在已有调用位置添加适配。
