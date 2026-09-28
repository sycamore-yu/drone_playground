# 任务模块

`tracking.py` 包含悬停、八字和随机样条；`racing.py` 管理 LSY 竞速门事件；`navigation.py` 管理静态/动态导航的到达、碰撞、越界与超时；`lotf.py` 和 `pointcloud.py` 保留具名论文任务语义。

参考生成位于 `methods/planners/reference.py`，由需要参考的任务显式选择。环境组合入口注入场景、传感器、观测、执行和任务目标，重置与步进保持模型原生状态。任务使用执行层产生的物理证据裁决结束条件。

LOTF 跟踪参考按 50 Hz 索引，频率变化需要具名重采样适配。随机化、种子、时序和任务协议进入运行记录，现役导航资产统一从 `assets/scenes/navigation/catalog.json` 读取。
