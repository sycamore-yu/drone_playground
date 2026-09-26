# 学习模块

`train.py` 调用原生Brax PPO/APG及`shac.py`；`lotf_bptt.py`保持作者的确定性策略、
时间展开、回报总和、随机数和Adam更新。四组配置为algorithm、network、objective、training。
网络构造和唯一奖励入口分别位于`networks.py`与`objectives.py`，LOTF任务奖励直接复用上游。

PPO支持原生参数暖启动；SHAC与LOTF保存优化器、随机数及环境状态。LOTF完整实验续训
同时继承恢复时刻以前的开发集最佳模型，来源存入`resume-selection.json`，未来评估不参与选择。
CPU小规模保存恢复与连续更新逐元素相同；跨进程GPU正式快照恢复的浮点差异单独报告。

LOTF两任务正式训练及独立评测见`docs/verification/composable-lotf-delivery.md`。
