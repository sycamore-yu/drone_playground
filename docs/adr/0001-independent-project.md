# 独立研究包依赖 Crazyflow

已接受。平台在 `simulation_dev/mujoco/drone_playground/` 建立独立 Git 历史，Crazyflow
继续提供原生仿真与动力学。训练、任务适配和结果记录集中在新项目，可以固定依赖版本并单独审查
研究改动；核心仿真缺陷以小补丁回到 Crazyflow，运行清单记录双方提交和工作树补丁。
