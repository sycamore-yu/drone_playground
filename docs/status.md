# 当前进度

更新时间：2026-09-29。现役工作树：`simulation_dev/.worktrees/drone_playground/architecture-v3`；当前分支`main`。架构分支已合入64838b5，后续开发在main完成，配置版本为3。

本次已执行六方法×五任务的30项实测。PPO、BPTT、SHAC、点云控制迁移的悬停／跟踪／竞速共12项均通过各32回合留出质量验收。SHAC竞速明确为BPTT热启动后真实SHAC更新；其余热启动来源、训练预算和冻结检查点见[验收矩阵](verification/final-acceptance/README.md)。

导航协议升级到300秒，原生规划器速度上限20米/秒，到达／失败终止录制。标准学习及本轮点云控制训练启用25–50毫秒逐回合随机命令延迟。所有方法使用真实执行路径；SUPER和EGO控制适配的备用控制帧数、原生轨迹与指令数分别保存。

SUPER静态／动态导航各6/6到达。EGO两类导航各0/6到达、6次碰撞。PPO/BPTT/SHAC六组导航均完成4096交互、真实更新和冻结评测，各0/6到达、6次碰撞；收敛按本轮授权保留为后续质量事项。原生三控制任务均完成真实规划器调用，精度和竞速失败保留。

原点云论文开发选中的30000更新权重经摘要校验迁入当前主线，八场景×四速度32格评测为0到达、29碰撞、3超时。原50000更新训练及协调器保留在pointcloud-paper工作树，属于独立原始配方；当前点云控制迁移的条件化网络与任务适配使用独立身份。

主要修复包括射线未命中分支的非数值梯度、全部终止批次的停算、点云状态输入饱和、SHAC热启动后参数变化核验、EGO三维目标传递、SUPER控制轨迹初始化，以及原生竞速起飞接管。正式结果、原始候选、命令和摘要在[本次验证](verification/final-acceptance/README.md)。测试、静态检查和源码冻结记录在该目录的verification.json与source-hashes.json。

旧架构验证274项及其运行结果保留在[历史验证](verification/architecture-v3/README.md)。本轮命令见[操作手册](runbook.md)和[验收命令](verification/final-acceptance/commands.md)，实际目录见[项目目录](project-tree.md)。

原主工作树refactor/native-planner-runtime保留原状态。research/pointcloud-paper-navigation8相对main独有提交为0，停止新增开发，保留冻结训练来源和结果；分支处置见[生命周期](research/branch-lifecycle.md)。LOTF仍为独立方法。

后续质量事项为学习导航、EGO导航、原生时间跟踪／竞速和原论文50000更新最终评测；Archify版本3交互图、P5历史归档及LOONG/AERO-MPPI/AC-MPC按[待办](backlog.md)独立维护。
