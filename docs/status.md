# 现役状态

更新日期：2026-09-29。主开发分支为 `main`，唯一开发位置为 `simulation_dev/drone_playground`。当前为研究预览；已确认的[第一版18格规格](release-plan.md)尚未全部实现或达到质量门槛，当前正在接口实现与开发集训练阶段。

历史基线采用[六方法、五任务正式验收](verification/final-acceptance/README.md)。主目录 `experiments/` 保留29个评测运行、18个对应训练来源和两组必要来源权重；选定参数、逐回合结果和正式回放均有独立备份。30格、488回合、19份选定权重和245份回放已重新核验，原始质量结果保持不变，不能据此宣称新的18格目标已达标。

三类公共物理消息Trajectory／Waypoint／具名MotionCmd已有Protobuf合同、Python客户端与C++ SDK，生命周期和真实C++进程互操作已验证。SUPER／EGO沿用ROS1容器，通过同一gRPC服务提供完整轨迹。下游可选择原跟踪器、AttitudeMPC或SamplingMPC；SUPER→AttitudeMPC已完成小样本闭环，但未达到全部场景质量要求。当前通用服务评测只接Trajectory，Waypoint／MotionCmd统一执行、多模块任意兼容串接仍未完成，已纳入持续goal，详见[组合验收](implementation-plan.md#三类物理接口的组合验收)。

深度飞行已有独立JAX导航适配、CNN／GRU和真实场景梯度检查；按用户确认采用D435i标称87°×58°视场及10m截止量程。当前仍是共同条件的组件适配，上游CUDA完整物理与损失复现尚未完成。传感来源、分辨率、时钟和缺项见[实现边界](research/depth-flight.md)。

首轮开发集：BPTT／SHAC／PPO跟踪及BPTT／PPO竞速各完成seed=0训练，选定快照均通过32个开发回合；SHAC竞速首轮0/32，正在诊断。真实SUPER原跟踪器小样本3/3到达；EGO首轮3例碰撞，发现原固定地图尺寸不足覆盖100m导航任务，已修复任务范围传入，待重新评测。所有这些均不是正式三种子／100回合18格质量验收。

本轮gRPC与MPC改动后的完整回归为395项及5个subtest通过；随后新增深度D435i配置的10项检查通过。后续改动仍需对应验证，不能把该数字当成未来所有变更均已测试。运行合同与真实study见[开发训练合同](training-pilot.md)。

原点云长训练已按用户要求结束：45000次完整检查点已校验、独立备份，并在旧目录退役后使用现役解释器成功读回。最后日志为49760次，未保存部分不计作可恢复状态；原50000次预算未完成，最终评测未启动。训练和协调器均已停止，冻结 `pointcloud-paper` 工作树保留源码与证据。

旧 `mujoco/drone_playground` 已移入项目外离线归档，原路径不存在。现役目录拥有独立 `.git` 与从固定版本重建的 acados；全部分支、标签和未提交内容得到保留。具体顺序和备份位置见[生命周期](research/branch-lifecycle.md)。

源码清理保留最新 Navigation v2 及仍有效的几何证据，删除旧场景 v1–v4，合并重复奖励配置。当前场景字节与两个导航环境的解析配置保持不变。临时设计表已并入[方法主表](methods.md)，实现顺序与剩余研究统一见[待办](backlog.md)。本轮验证见[清理与退役凭据](verification/cleanup-design-retirement.json)，此前的[历史清理记录](verification/release-cleanup.json)保持原样。

当前源码通过本机路径与常见凭据模式检查。历史 Git 对象中仍保留本机路径，公开材料采用确认提交的源码快照；本轮仅生成本地预览包，远端推送和发布另行授权。
