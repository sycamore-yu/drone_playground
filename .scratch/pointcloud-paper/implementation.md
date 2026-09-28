# 点云论文模块化复刻实施计划

执行：当前会话直接实现，使用测试先行与逐项验证；规格：[spec.md](spec.md)。
目标：公开配置能够构造论文式点云飞行方法，训练得到真实权重并在Navigation8独立测试。
技术：JAX、Flax、Optax、Brax存储/环境接口、现有RunRecorder和RScope。

## 全局约束

十个一级配置组保持，新增实现进入原有槽位。原论文缺失信息按规格公开。
正式配方32环境×160步×50000更新；工程短运行明确记录自身预算。
独立工作树复用主项目已安装依赖；通过PYTHONPATH明确加载本工作树源码。
训练/开发静态随机图元，Navigation8只作冻结迁移测试。

## 风险与验证

传感器空帧/掩码：输出有限且置换不变；射线与几何一致。
梯度规则：缩放只作用于声明的状态通路；动作通路完整；参数/观测/状态三类梯度分别验证。
循环状态：保存/恢复含优化器与随机数，每段初始GRU状态按规格重置。
事件：高速细杆穿越、同一步到达与碰撞、超时/非有限分别统计。
评测身份：八场景与三速度完整，参数hash前后一致，目录不可覆盖，RScope模型记录真实动力学身份。

## 任务一：物理与传感器

文件：dynamics/point_mass.py、controllers/acceleration.py、tasks/sensors/pointcloud.py；
扩展共享场景的球体距离与射线；tests/test_pointcloud_physics.py。
接口：PointMassState(pos,vel,acc,rotation)，PointMassLag.step(state,command,dt)；
UniformMid360Lidar复用已有cast_rays和最近有效交点契约；共享sphere距离/交点。
- [x] 写解析前向、雅可比、掩码与图元测试并看到失败。
- [x] 实现最小模块，运行本测试及既有传感器回归。
- [x] 提交模块与证据。

## 任务二：网络、观测、目标和训练环境

文件：learning/pointcloud_network.py、learning/pointcloud_objective.py、tasks/pointcloud.py、
tasks/scenes/pointcloud.py；tests/test_pointcloud_method.py。
接口：PointCloudPolicy.encode(points,valid)、act(embedding,proprio,hidden)；
PaperObjective(trajectory,dt)返回总损失及分量；PointCloudTask持有场景/物理/传感组件。
- [x] 先测试置换不变、可变点数、GRU重置/记忆、空点集和loss数值。
- [x] 实现32批量静态图元采样，公开来源范围。
- [x] 测试固定输入前向与梯度并提交。

## 任务三：训练、配置与恢复

文件：learning/pointcloud_bptt.py、runs/pointcloud.py、configs各槽位、composition.py；
tests/test_pointcloud_training.py。
接口：initialize(task,config)、make_update(task,network,optimizer,config)、save/load训练状态；
共享run_experiment分派新算法，训练/评测消费相同配置。
- [x] 先测试配置构造、参数更新与连续/保存恢复对照。
- [x] 实现scan/remat、AdamW、开发选模、真实预算和停止/恢复记录。
- [x] 真实GPU短训练测显存/吞吐，锁定可复现正式命令。

## 任务四：独立评测与回放

文件：evaluation/pointcloud.py、必要的共享回放模型参数；tests/test_pointcloud_evaluation.py。
接口：evaluate_pointcloud(config,root,run_id)按八场景/固定速度调用。
- [x] 先测事件/分母、固定场景身份与参数冻结。
- [x] 完成轨迹归档、RScope输出和读回验证，工程权重只作工程检查。

## 任务五：实际训练与交付

- [x] 提交全部运行代码；首阶段1000更新/512万交互完成，独立开发评测选模。
- [x] 在独立进程评测首阶段权重：24格全部完成并读回，0到达/24高度越界。
- [ ] 完整训练从原1000更新继续至50000更新，逐个里程碑保存完整状态及独立开发评测。
- [ ] 完整预算结束后，在独立进程评测开发集所选冻结权重并完成24格最终报告。
- [x] 汇总首阶段真实预算、结果、槽位配置、命令、来源偏差和回放；完整预算最终统计仍待运行完成。
- [x] DP-005保留进行中，记录已实现模块、阶段负结果及完整训练的恢复状态。
