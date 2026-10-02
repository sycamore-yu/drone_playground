# 当前目录与配置职责

## 配置入口

完整预定义实验统一使用 Hydra 的 experiment configuration：

```bash
pixi run train experiment=learning/ppo env=tracking
pixi run train experiment=papers/pointcloud_diffsim
pixi run train experiment=papers/depth_diffphysics
pixi run eval experiment=papers/super env=navigation/static
```

`experiment/papers` 记录论文组合与来源；`experiment/learning` 保存通用学习基线；`experiment/transfers` 保存有独立实验设置的迁移。Environment 配置描述任务、场景、传感和执行系统，不表示论文身份。
解析结果中的 `method` 映射只记录实际执行实现、输出接口及算法身份；它不是另一个完整实验配置组。旧 `method=...` 命令改用 `experiment=...`。

## 组件职责

| 位置 | 职责 |
|---|---|
| `native/` | 独立 C++ gRPC SDK；保留已安装工具链位置 |
| `ros1_planners/` | EGO/SUPER ROS1 构建、补丁和服务桥接 |
| `rpc/` | Python gRPC 客户端、服务端与消息编解码 |
| `execution/commands.py` | 动作单位/坐标、Trajectory/Waypoint/MotionCommand、未来参考采样 |
| `execution/controllers/mpc/` | MPC 实现与预测补偿 |
| `planning/` | 真正的轨迹规划算法 |
| `environments/references.py` | 悬停/跟踪/竞速参考生成；不是避障规划器 |
| `environments/scenes/` | 空场景、竞速、几何、固定目录与随机图元生成 |
| `environments/sensors/` | LiDAR、深度相机和公共射线查询；论文使用的均匀扫描/针孔模型已归入对应传感类型 |
| `environments/tasks/` | 按任务分组，内部名称说明实际动力学/状态接口 |
| `networks/` | 按网络结构命名，不按论文用途命名 |
| `artifacts/` | 持久化实验记录、权重、轨迹、源代码归档 |
| `runtime/` | 设备、时钟、执行循环和耗时测量 |
| `evaluation/` | 任务评测与基准配置消费；持久化已移回 artifacts |
| `composition.py` | Hydra 配置装配、兼容性验证和对象构造 |

`robot/` 原本只有缓存，已移除。`execution/reference.py` 已合并到 `execution/commands.py`，保留其有效期和 yaw 检查。`models/gradients.py` 原本只有 LOTF 一条自定义导数，已合并到 `models/lotf.py`；没有额外梯度框架。

## 前向与反向

`dynamics@env.execution.dynamics` 选择前向模型；`algorithm.gradient.transition` 选择导数规则。`direct` 对选中的前向函数自动求导；`lotf_analytical` 保留真实前向值，使用上游简化模型的解析导数；点质量的 `exponential` 规则仍由点质量实现拥有。模型参数、积分、网络权重结构不因文件移动而改变。

## Scene 与资产

Scene 在本项目描述几何、障碍和运动。`assets/scenes/` 只存需要外部文件的数据：空世界和随机图元不需要占位文件，竞速资产保留在原始上游目录以保留相对路径和许可。详见 `assets/scenes/README.md`。
Isaac Lab 的 InteractiveScene 更宽，包含机器人、传感器等仿真实体；这里沿用可组合的语义，没有声称复制其完整类层级。

## 依据与范围

- Hydra experiment configuration: https://hydra.cc/docs/patterns/configuring_experiments/
- Isaac Lab scene: https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.scene.html
- Google Python naming/import style: https://google.github.io/styleguide/pyguide.html
- MuJoCo Python formatting (pyink/isort): https://github.com/google-deepmind/mujoco/blob/main/STYLEGUIDE.md

具体目录布局是本项目按实际职责作出的组织选择，不是上述项目强制的一套层级。上游源码、容器、已保存实验与权重没有被重写。本轮只做语法、导入、配置和构造检查，不跑完整回归或训练。
