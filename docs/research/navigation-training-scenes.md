# 导航训练与评测场景

当前第二阶段冻结源码`927f8a7`使用同一套Navigation8几何训练、开发选模和正式评测。训练随机抽取场景、出发／近终点／途中状态、动态障碍相位、2–4m/s目标速度和25–50ms延迟。开发使用50000–50063共64个初态／扰动种子；首个独立确认使用60000–60199共200个种子。几何相同、回合不同，结果只解释为固定场景下的初态／扰动鲁棒性，不能声称未见地图泛化。

用户允许单独创建训练场景后，新增可选的`training.scene`。它只供可微训练的rollout使用，`env.scene`继续提供开发和评测几何。默认null保留既有配方与旧检查点；`training=navigation_independent`启用独立的有限训练场景库。每次运行同时记录`training-scene-manifest.json`与原`scene-manifest.json`及两个内容摘要。显式独立训练库若与评测库相同会拒绝启动。

独立库复用项目已有的SANDO样式解析几何生成器，不读取Navigation8。默认几何种子81000／81001，静态圆柱林／混合障碍、动态林／横穿障碍各自覆盖三个难度、每难度4个实例，合计24个固定实例。课程为20×10×5m、起终点距离15m；难度和运动遵循原生成器语义，不宣称与100m Navigation8难度等价。生成器验证初始水平网格连通与端点安全，不能据此保证所有动态时刻均可达，也未覆盖Navigation8全部3D扩展拓扑。策略输入、净空目标、传感和碰撞仍共用同一训练几何事实，没有隐式参考路径。

示例配置解析：

```bash
JAX_PLATFORMS=cuda pixi run train method=learning/depth_navigation \
  training=navigation_independent runtime.device=gpu \
  training.num_envs=32 training.policy_updates=1000 \
  algorithm.horizon_length=32 training.max_wall_seconds=3600 \
  training.num_evals=9 run_id=<新运行身份>
```

点云可将method替换为`learning/pointcloud_navigation`。几何种子和参数初始化种子分别归`training.scene.seed`和`training.seed`所有。同一个冻结训练库可用于多个参数种子；若研究几何采样影响，应单列更换几何种子。评测仍会在Navigation8开发集上选模，因此后续即使启用独立训练库，正式Navigation8结果也不能称为从未用于模型选择的新几何测试。

入口已通过真实JAX采样和隔离检查，尚未以此训练新的质量候选。现有冻结批次继续使用原几何；启用新库需另立配方，不能改写旧结果或额外消耗已用完的候选额度。

GPU训练已启用：公开配置为`runtime.device=gpu`，JAX环境变量使用`JAX_PLATFORMS=cuda`；当前确认启动器为`cuda,cpu`，实际运行通过`jax.devices("gpu")`显式选择显卡，CPU同时用于归档等宿主操作。`jax=gpu`不是本项目的配置项。导航感知／循环网络／可微动力学在GPU上执行，MuJoCo回放构建、压缩和摘要检查仍有CPU成本。
