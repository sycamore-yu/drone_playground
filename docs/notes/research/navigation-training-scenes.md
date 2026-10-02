# 导航训练与评测场景

当前历史冻结批次使用同一套 Navigation8 几何训练、checkpoint_eval 选模和正式评测。训练随机抽取场景、出发／近终点／途中状态、动态障碍相位、2–4m/s目标速度和25–50ms延迟。几何相同、运行 role 和回合不同，结果只解释为固定场景下的初态／扰动鲁棒性，不能声称未见地图泛化。

可选的 `training.scene_distribution` 只供训练 rollout 使用；`env.scene` 继续定义 checkpoint_eval 与 benchmark 的名义几何。`experiment=navigation/differentiable_pointcloud` 与 `experiment=papers/depth_diffphysics` 直接在 `training.scene_distribution` 中声明 generated scene distribution。每次运行同时记录训练与评测场景摘要；显式独立训练库若与评测库相同会拒绝启动。

独立库复用项目已有的解析几何生成器，不读取 Navigation8。默认几何种子81000／81001，静态圆柱林／混合障碍、动态林／横穿障碍各自覆盖三个难度、每难度4个实例，合计24个固定实例。它是固定 training distribution，不称为 curriculum。生成器验证初始水平网格连通与端点安全；策略输入、净空目标、传感和碰撞共用同一训练几何事实，没有隐式参考路径。

示例配置解析：

```bash
JAX_PLATFORMS=cuda pixi run train experiment=papers/depth_diffphysics \
  runtime.device=gpu training.num_envs=32 training.policy_updates=1000 \
  algorithm.horizon_length=32 training.max_wall_seconds=3600 \
  training.num_evals=9 run_id=<新运行身份>
```

点云导航使用 `experiment=navigation/differentiable_pointcloud`；recurrent trainer、generated training scene 与 `benchmarks/navigation.yaml` 都由该 recipe 明确记录。几何种子属于 `training.scene_distribution`，参数初始化种子属于 `training.seed`。checkpoint_eval 仍使用 Navigation8，因此即使训练库独立，也不能把最终 Navigation8 结果描述成完全未参与模型选择的新几何测试。

入口已通过真实JAX采样和隔离检查，尚未以此训练新的质量候选。现有冻结批次继续使用原几何；启用新库需另立配方，不能改写旧结果或额外消耗已用完的候选额度。

GPU训练已启用：公开配置为`runtime.device=gpu`，JAX环境变量使用`JAX_PLATFORMS=cuda`；当前确认启动器为`cuda,cpu`，实际运行通过`jax.devices("gpu")`显式选择显卡，CPU同时用于归档等宿主操作。`jax=gpu`不是本项目的配置项。导航感知／循环网络／可微动力学在GPU上执行，MuJoCo回放构建、压缩和摘要检查仍有CPU成本。
