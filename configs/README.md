# 配置

当前唯一实验配方为Hydra YAML：`config.yaml`组合policy/controller/dynamics/scene/observation/task，以及algorithm/network/objective/training。`experiment/`保存已经对照和实际运行的完整预设。

```bash
pixi run experiment --cfg job experiment=lotf_hybrid_hover
pixi run train experiment=lotf_hybrid_tracking run_id=my-lotf-tracking
pixi run train experiment=figure8_ppo dynamics.forward=so_rpy_rotor run_id=my-ppo
```

普通覆盖检查已知字段；跨多个算法预设的批量覆盖中，某字段仅存在于部分算法时使用Hydra的`++`显式新增/覆盖，例如`++algorithm.horizon_length=8`。
PPO预设声明`training.num_timesteps`；APG/SHAC/LOTF由更新次数×环境数×展开步数得到交互数，
其`num_timesteps`默认null。显式填写时必须与实际展开预算相等，矛盾配置在启动前拒绝。
LOTF原生CSV跟踪固定50Hz；保持原采样语义后，才能比较同一参考上的训练结果。

原P1–P4 JSON已迁移，历史运行内的JSON和检查点保留。依赖由Pixi固定；新机器先`git submodule update --init --recursive`再`pixi install`。
