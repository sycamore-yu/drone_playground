# 已执行命令与冻结权重

命令由原运行command.txt生成，工作目录为当前main工作树。再次执行训练／评测时给run_id追加独立后缀，原目录保留。冻结权重路径与摘要见evidence.json。
环境准备：pixi install --locked；原生规划器：bash native_planners/setup.sh；完整MPC回归：bash scripts/tools/setup_acados.sh或设置ACADOS_SOURCE_DIR。

## PPO · 悬停

运行：`final-acceptance-heldout-ppo-hovering-v1`。范围：原控制任务和25–50毫秒随机命令延迟。

训练：

```bash
pixi run train method=learning/ppo env=hovering run_id=final-acceptance-ppo-hovering-t0 runtime.device=gpu training.num_envs=512 training.num_timesteps=2097152 training.num_evals=5 training.development_episodes=16 training.publish_live=false training.max_wall_seconds=1800 algorithm.batch_size=64 algorithm.num_minibatches=8 algorithm.num_updates_per_batch=8 network.distribution_type=normal
```

已执行冻结评测：

```bash
pixi run eval checkpoint=experiments/final-acceptance-ppo-hovering-t0/checkpoints/step-0001048576.pkl runtime.device=gpu evaluation.role=heldout evaluation.episodes=32 run_id=final-acceptance-heldout-ppo-hovering-v1
```

已有物理轨迹回放：

```bash
pixi run play replay=experiments/final-acceptance-heldout-ppo-hovering-v1/rollouts visualization=headless
```

## PPO · 跟踪

运行：`final-acceptance-heldout-ppo-tracking-v1`。范围：原控制任务和25–50毫秒随机命令延迟。

训练：

```bash
pixi run train method=learning/ppo env=tracking run_id=final-acceptance-ppo-tracking-t0 runtime.device=gpu training.num_envs=512 training.num_timesteps=2097152 training.num_evals=5 training.development_episodes=16 training.publish_live=false training.max_wall_seconds=1800 algorithm.batch_size=64 algorithm.num_minibatches=8 algorithm.num_updates_per_batch=8 network.distribution_type=normal training.warm_start=experiments/final-acceptance-legacy/ppo_tracking/step-0001310720.pkl
```

已执行冻结评测：

```bash
pixi run eval checkpoint=experiments/final-acceptance-ppo-tracking-t0/checkpoints/step-0001572864.pkl runtime.device=gpu evaluation.role=heldout evaluation.episodes=32 run_id=final-acceptance-heldout-ppo-tracking-v1
```

已有物理轨迹回放：

```bash
pixi run play replay=experiments/final-acceptance-heldout-ppo-tracking-v1/rollouts visualization=headless
```

## PPO · 竞速

运行：`final-acceptance-heldout-ppo-racing-v1`。范围：原控制任务和25–50毫秒随机命令延迟。

训练：

```bash
pixi run train method=learning/ppo env=racing run_id=final-acceptance-ppo-racing-t0 runtime.device=gpu training.num_envs=512 training.num_timesteps=2097152 training.num_evals=5 training.development_episodes=16 training.publish_live=false training.max_wall_seconds=1800 algorithm.batch_size=64 algorithm.num_minibatches=8 algorithm.num_updates_per_batch=8 network.distribution_type=normal training.warm_start=experiments/final-acceptance-legacy/ppo_racing/step-0004194304.pkl algorithm.discounting=0.94 algorithm.gae_lambda=0.97 algorithm.clipping_epsilon=0.26 algorithm.max_grad_norm=1.5
```

已执行冻结评测：

```bash
pixi run eval checkpoint=experiments/final-acceptance-ppo-racing-t0/checkpoints/step-0002097152.pkl runtime.device=gpu evaluation.role=heldout evaluation.episodes=32 run_id=final-acceptance-heldout-ppo-racing-v1
```

已有物理轨迹回放：

```bash
pixi run play replay=experiments/final-acceptance-heldout-ppo-racing-v1/rollouts visualization=headless
```

## PPO · 静态导航

运行：`final-acceptance-heldout-ppo-navigation-static-v2`。范围：4096交互的导航工程验收。

训练：

```bash
pixi run train method=learning/ppo env=navigation/static run_id=final-acceptance-ppo-navigation-static-smoke-v2 runtime.device=cpu training.num_envs=8 training.num_evals=2 training.development_episodes=1 training.max_wall_seconds=1800 training.publish_live=false env.task.duration=300 training.num_timesteps=4096 algorithm.batch_size=8 algorithm.num_minibatches=2 algorithm.num_updates_per_batch=2 algorithm.unroll_length=8
```

已执行冻结评测：

```bash
pixi run eval checkpoint=experiments/final-acceptance-ppo-navigation-static-smoke-v2/checkpoints/step-0000004096.pkl run_id=final-acceptance-heldout-ppo-navigation-static-v2 runtime.device=cpu evaluation=navigation_v2 evaluation.role=heldout evaluation.episodes=2
```

已有物理轨迹回放：

```bash
pixi run play replay=experiments/final-acceptance-heldout-ppo-navigation-static-v2/rollouts visualization=headless
```

## PPO · 动态导航

运行：`final-acceptance-heldout-ppo-navigation-dynamic-v2`。范围：4096交互的导航工程验收。

训练：

```bash
pixi run train method=learning/ppo env=navigation/dynamic run_id=final-acceptance-ppo-navigation-dynamic-smoke-v2 runtime.device=cpu training.num_envs=8 training.num_evals=2 training.development_episodes=1 training.max_wall_seconds=1800 training.publish_live=false env.task.duration=300 training.num_timesteps=4096 algorithm.batch_size=8 algorithm.num_minibatches=2 algorithm.num_updates_per_batch=2 algorithm.unroll_length=8
```

已执行冻结评测：

```bash
pixi run eval checkpoint=experiments/final-acceptance-ppo-navigation-dynamic-smoke-v2/checkpoints/step-0000004096.pkl run_id=final-acceptance-heldout-ppo-navigation-dynamic-v2 runtime.device=cpu evaluation=navigation_v2 evaluation.role=heldout evaluation.episodes=2
```

已有物理轨迹回放：

```bash
pixi run play replay=experiments/final-acceptance-heldout-ppo-navigation-dynamic-v2/rollouts visualization=headless
```

## BPTT · 悬停

运行：`final-acceptance-heldout-bptt-hovering-v1`。范围：原控制任务和25–50毫秒随机命令延迟。

训练：

```bash
pixi run train method=learning/bptt env=hovering run_id=final-acceptance-bptt-hovering-t0 runtime.device=gpu training.num_evals=5 training.development_episodes=16 training.publish_live=false training.max_wall_seconds=1800 training.num_envs=16 training.policy_updates=1024 algorithm.horizon_length=40
```

已执行冻结评测：

```bash
pixi run eval checkpoint=experiments/final-acceptance-bptt-hovering-t0/checkpoints/step-0000655360.pkl runtime.device=gpu evaluation.role=heldout evaluation.episodes=32 run_id=final-acceptance-heldout-bptt-hovering-v1
```

已有物理轨迹回放：

```bash
pixi run play replay=experiments/final-acceptance-heldout-bptt-hovering-v1/rollouts visualization=headless
```

## BPTT · 跟踪

运行：`final-acceptance-heldout-bptt-tracking-v1`。范围：原控制任务和25–50毫秒随机命令延迟。

训练：

```bash
pixi run train method=learning/bptt env=tracking run_id=final-acceptance-bptt-tracking-t0 runtime.device=gpu training.num_evals=5 training.development_episodes=16 training.publish_live=false training.max_wall_seconds=1800 training.num_envs=16 training.policy_updates=1024 algorithm.horizon_length=40 training.warm_start=experiments/final-acceptance-legacy/apg_tracking/step-0000655360.pkl
```

已执行冻结评测：

```bash
pixi run eval checkpoint=experiments/final-acceptance-bptt-tracking-t0/checkpoints/step-0000655360.pkl runtime.device=gpu evaluation.role=heldout evaluation.episodes=32 run_id=final-acceptance-heldout-bptt-tracking-v1
```

已有物理轨迹回放：

```bash
pixi run play replay=experiments/final-acceptance-heldout-bptt-tracking-v1/rollouts visualization=headless
```

## BPTT · 竞速

运行：`final-acceptance-heldout-bptt-racing-v1`。范围：原控制任务和25–50毫秒随机命令延迟。

训练：

```bash
pixi run train method=learning/bptt env=racing run_id=final-acceptance-bptt-racing-t0 runtime.device=gpu training.num_evals=5 training.development_episodes=16 training.publish_live=false training.max_wall_seconds=1800 training.num_envs=16 training.policy_updates=1024 algorithm.horizon_length=40 training.warm_start=experiments/final-acceptance-legacy/apg_racing/step-0000655360.pkl
```

已执行冻结评测：

```bash
pixi run eval checkpoint=experiments/final-acceptance-bptt-racing-t0/checkpoints/step-0000491520.pkl runtime.device=gpu evaluation.role=heldout evaluation.episodes=32 run_id=final-acceptance-heldout-bptt-racing-v1
```

已有物理轨迹回放：

```bash
pixi run play replay=experiments/final-acceptance-heldout-bptt-racing-v1/rollouts visualization=headless
```

## BPTT · 静态导航

运行：`final-acceptance-heldout-bptt-navigation-static-v2`。范围：4096交互的导航工程验收。

训练：

```bash
pixi run train method=learning/bptt env=navigation/static run_id=final-acceptance-bptt-navigation-static-smoke-v3 runtime.device=cpu training.num_envs=8 training.num_evals=2 training.development_episodes=1 training.max_wall_seconds=1800 training.publish_live=false env.task.duration=300 training.policy_updates=32 algorithm.horizon_length=16
```

已执行冻结评测：

```bash
pixi run eval checkpoint=experiments/final-acceptance-bptt-navigation-static-smoke-v3/checkpoints/step-0000004096.pkl run_id=final-acceptance-heldout-bptt-navigation-static-v2 runtime.device=cpu evaluation=navigation_v2 evaluation.role=heldout evaluation.episodes=2
```

已有物理轨迹回放：

```bash
pixi run play replay=experiments/final-acceptance-heldout-bptt-navigation-static-v2/rollouts visualization=headless
```

## BPTT · 动态导航

运行：`final-acceptance-heldout-bptt-navigation-dynamic-v2`。范围：4096交互的导航工程验收。

训练：

```bash
pixi run train method=learning/bptt env=navigation/dynamic run_id=final-acceptance-bptt-navigation-dynamic-smoke-v3 runtime.device=cpu training.num_envs=8 training.num_evals=2 training.development_episodes=1 training.max_wall_seconds=1800 training.publish_live=false env.task.duration=300 training.policy_updates=32 algorithm.horizon_length=16
```

已执行冻结评测：

```bash
pixi run eval checkpoint=experiments/final-acceptance-bptt-navigation-dynamic-smoke-v3/checkpoints/step-0000004096.pkl run_id=final-acceptance-heldout-bptt-navigation-dynamic-v2 runtime.device=cpu evaluation=navigation_v2 evaluation.role=heldout evaluation.episodes=2
```

已有物理轨迹回放：

```bash
pixi run play replay=experiments/final-acceptance-heldout-bptt-navigation-dynamic-v2/rollouts visualization=headless
```

## SHAC · 悬停

运行：`final-acceptance-heldout-shac-hovering-v1`。范围：原控制任务和25–50毫秒随机命令延迟。

训练：

```bash
pixi run train method=learning/shac env=hovering run_id=final-acceptance-shac-hovering-t0 runtime.device=gpu training.num_evals=5 training.development_episodes=16 training.publish_live=false training.max_wall_seconds=1800 training.num_envs=64 training.policy_updates=320
```

已执行冻结评测：

```bash
pixi run eval checkpoint=experiments/final-acceptance-shac-hovering-t0/checkpoints/step-0000655360.pkl runtime.device=gpu evaluation.role=heldout evaluation.episodes=32 run_id=final-acceptance-heldout-shac-hovering-v1
```

已有物理轨迹回放：

```bash
pixi run play replay=experiments/final-acceptance-heldout-shac-hovering-v1/rollouts visualization=headless
```

## SHAC · 跟踪

运行：`final-acceptance-heldout-shac-tracking-v1`。范围：原控制任务和25–50毫秒随机命令延迟。

训练：

```bash
pixi run train method=learning/shac env=tracking run_id=final-acceptance-shac-tracking-t0 runtime.device=gpu training.num_evals=5 training.development_episodes=16 training.publish_live=false training.max_wall_seconds=1800 training.num_envs=64 training.policy_updates=320
```

已执行冻结评测：

```bash
pixi run eval checkpoint=experiments/final-acceptance-shac-tracking-t0/checkpoints/step-0000655360.pkl runtime.device=gpu evaluation.role=heldout evaluation.episodes=32 run_id=final-acceptance-heldout-shac-tracking-v1
```

已有物理轨迹回放：

```bash
pixi run play replay=experiments/final-acceptance-heldout-shac-tracking-v1/rollouts visualization=headless
```

## SHAC · 竞速

运行：`final-acceptance-heldout-shac-racing-trained-v1`。范围：原控制任务和25–50毫秒随机命令延迟。

训练：

```bash
pixi run train method=learning/shac env=racing run_id=final-acceptance-shac-racing-t1 runtime.device=gpu training.num_envs=64 training.policy_updates=320 training.num_evals=5 training.development_episodes=16 training.publish_live=false training.max_wall_seconds=1800 'network.hidden_sizes=[32,32]' algorithm.learning_rate=0.0001 training.warm_start=experiments/final-acceptance-bptt-racing-t0/checkpoints/step-0000491520.pkl
```

已执行冻结评测：

```bash
pixi run eval checkpoint=experiments/final-acceptance-shac-racing-t1/checkpoints/step-0000655360.pkl runtime.device=gpu evaluation.role=heldout evaluation.episodes=32 run_id=final-acceptance-heldout-shac-racing-trained-v1
```

已有物理轨迹回放：

```bash
pixi run play replay=experiments/final-acceptance-heldout-shac-racing-trained-v1/rollouts visualization=headless
```

## SHAC · 静态导航

运行：`final-acceptance-heldout-shac-navigation-static-v2`。范围：4096交互的导航工程验收。

训练：

```bash
pixi run train method=learning/shac env=navigation/static run_id=final-acceptance-shac-navigation-static-smoke-v3 runtime.device=cpu training.num_envs=8 training.num_evals=2 training.development_episodes=1 training.max_wall_seconds=1800 training.publish_live=false env.task.duration=300 training.policy_updates=32 algorithm.horizon_length=16
```

已执行冻结评测：

```bash
pixi run eval checkpoint=experiments/final-acceptance-shac-navigation-static-smoke-v3/checkpoints/step-0000004096.pkl run_id=final-acceptance-heldout-shac-navigation-static-v2 runtime.device=cpu evaluation=navigation_v2 evaluation.role=heldout evaluation.episodes=2
```

已有物理轨迹回放：

```bash
pixi run play replay=experiments/final-acceptance-heldout-shac-navigation-static-v2/rollouts visualization=headless
```

## SHAC · 动态导航

运行：`final-acceptance-heldout-shac-navigation-dynamic-v2`。范围：4096交互的导航工程验收。

训练：

```bash
pixi run train method=learning/shac env=navigation/dynamic run_id=final-acceptance-shac-navigation-dynamic-smoke-v3 runtime.device=cpu training.num_envs=8 training.num_evals=2 training.development_episodes=1 training.max_wall_seconds=1800 training.publish_live=false env.task.duration=300 training.policy_updates=32 algorithm.horizon_length=16
```

已执行冻结评测：

```bash
pixi run eval checkpoint=experiments/final-acceptance-shac-navigation-dynamic-smoke-v3/checkpoints/step-0000004096.pkl run_id=final-acceptance-heldout-shac-navigation-dynamic-v2 runtime.device=cpu evaluation=navigation_v2 evaluation.role=heldout evaluation.episodes=2
```

已有物理轨迹回放：

```bash
pixi run play replay=experiments/final-acceptance-heldout-shac-navigation-dynamic-v2/rollouts visualization=headless
```

## 点云论文方法 · 悬停

运行：`final-acceptance-heldout-pointcloud-hovering-v1`。范围：完整点云编码器与显式0.02输入尺度的控制迁移。

训练：

```bash
pixi run train method=paper/pointcloud_flight env=paper/control/hovering run_id=final-acceptance-pointcloud-hovering-t2 network=paper_pointnet_gru_conditioned runtime.device=gpu training.num_envs=8 training.policy_updates=1024 training.num_evals=5 training.development_episodes=16 training.max_wall_seconds=1800 training.publish_live=false algorithm.horizon_length=32 +algorithm.max_grad_norm=1.0
```

已执行冻结评测：

```bash
pixi run eval checkpoint=experiments/final-acceptance-pointcloud-hovering-t2/training-state/update-0001024.pkl run_id=final-acceptance-heldout-pointcloud-hovering-v1 runtime.device=gpu evaluation.role=heldout evaluation.episodes=32
```

已有物理轨迹回放：

```bash
pixi run play replay=experiments/final-acceptance-heldout-pointcloud-hovering-v1/rollouts visualization=headless
```

## 点云论文方法 · 跟踪

运行：`final-acceptance-heldout-pointcloud-tracking-v1`。范围：完整点云编码器与显式0.02输入尺度的控制迁移。

训练：

```bash
pixi run train method=paper/pointcloud_flight env=paper/control/tracking run_id=final-acceptance-pointcloud-tracking-t1 network=paper_pointnet_gru_conditioned runtime.device=gpu training.num_envs=8 training.policy_updates=1024 training.num_evals=5 training.development_episodes=16 training.max_wall_seconds=1800 training.publish_live=false algorithm.horizon_length=32 +algorithm.max_grad_norm=1.0
```

已执行冻结评测：

```bash
pixi run eval checkpoint=experiments/final-acceptance-pointcloud-tracking-t1/training-state/update-0001024.pkl run_id=final-acceptance-heldout-pointcloud-tracking-v1 runtime.device=gpu evaluation.role=heldout evaluation.episodes=32
```

已有物理轨迹回放：

```bash
pixi run play replay=experiments/final-acceptance-heldout-pointcloud-tracking-v1/rollouts visualization=headless
```

## 点云论文方法 · 竞速

运行：`final-acceptance-heldout-pointcloud-racing-v1`。范围：完整点云编码器与显式0.02输入尺度的控制迁移。

训练：

```bash
pixi run train method=paper/pointcloud_flight env=paper/control/racing run_id=final-acceptance-pointcloud-racing-t1 network=paper_pointnet_gru_conditioned runtime.device=gpu training.num_envs=8 training.policy_updates=1024 training.num_evals=5 training.development_episodes=16 training.max_wall_seconds=1800 training.publish_live=false algorithm.horizon_length=32 +algorithm.max_grad_norm=1.0
```

已执行冻结评测：

```bash
pixi run eval checkpoint=experiments/final-acceptance-pointcloud-racing-t1/training-state/update-0001024.pkl run_id=final-acceptance-heldout-pointcloud-racing-v1 runtime.device=gpu evaluation.role=heldout evaluation.episodes=32
```

已有物理轨迹回放：

```bash
pixi run play replay=experiments/final-acceptance-heldout-pointcloud-racing-v1/rollouts visualization=headless
```

## 点云论文方法 · 静态导航

运行：`final-acceptance-pointcloud-navigation-30000-v2`。范围：原论文30000更新冻结权重的新导航协议。

已执行冻结评测：

```bash
pixi run eval method=paper/pointcloud_flight env=paper/pointcloud_navigation_v2 checkpoint=experiments/final-acceptance-artifacts/pointcloud-paper-update-0030000/update-0030000.pkl runtime.device=gpu run_id=final-acceptance-pointcloud-navigation-30000-v2
```

已有物理轨迹回放：

```bash
pixi run play replay=experiments/final-acceptance-pointcloud-navigation-30000-v2/rollouts visualization=headless
```

## 点云论文方法 · 动态导航

运行：`final-acceptance-pointcloud-navigation-30000-v2`。范围：原论文30000更新冻结权重的新导航协议。

已执行冻结评测：

```bash
pixi run eval method=paper/pointcloud_flight env=paper/pointcloud_navigation_v2 checkpoint=experiments/final-acceptance-artifacts/pointcloud-paper-update-0030000/update-0030000.pkl runtime.device=gpu run_id=final-acceptance-pointcloud-navigation-30000-v2
```

已有物理轨迹回放：

```bash
pixi run play replay=experiments/final-acceptance-pointcloud-navigation-30000-v2/rollouts visualization=headless
```

## SUPER · 悬停

运行：`final-acceptance-native-super-hovering-v2`。范围：原生规划器接收滚动目标；竞速起飞接管单独计数。

已执行冻结评测：

```bash
pixi run eval method=paper/super env=hovering observation@env.observation=state_reference runtime.device=cpu evaluation.episodes=2 method.port=55751 run_id=final-acceptance-native-super-hovering-v2
```

已有物理轨迹回放：

```bash
pixi run play replay=experiments/final-acceptance-native-super-hovering-v2/rollouts visualization=headless
```

## SUPER · 跟踪

运行：`final-acceptance-native-super-tracking-v2`。范围：原生规划器接收滚动目标；竞速起飞接管单独计数。

已执行冻结评测：

```bash
pixi run eval method=paper/super env=tracking observation@env.observation=state_reference runtime.device=cpu evaluation.episodes=2 method.port=55751 run_id=final-acceptance-native-super-tracking-v2
```

已有物理轨迹回放：

```bash
pixi run play replay=experiments/final-acceptance-native-super-tracking-v2/rollouts visualization=headless
```

## SUPER · 竞速

运行：`final-acceptance-native-super-racing-v3`。范围：原生规划器接收滚动目标；竞速起飞接管单独计数。

已执行冻结评测：

```bash
pixi run eval method=paper/super env=racing observation@env.observation=state_reference runtime.device=cpu evaluation.episodes=2 method.port=55753 run_id=final-acceptance-native-super-racing-v3
```

已有物理轨迹回放：

```bash
pixi run play replay=experiments/final-acceptance-native-super-racing-v3/rollouts visualization=headless
```

## SUPER · 静态导航

运行：`final-acceptance-native-super-static-v2`。范围：原生规划器导航第二版协议。

已执行冻结评测：

```bash
pixi run eval method=paper/super env=navigation/static evaluation=navigation_v2 evaluation.episodes=2 method.workers=2 method.port=55601 runtime.device=cpu run_id=final-acceptance-native-super-static-v2
```

已有物理轨迹回放：

```bash
pixi run play replay=experiments/final-acceptance-native-super-static-v2/rollouts visualization=headless
```

## SUPER · 动态导航

运行：`final-acceptance-native-super-dynamic-v2`。范围：原生规划器导航第二版协议。

已执行冻结评测：

```bash
pixi run eval method=paper/super env=navigation/dynamic evaluation=navigation_v2 evaluation.episodes=2 method.workers=2 method.port=55606 runtime.device=cpu run_id=final-acceptance-native-super-dynamic-v2
```

已有物理轨迹回放：

```bash
pixi run play replay=experiments/final-acceptance-native-super-dynamic-v2/rollouts visualization=headless
```

## EGO-Planner · 悬停

运行：`final-acceptance-native-ego_planner-hovering-v2`。范围：原生规划器接收滚动目标；竞速起飞接管单独计数。

已执行冻结评测：

```bash
pixi run eval method=paper/ego_planner env=hovering observation@env.observation=state_reference runtime.device=cpu evaluation.episodes=2 method.port=55751 run_id=final-acceptance-native-ego_planner-hovering-v2
```

已有物理轨迹回放：

```bash
pixi run play replay=experiments/final-acceptance-native-ego_planner-hovering-v2/rollouts visualization=headless
```

## EGO-Planner · 跟踪

运行：`final-acceptance-native-ego_planner-tracking-v2`。范围：原生规划器接收滚动目标；竞速起飞接管单独计数。

已执行冻结评测：

```bash
pixi run eval method=paper/ego_planner env=tracking observation@env.observation=state_reference runtime.device=cpu evaluation.episodes=2 method.port=55751 run_id=final-acceptance-native-ego_planner-tracking-v2
```

已有物理轨迹回放：

```bash
pixi run play replay=experiments/final-acceptance-native-ego_planner-tracking-v2/rollouts visualization=headless
```

## EGO-Planner · 竞速

运行：`final-acceptance-native-ego_planner-racing-v3`。范围：原生规划器接收滚动目标；竞速起飞接管单独计数。

已执行冻结评测：

```bash
pixi run eval method=paper/ego_planner env=racing observation@env.observation=state_reference runtime.device=cpu evaluation.episodes=2 method.port=55751 run_id=final-acceptance-native-ego_planner-racing-v3
```

已有物理轨迹回放：

```bash
pixi run play replay=experiments/final-acceptance-native-ego_planner-racing-v3/rollouts visualization=headless
```

## EGO-Planner · 静态导航

运行：`final-acceptance-native-ego_planner-static-v2`。范围：原生规划器导航第二版协议。

已执行冻结评测：

```bash
pixi run eval method=paper/ego_planner env=navigation/static evaluation=navigation_v2 evaluation.episodes=2 method.workers=2 method.port=55621 runtime.device=cpu run_id=final-acceptance-native-ego_planner-static-v2
```

已有物理轨迹回放：

```bash
pixi run play replay=experiments/final-acceptance-native-ego_planner-static-v2/rollouts visualization=headless
```

## EGO-Planner · 动态导航

运行：`final-acceptance-native-ego_planner-dynamic-v2`。范围：原生规划器导航第二版协议。

已执行冻结评测：

```bash
pixi run eval method=paper/ego_planner env=navigation/dynamic evaluation=navigation_v2 evaluation.episodes=2 method.workers=2 method.port=55626 runtime.device=cpu run_id=final-acceptance-native-ego_planner-dynamic-v2
```

已有物理轨迹回放：

```bash
pixi run play replay=experiments/final-acceptance-native-ego_planner-dynamic-v2/rollouts visualization=headless
```
