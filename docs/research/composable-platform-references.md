# 可组合平台的开源依据

核对日期：2026-09-26。本文记录实际读过的官方文档/源码及本项目选择；公开主线可能继续变化。
架构参考与运行依赖分开：本项目继续使用Crazyflow和Brax/JAX，其他平台用于对照设计。

## 训练编排

| 项目 | 核对到的组织方式 | 本项目采用 |
|---|---|---|
| Isaac Lab | 环境与智能体配置分开；环境有观测、动作、奖励、终止、事件等配置；支持管理器与直接实现 | 环境/算法配置分组，内部细节由模块封装 |
| MuJoCo Playground | env_cfg与ppo_params分开，network_factory嵌入学习参数，随机化通过环境注册表取得 | 任务/训练分开，网络可替换，一个训练入口 |
| Brax | train接收environment、network_factory、算法超参数、随机化函数与回调 | 保持小调用接口；优化器、批次等作为算法参数 |
| DiffAero | Hydra默认组algo/env/dynamics/network/sensor/logger | 一次选择实验配方，再覆盖实际需要的组 |
| OmniDrones | task、drone_model、algo及action_transform配置；控制器可作为动作变换 | 按任务、机器人和控制接口组合；仅作架构参考 |

Isaac Lab：
https://isaac-sim.github.io/IsaacLab/main/source/refs/reference_architecture/index.html
https://isaac-sim.github.io/IsaacLab/develop/_modules/isaaclab_tasks/utils/hydra.html

MuJoCo Playground：
https://github.com/google-deepmind/mujoco_playground/blob/4057c147714b6ac09b395377f1a1724bbeacc4d3/learning/train_jax_ppo.py

Brax：
https://github.com/google/brax/blob/main/brax/training/agents/ppo/train.py
读取文件blob：517ca81313a8e6064f3ccb3ed7c0284ebb128f8a。

DiffAero：
https://github.com/flyingbitac/diffaero/blob/291ea14196aefbebcf7387dd71f7e096c83878b7/cfg/config_train.yaml

OmniDrones：
https://github.com/btx0424/OmniDrones/blob/9ce7c2028b71be64d7e748c31f685cd3b54afe27/docs/source/rl.rst
https://github.com/btx0424/OmniDrones/blob/9ce7c2028b71be64d7e748c31f685cd3b54afe27/docs/source/demo/crazyflie.md
当前README披露维护困难；本项目只复用设计经验，继续保留已有运行技术栈。

## 同时比较学习与优化控制、可替换机器人的平台

safe-control-gym的同一环境支持PID、LQR、iLQR、MPC、GP-MPC及PPO/SAC等，
并提供符号预测模型、约束和扰动。它最接近本项目的“学习方法与优化控制共同比较”目标。
https://github.com/learnsyslab/safe-control-gym

robosuite将机器人、控制器、场景对象和任务组合；官方基准显式比较任务×机器人×控制接口。
它最适合参考组合本身如何成为研究变量和结果表。
https://robosuite.ai/
https://robosuite.ai/docs/_sources/algorithms/benchmarking.html

本轮核对范围里，跨平台的控制/任务组合已有直接范例；前向模型与代理反向模型的独立选择
以LOTF源码作为具体依据，避免假定每个模块化平台都提供这一能力。

## LOTF原论文与源码

论文III-C：BPTT策略优化；III-D：残差加速度目标；III-E：前向残差与简化解析反向；
IV-A：训练与在线更新安排；IV-C：模型和梯度设计对比。
https://arxiv.org/html/2508.21065v2
https://rpg.ifi.uzh.ch/lotf/

原论文明确将近期状态/动作写入滚动缓冲，学习实测加速度与解析预测的差，随后更新策略。
这属于完整在线适应方法。它在概念上能够进入额外运行调度，当前模块复现仅取高保真前向、
简化反向、BPTT及相应原任务。

https://github.com/uzh-rpg/learning_on_the_fly/blob/cba6e5370773ace8a08107f02810eecabf16c793/lotf/objects/quadrotor_obj.py
该文件step包含高保真低层循环，custom_jvp中的p/R/v切向量由simplified_dyn计算。
https://github.com/uzh-rpg/learning_on_the_fly/blob/cba6e5370773ace8a08107f02810eecabf16c793/lotf/algos/bptt.py

MPC当前输出约定：
https://github.com/learnsyslab/lsy_drone_racing/blob/b1f5b36adb8e08e8e2adea85de790bd0e0a1d118/lsy_drone_racing/control/attitude_mpc.py

## 配置与文档位置

Hydra对象构造：
https://hydra.cc/docs/advanced/instantiate_objects/overview/

Matt原文使用小接口封装完整行为，领域术语放CONTEXT.md，重要决策放docs/adr，
规格与一个任务一个文件的工作单放.scratch/<feature>。本项目按这些位置维护，
用户要求的模块并行分工覆盖默认的逐阶段交付形式，共同验收仍是完整训练结果。
https://github.com/mattpocock/skills/blob/main/skills/engineering/codebase-design/SKILL.md
https://github.com/mattpocock/skills/blob/main/skills/engineering/domain-modeling/SKILL.md
https://github.com/mattpocock/skills/blob/main/skills/engineering/domain-modeling/ADR-FORMAT.md
https://github.com/mattpocock/skills/blob/main/skills/engineering/setup-matt-pocock-skills/issue-tracker-local.md
