# Learning update and recovery interface

`Trainer` performs one real parameter update through the shared Simulation environment.
The caller controls the training loop, evaluations, stopping decisions and GPU assignment.
There is no total interaction cap and an update does not indicate convergence.

```python
from drone_playground.learning.trainer import Trainer
from drone_playground.simulation.environment import TrackingEnv, make_environment

env = TrackingEnv(num_envs=32, device="gpu")
trainer = Trainer(env, kind="state", algorithm="ppo", seed=0,
                  config={"horizon": 32, "lr": 3e-4, "minibatches": 4})
state = trainer.initialize()
state, metrics = trainer.update(state)
trainer.save_state("results/run/checkpoints/latest.training.zip", state,
                   config={"simulation": resolved_environment_config},
                   provenance={"git_revision": revision, "run_id": "run"})
state, metadata = trainer.load_state("results/run/checkpoints/latest.training.zip")
trainer.save_inference("results/run/checkpoints/step-000001/policy.zip", state,
                       config={"simulation": resolved_environment_config},
                       provenance={"git_revision": revision, "run_id": "run"})
```

For a Navigation depth recipe, construct the sensor with explicit image settings:

```python
env = make_environment(
    task="navigation", scene="S01", sensor="depth", action={"level": "acceleration"},
    sensor_config={"width": 64, "height": 48, "frequency_hz": 30,
                   "pitch_deg": 20, "min_range": 0.3, "max_range": 10},
)
trainer = Trainer(env, kind="depth", loss="zhang")
```

LiDAR recipes select `sensor="lidar"`, `kind="lidar"` and `loss="liu"`. Device
constructors no longer infer a training image mode. Sensor/Experiment YAML holds these
settings. Both devices produce snapshots; checkpoint identity records acquisition
semantics, the MID360 resource hash, point budget, preprocessing and latency bounds.
Full training archives require the current state structure and matching identity.

## 可训练 Method 与 Brax 互通

### Results v3

新的长时间训练按照 [ADR-0018](adr/0018-training-patience-and-frozen-selection.md)
启用训练诊断早停：`early_stopping.min_updates=800`，
`patience=6`，`min_delta=0.02`。Patience 从第 800 次更新后的
轻量筛查累计，最差场景优先。若首次通过完整候选选模，则只进行一次
冻结 Benchmark；失败后新开随机初始化试验，不重复冻结挑选同一 run
的其他 checkpoint。旧 v2 已有单次达标候选只允许使用
`tools/freeze_legacy_candidate.py --run <旧 run 目录>` 独立审查一次。

新 run 按 [ADR-0016](adr/0016-results-layout-orbax-tensorboard.md) 保存：
`config.yaml`、TensorBoard 事件、`checkpoints/model_NNNNNN/`（Orbax
`training/` + 选模通过才有的 `actor/`）、正式冻结 `episodes.csv`
及按需生成的 `replays/best/`。唯一额外索引 `run.json` 保存完成性、来源和
所选模型路径。每个模型目录的 `metadata.json` 保存其训练状态来源。
旧 v2 ZIP 训练归档仅保留供历史追溯，不由调度器自动续训；新 v3 推理加载直接指定
`checkpoint=results/<run_id>/checkpoints/model_NNNNNN`。

## Navigation 程序化训练 / 固定 Benchmark

根据 [ADR-0019](adr/0019-procedural-training-fixed-navigation-benchmark.md)，
程序化训练使用 `scene=train/procedural`，每个并行世界独立持有
`EnvState.scene_state`：障碍位置、形状、掩码、动态运动目标、
速度/相位，以及无人机起终点。每次 reset 使用 JAX PRNG 采样；
碰撞、Depth、LiDAR、privileged Critic 和训练损失使用同一几何。
训练地图保留位于 y=-17.5 的可行外侧通路作为构造性可达性证明。

新训练入口是 `navigation_procedural_depth` /
`navigation_procedural_lidar`（PPO 可使用各自 `_ppo` 后缀）。
对应 YAML 在 `learning.scenes: null`，不再每 20 个 updates
轮换 S01–D06；原 S01–D06 冻结 MJCF、种子基段与正式门槛不变。
S06/D06 只作扩展报告，不作为已见几何的泛化证明。

`simulation.procedural.start_sampling=distance_mixture` 使用固定的20/50/80米
起终距离混合；`distance_curriculum` 根据已完成交互数占总预算的比例改变
三个距离的采样权重，并保留旧距离。PPO、APG/BPTT、SHAC 均通过同一个
环境 reset 接收该进度，只影响下一次重置的世界。直接调用课程训练时须向
`Trainer.update(state, sampling_progress=...)` 传入 `[0,1]` 进度；CLI 使用保存的
交互计数和 `learning.max_interactions` 计算，恢复时禁止改变课程总预算。
评测不传训练进度，始终使用固定混合分布；每次更新记录实际重置距离计数。

原 12/18 已通过矩阵继续保存在 `results/scratch_v02/`；
新协议必须使用独立的 `results/procedural_v01/`：

```bash
python tools/train_scratch_matrix.py plan --regime procedural \
  --only depth_ppo,lidar_ppo,racing_ppo,racing_shac \
  --results-root results/procedural_v01
python tools/train_scratch_matrix.py run --regime procedural \
  --only depth_ppo,lidar_ppo,racing_ppo,racing_shac \
  --results-root results/procedural_v01 --max-parallel 1
```

训练前须验证 NVIDIA 驱动与用户态库版本一致。
CPU 端的程序化 Scene/JAX 烟雾测试不能视为 GPU 模型收敛。
该轮仅包含四个遗留难题，共 6 个待解决的正式单元，
不重跑已经通过验收的 12 个单元。独立监控器就是本地 Python
`train_scratch_matrix.py`，每 20 秒更新 `matrix.json`
中的 `observed_at_utc` 和活跃进程；无需 ChatGPT 定时调用。
Racing/SHAC 的新配方 `racing_shac_stable` 将非零 bootstrap
权重降低至 0.2，其他正式实验的物理任务、门数和冻结验收不变。

PPO 诊断候选可选 `learning.options.ppo_target_kl`（默认 0 不启用）
和 `learning.options.ppo_value_clip`（默认 0 不启用）；
每次更新记录 explained variance、Clip Fraction、KL、
Critic target std、动作饱和度与探索方差。
`racing_shac_bootstrap_ablation` 只用于 SHAC 梯度消融：
`bootstrap_weight=0` 不能计入正式 SHAC 的 3-seed 验收。
新 Racing 配方 `racing_shac_stable` 仍保留非零终端价值权重
（`bootstrap_weight=0.2`），按独立随机种子训练与正式验收；
`racing_ppo_stable` 另以 KL 限制、价值裁剪和过门进度奖励
进行受控重训。二者不得覆盖旧 `scratch_v02` 结果。

### PPO 诊断与旧版本控制接口的受控对照

[ADR-0014](adr/0014-rl-environment-and-method-interface.md) 已实现
`BraxTrainingEnv` 标准交互合同；Brax 原生 PPO/APG 属于 **可选对照**，
不是当前带 CNN/PointNet+GRU 感知策略的无缝替换。
[ADR-0020](adr/0020-ppo-diagnostics-before-brax-replacement.md)
记录旧仓库 Brax PPO、DiffAero、VisFly 与新自研 PPO 的差异和验收。

`navigation_procedural_depth_ppo_velocity` 和
`navigation_procedural_lidar_ppo_velocity` 是从零训练的独立配方：
四维世界系速度指令与 Yaw 经速度内环生成 Crazyflow 的
`attitude_thrust`，`reward_profile=legacy_motion_navigation_v2` 单独计算
旧版启发的速度跟踪、净空、进度和终端奖励；不复用旧 checkpoint。
`start_sampling=collision_free` 按 episode 采样不同的起点和起终距离，
并做实际障碍净空检查，独立可达性验证仍需后续补充。

当当前基线配方第 3 次尝试仍失败时，调度器把第 4 次尝试指向以上
速度接口对照。已运行的配方绝不热修改优化器或动作定义。
PPO 训练记录额外提供 `navigation_progress_metres`、
`navigation_collision_fraction`、`navigation_out_of_bounds_fraction`、
`advantage_progress_correlation`、`advantage_positive_fraction`、
`critic_return_rmse_preupdate` 和传感器有效测量比例。
以上均是训练轨迹诊断指标，不能替代独立冻结成功率。

### Task 奖励与 Learning 损失的归属

Navigation 的可选 `task.options.reward_profile=goal_reaching` 采用有限时域的
安全到达目标，要求 `task.options.finite_horizon=true`、`learning.options.gamma=1`
和 `learning.loss=null`，不叠加旧奖励 shaping。每步距离进度为
`(distance_before-distance_after)/20`，首次成功奖励 +1，首次失败奖励 -1。
时间、障碍净空、边界与动作代价的整回合总和上界为 0.5；同一起点下，
0.5 m 目标球内的成功回报比任一失败至少高 1.475。这是任务目标的性质，
不能替代训练收敛证据，也不声称保留旧奖励的最优策略。

`finite_horizon=true` 时，任务截止时间产生 `terminated=true`，不再 bootstrap；
Actor 的状态输入末尾加入归一化剩余时间。有限 rollout 的非终止末端仍正常
bootstrap。该区分遵循 [Gymnasium 的时间限制语义](https://gymnasium.farama.org/v0.26.3/tutorials/handling_time_limits/)。
此选项目前用于 Navigation；默认 false 保留已有模型的观测与时间截断合同。
它作为 Task 配置和 EnvState 静态元数据恢复，不增加 checkpoint 数组字段。

根据 [ADR-0001](adr/0001-simulation-learning-boundary.md)、
[ADR-0014](adr/0014-rl-environment-and-method-interface.md) 与
[ADR-0021](adr/0021-task-owned-reward-contract.md)，Task 参数归
Simulation 管理，任务奖励数学在 simulation/task_rewards.py。
NavigationEnv 通过 cost() 和 task_reward_components() 对所有方法
暴露相同的 Task 奖励；learning/objective.py 仅负责组合任务成本
与 Zhang/Liu 等网络专有的感知、预测和 jerk 学习损失。

**新配方**选择 task.options.reward_profile 与可选的
task.options.reward_shaping；参考
navigation_procedural_{depth,lidar}_ppo_velocity_task。
**已有 Trial-4** 的原始速度 PPO YAML 和 checkpoint 必须保持不变，
其历史 learning.options.reward_profile 仅在构造 Environment 时
转换为 Task 的奖励类型。原 Learning 训练参数里的进度／失败／高度
权重暂时保留为历史配置来源，但计算全部在 Simulation 执行。

训练算法与具体网络、Action 和执行 Method 分离。构造阶段由 Hydra 实例化
`method.actor`，Actor 的输出宽度从选定 Action 取得。可选择独立的 Critic；
`PolicyMethod` 默认使用连续 tanh-Gaussian PPO 分布，而 PPO 更新调用
`method.sample / log_probability / entropy`，APG/SHAC 通过
`method.deterministic / execute` 使用同一个前向 Method。自定义可训练方法应
在这些操作中定义自身的动作和概率／梯度合同，而不修改 PPO/APG/SHAC 代码。

`Trainer` 管理单次参数更新、优化器及可恢复训练状态。`EpisodeObjective`
独立计算各算法共用的实际任务代价、具名感知代价、因果历史和训练辅助项。
基础任务奖励仍来自 `Environment.cost`；后续若采用具体环境子类，再将
任务专属奖励项移动到相应任务实现，避免为同一种损失保留两份公式。

可选 `learning.brax_adapter.BraxTrainingEnv` 提供 Brax 标准
`Env.reset / step`、`State.pipeline_state/obs/reward/done/metrics/info`
和规格属性。其 `pipeline_state` 直接持有现有 EnvState，不复制 Crazyflow
的动力学、传感、任务和延迟状态。当前只验证了 State／无传感器的 PPO
兼容路径；Depth/LiDAR、循环网络和原生 Brax APG 仍需分别评估。

当环境已由 `Sim(n_worlds=num_envs)` 批量运行时，Brax PPO 对照使用
`wrap_env=False`，由适配器执行完成世界的新随机初态重置、回合
统计与终止前观测保留，禁止再套一层上游 `VmapWrapper`。

可在安装了 Brax 0.14.2 的隔离环境运行短程验证：

```bash
SCIPY_ARRAY_API=1 JAX_PLATFORMS=cpu PYTHONPATH=src \
  python tools/smoke_brax.py --algorithm ppo --num-envs 4 --timesteps 64
```

此入口直接调用上游 `brax.training.agents.ppo.train`，只用于接口验收和后续受控对照，
不会替换现有 PPO 或创建第二套项目级 Runner/Trainer。此次短程训练完成了
原生 PPO 更新调用，但没有验证收敛；完整算法对照需分别冻结配方、预算、随机种子和模型。
也可以选择 `--algorithm apg`，调用 Brax 原生 APG 进行同样的短程接口检查。

For Navigation acceleration recipes, select
`NavigationEnv(scene="S01", sensor="depth", action={"level": "acceleration"})`
and `Trainer(env, kind="depth", loss="zhang")`, or select LiDAR and `loss="liu"`.
The network, physical action interface and loss are independent choices. Environment owns the
batch, device, action conversion, sensor clock and observed fields. Trainer uses
`env.observe` for every actor kind.
The shared Actor receives the same tensor inputs for every algorithm. Its raw output
is squashed by `tanh` before `Environment.step`. The independent Critic receives only
`observation['state']` by default. Depth/LiDAR PPO can opt into
`critic_uses_sensor=true`, which constructs a separate small sensor encoder
for the value function using only measurements already visible to the Actor.
Alternatively, `critic_uses_privileged=true` selects AsymmetricPPO's independent
value MLP. This Navigation-only PPO option is mutually exclusive with
`critic_uses_sensor`. Its input includes true position, goal displacement,
quaternion, velocities, previous action, time, signed clearance, and 26 ideal
world-frame geometry ranges capped at 40 m. Fixed normalization scales are
position `[100,20,6]` m, displacement `[100,40,6]` m, linear velocity 3 m/s,
angular velocity 10 rad/s, task duration, and geometry 40 m. These inputs bypass
sensor noise, delay and field of view and are explicitly simulation privileges.
The Actor ignores `privileged_state`; its architecture and frozen input contract
remain the same. Only Actor variables are exported for inference. See
[ADR-0010](adr/0010-training-only-privileged-ppo-critic.md).

Independent Navigation PPO runs can set `++observation.navigation_world_rotation=true`
to append the row-major body-to-world rotation matrix to `observation['state']`.
The original 10 features become 19 (21 with the separate goal-observation option).
Actions remain world-frame accelerations. This changes the checkpoint input contract;
start a separately named run from random initialization. Frozen evaluation rejects
switching this flag even when `benchmark.allow_environment_change=true`.
See [ADR-0023](adr/0023-navigation-world-rotation-observation.md); this option is
implemented on the paper experiment branch, not yet integrated into `v0.2`.

`TrainingState` is a Flax dataclass containing full actor variables (`params`, including
its `params` collection), critic and target critic variables, both Optax states, RNG,
complete environment PyTree, recurrent memory, perception objective history, trainable
PPO `log_std`, update count and interaction count. New environment sensor leaves are
preserved automatically. Interaction means one world advancing one control interval;
physics substeps are not counted as separate training interactions. Host integer
counters avoid the default JAX int32 overflow without enabling global float64.

Each update is compiled, with batched environment operations and `lax.scan` over time
and optimizer epochs. Metrics are device scalar arrays; callers explicitly synchronize
when measuring wall time or exporting numeric logs. Count metrics are host integers.
Metrics include the actual optimized loss, actor gradient norm before clipping, mean
reward, task/perception costs, done fraction, updates and interactions. PPO and SHAC
also report their independently optimized critic loss and gradient norm.

## Algorithms and episode semantics

PPO samples a Gaussian in raw action space and stores the pre-tanh sample and its
Jacobian-corrected log density. Rollouts are detached from physics. GAE uses
`delta = reward + gamma*(1-terminated)*V(final_observation) - V(observation)`;
its recursive trace is multiplied by `1-done`. Thus timeouts bootstrap from the
last observation before reset and never propagate advantages into another episode.
The rollout boundary also bootstraps. Advantages use population **standard deviation**,
not variance. The actor objective uses the minimum of ordinary and clipped probability
ratios; the critic independently minimizes half mean squared return error.

Each PPO minibatch contains complete world sequences, not shuffled individual steps.
At every optimizer pass, the current actor recomputes all recurrent states from the
rollout's detached initial memory, zeroing rows immediately after either kind of done.
Sequence indices are shuffled each epoch with the checkpointed RNG. The carried memory
at the next rollout is the behavior policy's final memory; gradients never cross update
boundaries. Entropy, when weighted, is a reparameterized Monte Carlo estimate of the
**transformed** tanh distribution, including its Jacobian. `log_std` is optimized with
the actor and checkpointed.

APG computes the mean shared cost by reverse differentiation through Crazyflow.
Each whole rollout step is rematerialized with `jax.checkpoint`. The optional temporal
decay transforms only the incoming physics PyTree, after the actor has computed the
action: its state Jacobian is multiplied by `exp(-alpha*dt)` while the direct action
derivative stays intact. Simulation stops sensor acquisition gradients; actor network
parameters still receive gradients through the acquired tensor inputs. APG uses no
critic bootstrap or score-function estimator.

SHAC uses the same rematerialized rollout and sums discounted costs within each episode
segment. At truncation and the rollout boundary it subtracts the nonterminated target
critic value times the next discount. Target parameters stay fixed during actor
optimization, while the value's terminal-state Jacobian is retained. True termination
suppresses bootstrap. The discount restarts after a reset. Actor loss is normalized by
horizon times batch. Detached GAE returns train the critic, followed by
`target = target_alpha*old_target + (1-target_alpha)*new_critic`.

All algorithms reset only finished worlds and clear their recurrent/objective histories.
Reset acquisition is skipped entirely when no world has finished.
They preserve final observations before reset for rewards and bootstraps. APG/SHAC retain
within-rollout derivatives across unfinished worlds, and cut the graph at update boundaries.

## Navigation training sampling

The `navigation_mixed_depth` and `navigation_mixed_lidar` experiments address the limited
coverage of the S01-only nominal-start baseline. Its full checkpoint evaluation at update
100 (`results/apg_depth_diagnostic_s0/checkpoint_eval/update-00000100/report.json`) reported
0/25 successes in each of the eight scenes. These recipes change training sampling and
rollout settings; improved convergence still requires measured checkpoint evaluations.

`learning.options.randomize_navigation_start=true` samples Navigation positions uniformly
inside `[2,-18,1]` to `[94,18,5.5]` m, rejecting and resampling positions until body clearance
is at least 0.15 m at episode time zero. Velocity remains independently uniform within
±0.1 m/s per axis and yaw within ±5°. This applies to initialization and finished-world
resets for PPO, APG and SHAC. Other tasks reject the option. The underlying public API is
`Environment.reset(key, state=None, mask=None, *, randomize_position=False)`; the new keyword
is static under JAX compilation. Its default preserves the original nominal start and
jitter, including the original key-to-sample mapping. Masked resets preserve all inactive
world physics, task, action and sensor history.

Optional `learning.scenes` is an ordered, nonempty list of distinct Navigation8 scene IDs.
The CLI trains round-robin in blocks of `learning.scene_updates` updates (default 20 when
a list is supplied), constructing and caching each scene's Environment/Trainer on first
use. Omitting the list retains single-scene training. Both mixed recipes use
`[S01,S02,S03,D01,D02,D03,S06,D06]`, horizon 32, actor learning rate 0.0003, method rate
10 Hz and temporal gradient alpha 0.916290731874155; depth uses batch 128 and LiDAR batch 32.
The sampling settings and Actor architecture are shared across all three algorithms.

At each block boundary, before the next update, `Trainer.reset_episodes(state)` deliberately
truncates the whole training batch into the next scene. It zeros GRU memory and objective
histories, acquires fresh observations, and consumes one split of the carried RNG. Actor,
critic, target critic, optimizer states, log standard deviation and counters are retained;
parameters are never reinitialized at a switch. The `training_scene_switch` event records
both `reset_worlds` and unfinished `truncated_worlds`. This sampling truncation does not
alter Task events, rewards, physics, goal or collision rules.

Run headers and checkpoint `config.training_sampling` record the ordered scene list,
block length and every participating scene's geometry SHA-256. Checkpoint
`provenance.training_sampling` stores the active `scene_index` and `updates_in_scene`.
At a completed block, the saved scene still owns the batch and progress equals the block
length; resuming performs the pending switch before the next update. A checkpoint saved
after a switch but before a successful update has progress zero in the new scene.
Mid-block resumes restore the active scene's template and continue its existing episodes.
Schedule, geometry and progress validation prevent silently continuing a different run.
The current archive format is **v2 only**. Training resume requires matching complete
state, environment, action, network and learning configuration. No missing options are
filled from historical defaults. Standalone inference archives must declare the Actor
specification; old v1 files are historical evidence, not loadable inputs.

New mixed training follows [ADR-0015](adr/0015-three-stage-learning-acceptance.md):
checkpoint screening uses six main scenes and eight episodes per scene (48
rollouts), with permissive 6/8 per-scene candidate eligibility. A qualified
checkpoint receives one full validation of 25 episodes in each main scene
(150 rollouts) using disjoint checkpoint-validation seeds. On the first
full C4 pass, the selected actor is frozen for an independent benchmark in
all eight scenes (200 rollouts, S06/D06 reported but not graded).
State-only Tracking/Racing uses 8/100/100 screening/validation/benchmark
episodes and retains its original C2/C3 quality thresholds. Old in-flight
three-consecutive-pass runs retain their recorded selection protocol.
The CLI's `initial_checkpoint` remains optional (default null): supplying a frozen policy
initializes validated Actor variables with fresh optimizer/episode state and records source
checksum/provenance. `resume` instead restores full training state and sampling progress.

If a saved update was due for evaluation when the process stopped, resuming completes
that evaluation before another update or scene switch. The evaluation seed index advances
only after the complete evaluation returns. Existing partial output is retained under
`checkpoints/step-<update>/incomplete-*`; a `checkpoint_evaluation_retry` event records its
path. Partial output does not count as completed screening or validation;
for old runs it does not count toward their original three-consecutive rule.

```bash
pixi run train experiment=navigation_mixed_depth learning=apg seed=0
pixi run train experiment=navigation_mixed_lidar learning=shac seed=0
```

There is no hard training budget in these recipes. CPU sampling/resume tests do not
establish convergence or the 18-cell acceptance matrix; GPU training and the three-seed
checkpoint/frozen evaluations remain separate measured work.

## Common objectives and named method adaptations

State recipes use `task_weight * env.cost(before, after, action)` for all algorithms.
Perception recipes additionally use `perception_weight * dt * named_cost` from
`learning.losses.zhang_loss` (depth) or `liu_loss` (LiDAR). PPO receives the negative
of this same cost as reward; SHAC bootstraps rewards in the same units. The task term
includes Simulation's dense task cost and first-event success/failure terms. Both
weights default to one, so the named costs supplement the existing Navigation objective.
Set either weight to zero for an explicitly declared ablation.

Navigation can add `altitude_weight * dt * (z-goal_z)^2` inside the task term.
Its default is zero; the altitude diagnostic explicitly uses one. The option is shared
by all three algorithms and rejected for state-only tasks. The Actor still receives
the configured observation, without privileged height or geometry fields added by this loss.

Navigation may also add `progress_reward_scale * (distance_before-distance_after)`
as a per-transition **reward** (subtracted from the task cost), gated by the prior
episode's active state. This follows the goal-progress term used in the previous
Navigation PPO recipe. It defaults to zero and does not change the terminal events,
physics or evaluation. The previous PPO also used a velocity-tracking controller,
a different action contract, and `reward_scale=0.01`; its raw reward coefficients
cannot be substituted into the present acceleration-action objective.

Two independent, opt-in diagnostics isolate the new options from existing runs:

```bash
pixi run drone-playground mode=train experiment=navigation_depth_scratch_ppo_stable \
  learning=ppo seed=0 '++learning.options.progress_reward_scale=0.5' \
  output=results/diagnostic_depth_ppo_progress_s0
pixi run drone-playground mode=train experiment=navigation_depth_scratch_ppo_stable \
  learning=ppo seed=0 '++learning.options.critic_uses_sensor=true' \
  output=results/diagnostic_depth_ppo_sensor_critic_s0
```

The current 18-cell acceptance matrix retains its existing recipes and checkpoints.
These diagnostics use fresh initialization and must be evaluated by the same C5/C6
protocol before considering a recipe replacement. The sensor-aware critic costs
additional memory and compute; it is restricted to PPO because SHAC differentiates
through the state-value bootstrap.

This is a Crazyflow method adaptation with a causal per-step reward, not a claim to
reproduce the original point-mass experiment. All spatial objective quantities use
world coordinates and SI units. Velocity is a trailing causal mean with shorter episode
prefixes, carried across updates; the goal velocity follows Simulation's 3 m/s goal
approach rule. Commanded net acceleration is `6*tanh(raw)` m/s², matching Environment.
Geometric clearance is the nearest scene-surface distance minus body radius, independent
of actor observation permissions. Approach speed is the finite difference of that
clearance over the actual control interval, including obstacle motion. The named helper
detaches the approach weight. Empty-scene distances are masked before arithmetic.

Both named recipes use the current command's squared acceleration norm.
Jerk uses consecutive acceleration commands and is zero at the first episode step;
its history survives update boundaries. Liu's variance term is the online population
variance of jerk magnitudes within the current episode, using Welford moments. This
causal prefix variance is an explicit adaptation of the source's whole-trajectory
variance; it avoids future-dependent rewards and mixing episodes. Its weight can be
set to zero for an instantaneous-jerk comparison. Identical temporal adaptations and
weights apply to PPO, APG and SHAC. Reset clears velocity, acceleration and jerk moments.

Depth's optional velocity auxiliary head is trained directly against detached current
body-frame measured velocity, with component MSE and `velocity_aux_weight`. It is a
separate supervised actor term for **all three algorithms**, never an environmental
reward for PPO. Default weight is zero. LiDAR and state actors reject a nonzero auxiliary
weight because they have no learned velocity head.

## Resolved options

Unknown options and invalid ranges fail when constructing Trainer. Batch and physical
rates are configured on Environment. `resolved_config` records all defaults, named loss
weights, kind, algorithm, seed, batch and dt; caller-supplied Simulation config and source
revision belong in checkpoint metadata.

| Option | Default | Meaning |
| --- | ---: | --- |
| `horizon` | 32 | Control intervals per update |
| `lr`, `critic_lr` | 0.0003, 0.001 | Actor and critic learning rates |
| `gamma`, `gae_lambda` | 0.99, 0.95 | Discount and return trace |
| `ppo_epochs`, `minibatches` | 4, 1 | PPO passes; world partitions, must divide batch |
| `critic_epochs` | 4 | SHAC critic passes; PPO fits critic once per actor minibatch |
| `clip_epsilon` | 0.2 | PPO probability ratio clipping |
| `entropy_weight` | 0 | PPO transformed-distribution entropy bonus |
| `max_grad_norm` | 1 | Separate actor and critic gradient clipping |
| `target_alpha` | 0.995 | SHAC old-target coefficient |
| `temporal_gradient_alpha` | 0 | APG/SHAC incoming physics decay, in 1/s |
| `initial_log_std` | -0.5 | PPO diagonal Gaussian initialization |
| `weight_decay` | 0 | Actor AdamW decay; critic uses Adam |
| `task_weight`, `perception_weight` | 1, 1 | Common cost weights |
| `altitude_weight` | 0 | Navigation goal-height squared error inside the task cost |
| `progress_reward_scale` | 0 | Navigation progress reward per metre advanced toward the goal |
| `critic_uses_sensor` | false | Independent Depth/LiDAR PPO sensor encoder for the Critic |
| `critic_uses_privileged` | false | Navigation AsymmetricPPO with training-only true state and geometry |
| `velocity_aux_weight` | 0 | Depth auxiliary component MSE |
| `randomize_navigation_start` | false | Wide collision-checked Navigation training starts |
| `perception_loss` | Named defaults | Keyword overrides for the selected named helper |

Named defaults are resolved from the helper's signature and copied into saved config.
For depth these include velocity1, clearance1.5, collision2, acceleration0.01,
jerk0.001, window30, Huber delta1, margin1, beta32 and approach floor1.
LiDAR includes velocity1, collision1.5, acceleration0.01, jerk0.001, velocity norm/component
weights0.8/0.6, beta1=4/3, beta2=32, jerk mean/variance weights1/0.1 and approach floor0.
`velocity_window` is handled by the trainer's saved history; the primitive is evaluated
with window1 after smoothing. Set the auxiliary weight at the top level.

## Checkpoint contract and validation scope

`checkpoint.save_state(path, state, config=..., provenance=...)` atomically replaces a
single archive after flushing/fsync. It contains versioned JSON metadata and Flax
MessagePack for the **entire** TrainingState, with a SHA-256 payload checksum.
`checkpoint.load_state(path, template)` returns `(state, metadata)` and checks purpose,
version, checksum, structure, leaf shapes and dtypes. `Trainer.load_state` additionally
compares the resolved learning recipe. The caller must recreate the same Simulation
configuration, including task, scene, dynamics, clocks and sensor calibration. Static
Simulation objects are reconstructed from that configuration rather than pickled.

`save_inference(path, params, kind=..., config=..., provenance=...)` writes a separate
inference archive. `load_inference(path)` returns `params`, `kind`, config, provenance
and format metadata. The policy variables work directly with Simulation's `Actor`;
no trainer is needed to execute it. The frozen runner initializes/reset memory and
uses `tanh(actor.apply(params, observation, memory)[0])` as its bounded action.

CPU regression tests exercise real official Crazyflow physics for all three updates,
termination/truncation arithmetic, recurrent recomputation/reset, SHAC terminal-value
state gradients, incoming-physics decay versus action gradients, exact checkpoint
continuation, inference equality, and interrupted-save preservation. These tests do not
establish L3 convergence, L4 GPU throughput, or the 18-cell benchmark acceptance matrix;
those remain the caller's independent training/evaluation work.

## Procedural PPO: controlled start-distribution comparison

Training uses `train/procedural`: obstacle geometry and movement are independently
sampled by world and episode. Fixed S01–D03 scenes appear only in checkpoint
validation and the final benchmark. Historical checkpoint screens varied their
initialization seed with `evaluation_index`, so scores from different updates were
measured on different initial-state samples.

The new opt-in `benchmark.fixed_checkpoint_seeds=true` compares every checkpoint
on the same eight initial states per scene. The screening and full-validation
seed bases remain separate, and final benchmark seeds retain their independent
partition. `evaluation_index` continues to identify each report. New runs may
also set `results.keep_best_screening=true` to retain the best incomplete
screening checkpoint in addition to the most recent resumable states. Selection
still requires a passing full validation; a good screening score alone is not
accepted as a model or frozen benchmark result.
Training can set `benchmark.frozen_seed_base=3000000` to reserve a new final-test
cohort without changing checkpoint-validation seeds. It must be an integer of
at least 2,000,000; null preserves the historical default. If a frozen-test failure
informs another recipe, reserve a new cohort before that next final test.
With Orbax, ordinary periodic checkpoints save resumable state without exporting
an inference Actor. This lets retention prune unselected checkpoints while
keeping the best screening checkpoint and any accepted Actor.

The old `start_sampling=collision_free` mixture places 35% of starts near the
goal and draws the rest uniformly along the course. It covers nominal
benchmark starts near x=2 m infrequently. The `stratified` preset divides
worlds into near-start, uniform, and near-goal cohorts and rotates cohort
assignments each episode. Collision checking, independent obstacle layouts,
sensor models, dynamics and goal conditions remain unchanged.

Paired, independently initialized configurations:

| Sensor | Existing start distribution | Balanced start distribution |
| --- | --- | --- |
| Depth | `navigation_procedural_depth_ppo_validation` | `navigation_procedural_depth_ppo_stratified` |
| LiDAR | `navigation_procedural_lidar_ppo_validation` | `navigation_procedural_lidar_ppo_stratified` |

Run each pair with the same seed and training budget, in separate output
directories; compare fixed-checkpoint screening trajectories and then a held-out
full validation. Training metrics now include completed-episode counts, success
rate, collision rate and out-of-bounds rate. They are distinct from per-step
event frequencies and from checkpoint-validation success rates. Prior runs and
their checkpoints retain the original seed and start-sampling contracts.

The procedural scratch scheduler reserves PPO candidates 9 and 10 for this
comparison. Candidate 9 copies the best recorded non-velocity trial's explicit
optimization settings; candidate 10 reuses candidate 9's settings and changes
only `simulation.procedural.start_sampling`. Both initialize new Actor/Critic
weights. Historical `best_screen_score` in `run.json` remains usable for this
choice after rolling checkpoint cleanup.
