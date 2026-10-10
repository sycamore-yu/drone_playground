# Learning update and recovery interface

`Trainer` performs one real parameter update through the shared Simulation environment.
The caller controls the training loop, evaluations, stopping decisions and GPU assignment.
There is no total interaction cap and an update does not indicate convergence.

```python
from drone_playground.learning.trainer import Trainer
from drone_playground.simulation.environment import TrackingEnv

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

## 可训练 Method 与 Brax 互通

### Results v3

新 run 按 [ADR-0014](adr/0014-results-layout-orbax-tensorboard.md) 保存：
`config.yaml`、TensorBoard 事件、`checkpoints/model_NNNNNN/`（Orbax
`training/` + 选模通过才有的 `actor/`）、正式冻结 `episodes.csv`
及按需生成的 `replays/best/`。唯一额外索引 `run.json` 保存完成性、来源和
所选模型路径。每个模型目录的 `metadata.json` 保存其训练状态来源。
旧 v2 ZIP 训练归档仍可以按原合同恢复；新 v3 推理加载直接指定
`checkpoint=results/<run_id>/checkpoints/model_NNNNNN`。

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

New mixed training follows [ADR-0013](adr/0013-three-stage-learning-acceptance.md):
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
