# Training

`pixi run train` runs one declared recipe through Autoresearch Gym. The native
Learning runner collects a rollout, updates parameters, logs measurements, and
saves numbered Orbax checkpoints. Optional Best selection measures completed
Episode Reward during training. Autoresearch evaluates the selected frozen Actor
in a separate Benchmark process after training. Learning does not invoke Benchmark.

```bash
pixi run train experiment=tracking learning=bptt seed=0
pixi run train experiment=racing learning=shac seed=1
pixi run train experiment=oa_static_depth learning=ppo seed=2
pixi run train experiment=oa_dynamic_lidar learning=bptt seed=0
```

## Budget

One rollout/update collects:

```text
interactions per update = simulation.num_envs × learning.options.horizon
total interactions = interactions per update × number of updates
```

These correspond to Isaac Lab's environment count, steps per environment, and
training iterations. A step is one policy/environment interaction; internal
physics steps and repeated optimizer epochs do not add interactions.
`learning.max_interactions` sets the total and must be a positive multiple of
the rollout batch. The resolved configuration records all factors. A completed
budget reports `budget_exhausted`; policy quality is measured separately.

```bash
pixi run train experiment=tracking learning=ppo \
  simulation.num_envs=256 learning.options.horizon=64 \
  learning.max_interactions=4194304
```

This example performs 256 updates. Its budget illustrates accounting, not a
convergence guarantee. OA defaults use 16 environments and a 32-step rollout;
Tracking/Racing use 256 environments, with 64 steps for BPTT/PPO and 32 for SHAC.

## State and recovery

Each native training directory contains `config.yaml`, `run.json`, TensorBoard
events, a machine-readable `metrics.jsonl` stream, and `checkpoints/model_XXXXXX/`. A model directory contains metadata,
a committed `actor/` payload, and, while retained, a complete `training/` payload.
Training state includes parameters, optimizer, RNG, recurrent memory, objective
history, unfinished episode returns and lengths, and the batched environment.
Checkpoint metadata also retains the Best score and measurement-window totals. `results.keep_checkpoints` retains the
latest complete training states while preserving exported Actors and evidence.

SIGINT or SIGTERM completes the current update and saves the last finite state.
To resume, repeat the original recipe and budget with its exact model path:

```bash
pixi run train experiment=tracking learning=ppo \
  resume=/absolute/path/to/external/train/checkpoints/model_000100
```

Autoresearch creates a new attempt directory. Resume checks the saved task,
scene, sensor, simulation, method, controller, observation, learning, and seed
contracts. Counter values remain cumulative. Older ZIP formats and missing
configuration defaults are not restored.

## Algorithms and interfaces

`Trainer.initialize()` creates explicit state; `Trainer.update(state)` returns
new state and measurements. BPTT differentiates through the rollout, SHAC adds a
terminal value estimate, and PPO uses sampled actions and clipped policy updates.
Actor input is the method-visible observation. An optional privileged Critic is
training-only. Task owns rewards; Learning combines the task objective with
explicit auxiliary losses.

`method.actor`, `method.action`, and `learning.loss` describe distinct contracts.
Actions determine output width and physical units. `learning.backward_model`
selects native derivatives, `lotf`, or `point_mass_lag` for differentiable methods.
LOTF requires body-rate/thrust actions; point-mass lag requires acceleration.
The Zhang/Liu acceleration losses require acceleration actions, so the builder
uses the Task objective (`learning.loss=null`) for OA with LOTF.
See [control models](control-models.md) for dynamics and delays.

## Optional Best selection

Best selection is disabled by default. Choose one source:

| Override | Selection measurement |
|---|---|
| `learning.best.source=null` | Save Latest; `checkpoint=best` reports its fallback |
| `learning.best.source=evaluation` | Mean return over independent, deterministic complete episodes |
| `learning.best.source=training` | Mean return over live episodes completed since the previous selection |

```bash
pixi run train experiment=tracking learning=ppo \
  learning.best.source=evaluation learning.best.interval=262144 \
  learning.best.episodes=8 learning.best.seed=1000000
```

The interval counts environment transitions, not optimizer calls. Selection runs
at the first completed update crossing an interval, and at the final budget
update. Updates pause while independent evaluation completes. A strictly higher
mean reward publishes `best_model`; equal or worse scores preserve the previous
Best. Training then continues. No automatic patience or reward-threshold stop is
implied; the fixed budget and explicit interrupt determine when training ends.

Evaluation uses the same physical simulation, action interpretation, observation,
and EpisodeObjective as training, with deterministic actions. It neither updates
parameters nor consumes the training RNG, recurrent memory, or environment state.
The evaluation seed also fixes procedural geometry independently of the training
seed. Distance curricula are evaluated at their fixed final difficulty. Use
`evaluation` when training difficulty changes; training-return windows then
measure different distributions and are not comparable quality estimates.

Reward is the undiscounted sum of `-EpisodeObjective.cost` through episode end.
Auxiliary network-prediction losses and SHAC bootstrap values are not Episode
Reward. Partial episode returns carry across updates; only completed episodes
enter training-source means. Monte Carlo tail copies used for value targets do
not count as extra live episodes. Evaluation records preserve per-episode return,
terminal event, seed/world identity, length, and task-specific quality measures.

Orbax payloads and metadata commit before one atomic `checkpoints/latest`
reference publishes them. `best_model` follows that publication's Best reference.
Selectors read committed metadata; `run.json` is a status projection. Retention
may remove old resumable states but keeps their frozen Actors, including Best.
Resume retains the previous Best score and incomplete measurement window.

The independent-evaluation behavior follows the
[Stable-Baselines3 EvalCallback](https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback).
The training-source alternative uses completed episode returns, as in
[skrl 1.4.3 checkpoint selection](https://github.com/Toni-SM/skrl/blob/1.4.3/skrl/agents/torch/base.py#L619-L632).
These are behavior references; neither library is added as a dependency.

## Training distributions and evidence

Static and dynamic OA use separate randomized obstacle distributions. Fixed S/D
maps belong to Benchmark. Select native, LOTF, or point-mass-lag backward dynamics
for BPTT and SHAC; PPO does not differentiate through the environment.

Keep selection, search, and final test cohorts separate. A local experiment
protocol declares coverage, seeds, budget, and quality criteria before execution.
Trial schedules, queue scripts, and convergence-study matrices belong in ignored
`tmp/` and `results/` directories, outside the public source tree. Preserve source
and configuration fingerprints, every attempt, and paired outcomes for previously
successful episodes. Budget completion and rising reward do not prove convergence.


## Release training defaults

The Command Builder selects the task's default training recipe. Passing state
recipes retain their recorded settings. Racing PPO uses the `reference_options`
optimizer settings and starts from random weights. OA PPO uses finite-horizon
`goal_reaching` returns with a 512-step rollout. Parameter choices and current
convergence evidence are separate; a candidate is not labelled passed merely
because its configuration composes.

The OA defaults use `simulation.procedural.layout=matched` and the existing
`distance_curriculum` sampler. The optional check stays inside Learning:

```yaml
learning:
  best:
    source: evaluation
    metric: success_rate
    episodes: 24
    confirmation_episodes: 96
  early_stopping:
    metric: success_rate
    target: 1.0
    patience: 0
```

A 24-case success triggers the larger check. A configured success target applies
only to its confirmed result. Set `learning.best.source=null` and disable the
early-stopping target to use the ordinary budget-only training loop. The final
Benchmark is still a separate command. See [ADR-0034](adr/0034-release-trainable-defaults.md).
