# Fan LiDAR / VisFly Depth PPO transfer (11 Oct 2026)

## Status and evidence

- Source branch: `feat/ppo-reward-ablation-v01` at `c60ab63`. It is preserved.
- Independent development: `feat/paper-ppo-fan-visfly-v01`; old Trials 9/10 and reward A/B/C/D results are never resumed or overwritten.
- Hydra smoke composition: `navigation_paper_fan2025_ppo` and `navigation_paper_visfly2025_ppo` both returned code 0.
- Static Ruff and compilation validated. Two real full-size PPO seed-0 transfer pilots have started; independent six-scene benchmarks are **not yet accepted**.
- Host precaution: earlier on 11 Oct the shared host showed >70% 10-second I/O PSI `full`, >30 GiB swap and another account's D-state AirVLN workers. After the host recovered, two GPU pilots were admitted with per-job CPU/memory/swap limits and no JAX GPU preallocation. Do not terminate other users' processes.

## Reproduced architecture versus transfer assumptions

**Fan et al., *Flying in Highly Dynamic Environments with End-to-end Learning Approach* (RA-L, 2025):** The policy receives a 36-azimuth by 36-history radial-distance tensor, planar velocity/prior-action descriptors (4D) and planar relative goal (2D). The Flax residual CNN and two-dimensional horizontal acceleration head are an **explicit architectural reconstruction**; no complete original parameter release was verified. This transfer retains the independent Crazyflow Mellinger tracker and introduces an explicit PD altitude-hold term (gain 2.0 / derivative 1.5; ±3 m/s²). This height controller is our implementation choice, not a reported original component.

**VisFly `cluttered_flight` reference:** The local upstream checkout is
`/home/tong/tongworkspace/reference_repos/VisFly` at `6ff29bf`. Its experiment configs live under
`exps/examples/{alg_cfgs,env_cfgs}/cluttered_flight`. The original representation comprises 64×64 metric depth, 13D world pose/orientation/velocity/angular rate, and 3D absolute goal. Default image channels are 6/12/18 with 5×5/3×3/3×3 kernels, depth projection 128, state/target MLPs 128→64, separate 64→64 actor/value branches and four-dimensional body-rate/thrust actions. This repository uses a native Flax port, not direct PyTorch checkpoint conversion or the upstream SB3 implementation.

Both current experiments are **transfer-to-Drone-Playground** studies, not a replication of original Unity / VisFly scene geometry, camera pipeline, runtime, or original success-rate tables. Existing JAX PPO uses its own rollout/GAE/clipping. Training uses 500Hz physics, 10Hz policy and long procedural courses, unlike the VisFly official 0.03-second timestep, 256-step PPO rollout and 10M-step budget. Results must be labeled as transfers.

## Isolated 80m training distribution

`SceneDistribution.min_start_goal_distance_m: 80.0` is an opt-in protocol for the two paper configs. It restricts `collision_free` candidates to start_x in [2,16] and retains collision-free selection. With a goal_x at least 96.8, horizontal separation is at least 80.8m even before accounting for y/z. The historical v1 generator SHA and identity are preserved exactly when the new option is disabled, including omission of the new field from the hashed JSON.

## Validation gates

1. Run `ruff check` and focused `tests/test_paper_ppo_replica.py`.
2. Run complete **actual** `Trainer.update` through sensor → policy → Crazyflow → reward → GAE → critic/actor gradients. User selected direct full-size PPO training rather than a separate small smoke test.
3. When host health permits, use cgroup CPU/memory/zero-swap limits and separate `results/paper_ppo_v01/{fan_lidar_s0,visfly_depth_s0}` identities. Record JAX compilation time separately. Simultaneous processes are admitted only when GPU/host monitoring supports them; don't confuse utilization with training convergence.
4. Keep the same independent six fixed S01–S03/D01–D03 scene validation and frozen benchmark as the prior program. Replicate successful variants on seeds 1 and 2. Never claim success from random training returns alone.

## Full-size pilot evidence and resolved fixed-scene bug

The first Fan training run reached **160 PPO updates (81,920 environment
interactions)** but its initial six-scene evaluation crashed because a fixed
Navigation scene supplies an unbatched `[3]` goal whereas the experimental
Actor expected `[B,3]`. Commit `b8bc05d` fixed both Fan/VisFly goal
observations and the Fan altitude-hold controller's target broadcasting.
Two focused regression tests passed. Fan training then resumed from its
existing Orbax `model_000160` (parameters, RNG and optimizer preserved);
no new random initialization or replay of the prior 160 updates.

At the complete, fixed-seed diagnostic at **update 160**, six scenes × 8
episodes each:

| Pilot | Fixed success | Failures | Diagnostic |
| --- | ---: | --- | --- |
| Fan LiDAR PPO | 0/48 | 48 collisions | Median collision time 5.54s |
| VisFly Depth PPO | 0/48 | 24 collisions, 24 out-of-bounds | Static collision median 1.14s; dynamic out-of-bounds median 4.35s |

VisFly's training `saturated_action_fraction` was approximately **0.79–0.80**
at updates 210–250 and `kl_actor_update_skipped` approximately 0.7 in
several updates. This is strong evidence for an output/observation-scale
diagnostic; it does **not** prove that any proposed replacement will succeed.
Do not change this saved pilot's network, reward or sensor mid-run.

The two live pilots run in isolated user systemd services
`dp-paper-fan-resume-20261011.service` and
`dp-paper-visfly-train-20261011.service`, each CPUQuota=600%,
MemoryHigh=10GiB, MemoryMax=15GiB, MemorySwapMax=0,
`XLA_PYTHON_CLIENT_PREALLOCATE=false`. Simultaneous training has reached
**97–100% GPU utilization**, approximately **5.7GiB** device memory and
negligible host I/O pressure. Shared-GPU PPO updates are around 0.95s
per process per 512-interaction batch, compared with approximately
0.5s when a method runs alone. This is throughput evidence only.

## Faster follow-up evaluations without changing PPO

The original two seed-0 pilots remain frozen with their original complete
six-scene screen every 160 updates. For **new** follow-up runs, select:

- `navigation_paper_fan2025_ppo_staged`
- `navigation_paper_visfly2025_ppo_staged`

Their sole difference from the corresponding base recipe is
`screening.probe_between_full=true` and `screening.full_every=3`:
intermediate checkpoints evaluate two paired static/dynamic scenes
(16 episodes), every third evaluates all six scenes (48 episodes);
any successful probe promotes to a full screen. The PPO network,
reward, dynamics, initialization and decision clock remain **identical**.
The Hydra contract is covered by two tests. A 16-episode probe must never be
reported as a 48-episode or formal 25-per-scene benchmark.

## Next single-variable transfer study: VisFly position scale

The original VisFly reference goal is about 9m in its small indoor scene,
but Drone Playground's long course provides about 98m as an **absolute
goal position**. The initial baseline has roughly 80% tanh action saturation,
with frequent KL update skips and fixed-scene altitude failures.

With frozen synthetic camera/self-state inputs and the same randomly
initialized Flax Actor parameters per seed, replacing **only** world
position and goal by position/10 and goal/10 produced:

| Initialization seed | Direct world-coordinates tanh saturation | Position/goal divided by 10 |
| --- | ---: | ---: |
| 0 | 75% | 0% |
| 1 | 75% | 0% |
| 2 | 100% | 0% |

This numerical test is **not** an evaluation rollout and does not predict
success. It establishes a plausible observation-scale mismatch, independent
of the velocity/angular-rate and depth inputs.

An opt-in `observation.visfly_position_scale` (default 1.0, preserving
existing experiments) and a new
`navigation_paper_visfly2025_ppo_posscale10_staged` config implement
this isolated hypothesis; the only training-input change relative to
`navigation_paper_visfly2025_ppo_staged` is simultaneous division of the
state's position and the absolute target by 10. The CNN, action head,
exploration distribution, reward, optimizer, random-seed contract and
control frequency are all unchanged. Static tensor and Hydra contracts
passed. This is a **future distinct run**, never applied to currently
running `visfly_depth_s0` or its saved checkpoints.

Runtime diagnostic command (read-only):

```bash
cat /proc/pressure/io
free -h
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
```

Smoke test (must use the already-pinned Pixi environment):

```bash
cd /home/tong/tongworkspace/simulation_dev/.worktrees/new_drone_playground/paper-ppo-fan-visfly-v01
PY=/home/tong/tongworkspace/simulation_dev/new_drone_playground/.pixi/envs/default/bin/python
SCIPY_ARRAY_API=1 XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=src \
  "$PY" tmp/paper_ppo_smoke.py --experiment navigation_paper_fan2025_ppo
```
