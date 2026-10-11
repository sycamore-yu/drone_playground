# Fan LiDAR / VisFly Depth PPO transfer (11 Oct 2026)

## Status and evidence

- Source branch: `feat/ppo-reward-ablation-v01` at `c60ab63`. It is preserved.
- Independent development: `feat/paper-ppo-fan-visfly-v01`; old Trials 9/10 and reward A/B/C/D results are never resumed or overwritten.
- Hydra smoke composition: `navigation_paper_fan2025_ppo` and `navigation_paper_visfly2025_ppo` both returned code 0.
- Static Ruff and compilation validated. End-to-end PPO training and independent six-scene benchmarks are **not yet accepted**.
- Warning: on 11 Oct, shared host showed >70% 10-second I/O PSI `full`, >30 GiB swap, and another account's eight D-state AirVLN workers. Do **not** start simultaneous full PPO jobs until the host is healthy. Do not terminate other users' processes.

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
2. Run one complete **actual** `Trainer.update` through sensor → policy → Crazyflow → reward → GAE → critic/actor gradients at reduced batch/horizon. Log finite gradients, initial observation shapes and wall time.
3. Only once a pilot succeeds and the shared host has reasonable I/O PSI, train one method at a time, with cgroup memory/CPU limits, no JAX default GPU preallocation and separate `results/paper_ppo_v01/{fan_lidar_s0,visfly_depth_s0}` identities.
4. Keep the same independent six fixed S01–S03/D01–D03 scene validation and frozen benchmark as the prior program. Replicate successful variants on seeds 1 and 2. Never claim success from random training returns alone.

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
