# Navigation Actor input and benchmark audit — 2026-10-11

**Status:** Measured; proposed optimization rejected. No existing training or checkpoint modified.

## Resolved current PPO recipes

The audited recipes are the *stored* A–D runs under
`results/ppo_reward_ablation_v01` (not the newer Sensor P0/P1 worktree).
The two actors are in `simulation/networks.py` and receive an egocentric
10-dimensional state from `NavigationEnv._observe_state`:

```
[
  body_frame_velocity_xyz,                       # 3
  body_frame_goal_velocity_xyz (speed <= 3 m/s), # 3
  body_frame_world_up_xyz,                        # 3
  nominal_body_radius = 0.07,                    # 1
]
```

The optional `simulation.navigation_goal_observation=true` **adds two scalar
features**, normalized target height error and normalized remaining distance,
making the state 12-dimensional. All Trial 9/10 and A–D runs use **false**.

| | Depth Actor | LiDAR Actor |
| --- | --- | --- |
| Sensor | Simulated D435i 64×48 at 30 Hz, 10 m max | Simulated MID360 at 10 Hz, 40 m max |
| Policy preprocessing | `3/clip(depth, 0.3, 10)-0.6`, invalid masking, 4×4 maximum pooling; [B,12,16,1] | **1024 selected raycasts before measurement**, body XYZ metres [B,1024,3] plus validity [B,1024]; not a dense 20,000-point Actor input |
| Encoder | Conv32(2×2,stride2) → Conv64(3×3,stride1) → Conv128(3×3,stride1), VALID, LeakyReLU, flatten 1024 | Shared point-wise dense 3→64→128→1024, LeakyReLU, valid-only Max Pool 1024 |
| Fusion | Perception 1024→192 **added** to state 10→192 | Same |
| Recurrent | GRU hidden=192 per environment | Same |
| Actor output | 3 raw accelerations, squashed via tanh | Same |
| Physical execution | World XYZ acceleration in [-6,+6] m/s² → Mellinger attitude/thrust → Crazyflow 500 Hz | Same |
| Actor trainable parameters | **514,112** | **561,920** |
| Independent sensor Critic | CNN16/32 + MLP256/128→1; **49,601** parameters | Pointwise32/64 + MLP256/128→1; **55,233** parameters |

Stored A–D configs use `critic_uses_sensor=true`,
`critic_uses_privileged=false`; the Actor/Critic have independent weights.
The Depth Actor has a velocity-prediction head, but the A–D setting
`velocity_aux_weight=0` means that auxiliary head is not trained by this loss.
Returned auxiliary metadata such as `point_times`, `acquisition_time` and
`depth_mask` are not additional Actor inputs. The PPO decision period is
0.1 s; the simulator checks physics/collisions at 500 Hz.

The LiDAR training logs have only about **14.1%** valid returns in their latest
twenty updates (C/D trials), and Depth A has about **20.5%** valid pooled pixels.
These are train-rollout aggregates, not held-out sensor coverage estimates.

## GPU benchmark: cost must be measured end-to-end

Hardware: RTX 4090, JAX 0.11.2, Flax 0.12.10, 16 environments, 1024 rays,
10 Hz actions, 500 Hz physics and 208 procedural obstacle slots. Timings use
JAX synchronization and exclude first JIT compilation from warm medians.

Variant: compact all valid points into a 256-point static buffer when **all**
worlds have at most 256 returns; fall back to the original 1024-point encoder
otherwise. No rays are dropped in either path. The same original Flax
parameter keys/weights are retained. All-invalid, 120/256/257/716/1024-valid
point cases passed equivalent Actor output and parameter-gradient tests
(7 focused checks). The synthetic 12%-valid encoder subset and full Actor
tests were significantly faster:

| Workload | Original | Compact (256) | Outcome |
| --- | ---: | ---: | --- |
| PointNet encoder forward, B=16, 12%-valid | 0.448 ms | 0.211 ms | 2.12× faster |
| Encoder forward/backward, same | 1.008 ms | 0.412 ms | 2.45× faster |
| Whole Actor forward, B=16, same | 0.272 ms | 0.211 ms | 1.29× faster |
| Whole Actor forward/backward, same | 1.150 ms | 0.517 ms | 2.23× faster |
| Full PPO update, **real Crazyflow + LiDAR**, B=16, horizon32 | **0.350 s** | **0.448 s** | **28% slower** |

In the real end-to-end PPO run, initial loss differed by 1.5e-5 and
first-step Actor parameters by at most 5.85e-4, despite equivalent
encoder-only gradients. The observed actual sensor valid fraction during
the real run was 18–22% (versus 12% synthetic). The extra dynamic
indexing, branching and generated JAX graph increased total cost.
Dense input fallback also exhibited a small backward regression.

**Decision: REJECT compact PointNet implementation.** The temporary variant
was not committed. `simulation/networks.py` is unchanged; original A–D
checkpoints remain compatible. Raw paired timings and experimental scripts are
in the ignored worktree `tmp/benchmark_pointnet_paths.*`,
`tmp/benchmark_full_actor.*` and `tmp/benchmark_real_ppo_update.*`.
No speedup is claimed from microbenchmark results alone.

The discarded variant can be reconstructed **only inside ignored `tmp/`** for
reproduction; no production code needs modification:

```bash
python tmp/build_rejected_pointnet_candidate.py
PY=/home/tong/tongworkspace/simulation_dev/new_drone_playground/.pixi/envs/default/bin/python
SCIPY_ARRAY_API=1 PYTHONPATH=src:tmp \
  "$PY" tmp/benchmark_full_actor.py
SCIPY_ARRAY_API=1 PYTHONPATH=src:tmp \
  "$PY" tmp/benchmark_real_ppo_update.py
```

## Measured training/validation mismatch

Using `ProceduralScene.sample(key=99, worlds=256)`, capacity 208,
the same training seed and three existing start-sampling contracts:

| Reset choice | Median distance | Under 40 m | At least 80 m |
| --- | ---: | ---: | ---: |
| `collision_free` (Trial9/A–D) | **31.98 m** | **59.4%** | **11.3%** |
| `stratified` (Trial10) | 58.17 m | 43.4% | 40.6% |
| `endpoints` | 95.12 m | 0% | 100% |
| Frozen fixed scenes | **96 m nominal** | — | 100% |

The `collision_free` generator intentionally selects near-goal starts with
a nominal 35% probability; other starts are widely distributed over x=2..93 m.
Thus its high training success rate substantially reflects shorter
episodes; these cannot be interpreted as evidence of full 96 m traversal.

**Trial10 already tested stratified starts with all other Trial9 settings
held constant**, and did not achieve formal acceptance: Depth max 1/48,
LiDAR max 0/48 among recorded complete fixed screens. Do not repeat it
as an unevaluated new idea.

An independently verified observation alias shows that at [2,0,3] versus
[20,0,3], with identical world goal [98,0,3], velocity and attitude,
the current 10-D state vectors are **exactly equal**. With the optional 12-D
goal observation they differ by 0.1875 in the normalized remaining-distance
component. The sensor observations could break this alias when informative
obstacles are in view, but workspace-distance awareness is not explicit.

## Actual failure patterns

- Depth A best checkpoint update 560: **17/48** full-screen successes,
  confined to S02 (8), D01 (8) and D03 (1). Success flight-time median
  67.45 s. Other outcomes: 18 collisions, 13 out-of-bounds.
- Depth A update 1440: **0/48**, 17 collisions and 31 out-of-bounds.
  Training success in the final 20 log measurements still averaged roughly
  52% (unweighted). Indicates unstable transfer from sampled training starts.
- LiDAR C final: **0/48**, all 48 episodes collisions, median collision
  time 18.77 s. Recent training valid-return fraction ≈14.1% and
  Critic explained variance ≈0.923, so a well-fitting training Critic is
  **not** reliable evidence of navigation generalization.
- LiDAR D best update 960: **3/48**, all three successful episodes in D03;
  the other 45 ended in collision.

## Next controlled experiment (not yet launched)

1. **E1: observation ablation**. Only enable
   `simulation.navigation_goal_observation=true` (10→12 state dimensions)
   against Trial9. Keep start sampling, seeds, sensor, 500/10 Hz timing,
   reward, network widths, PPO parameters and frozen validation unchanged.
   This is a new network input contract, so initialize fresh weights; never
   resume a 10-dimensional Actor/Critic checkpoint.
2. **E2: long-flight curriculum**, separately: begin with short collision-free
   starts, increase the long-course proportion toward endpoint-like starts
   over training, and compare with E1/Trial9 using equal *environment
   interactions*, not just update counts. E2 changes the reset distribution,
   not the reward/network. A fixed `stratified` mix was already tested and
   does not count as a new candidate.
3. Score each candidate with full fixed-scene 48-episode screens, first
   arrival by scene, out-of-bounds/collision fractions and persistent
   multi-scene success, then independent 25-per-scene validation and frozen
   benchmark only when promoted. Never declare success from aggregate
   training return or selected single-scene successes.

The next optimization priority is **information/credit quality and
train–validation distance matching**, not making the PointNet MLP larger.
