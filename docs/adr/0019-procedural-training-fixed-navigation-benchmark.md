# ADR-0019 — Procedural Navigation Training vs Fixed Benchmark

**Status:** Accepted; staged implementation and CPU verification, 2026-10-10

**Target repository:** `new_drone_playground` / `drone_playground` v0.2
**Source branch at last verified access:** `feat/scratch-matrix-v02`

## Context

The current `navigation_mixed_depth.yaml` and `navigation_mixed_lidar.yaml` rotate through `[S01, S02, S03, D01, D02, D03, S06, D06]` with `learning.scene_updates: 20`. For 64 parallel environments and horizon 32, 20 parameter updates yield `20 * 64 * 32 = 40,960` environment transitions on **one shared, fixed scene geometry**. Environments have separate states and randomized navigation starts but are not separately randomized obstacle maps. `simulation.randomization` and `simulation.disturbance` are empty in current formal recipes. Thus, former S06/D06 evaluations do not establish held-out *geometry* generalization.

The benchmark scores of 12/18 accepted cells must remain historically intact. Altering the training data protocol changes an experiment version; never silently rewrite an existing `run/config.yaml`, checkpoint or frozen report.

## Decision

1. **Fixed benchmark.** Freeze the geometry, motion laws, hashes and episode-seed protocol for `S01–S03`, `D01–D03`, `S06`, `D06`. `S01–D03` remain primary acceptance scenes (25 episodes/scene, at least 23 successes per scene); `S06/D06` remain supplementary, report-only stress tests. Formal scores are computed only from fixed, independent seeds, never backpropagated into training.
2. **Procedural training.** New `Navigation` recipes must use a separate `scene=train/procedural` scene family; no training episode may use those eight benchmark MJCF geometries. Sample per-world, per-episode layouts of static cylinders, vertical/horizontal boxes, and moving cubes with randomized positions, size, density, motion phase, speed, direction and dynamic-goal state. Include randomized start and target states with feasible separation and safe clearances.
3. **Same Task, physics, and observation.** Physics and collision clearance, LiDAR/depth ray intersections, policy observations, rewards and reset termination must use **the same per-world geometry state** in the same simulation step. No observation-only fake obstacle randomization. Continue Crazyflow physics and common Actor/Trainer interfaces.
4. **Static JAX shapes.** Specify `max_geometries` at construction; represent variable obstacle count by active masks, and store geometric parameters and dynamic-motion parameters as JAX arrays with leading `num_envs`. Do not parse or recompile MJCF for every episode. Every reset uses PRNG keys derived from `(training_seed, world_index, episode_counter)`; no Python/NumPy RNG during jitted training steps.
5. **Reachability and reset validity.** Reject collisions against start/goal safety bubbles and physical bounds. Enforce an explicitly demonstrated path-existence constraint for static geometry (initially via validated pre-generated geometry bank / offline 3-D reachability check); random moving obstacles must not make the task permanently impossible by construction. Save generator version and distribution config alongside the run.
6. **Curriculum is optional, not an implicit evaluation shortcut.** First release can sample easy/medium/hard and static/dynamic distributions in a declared, reproducible mixture. If a difficulty curriculum is used, give its explicit stage schedule in YAML, common across PPO/APG/SHAC. Do not tune the schedule on frozen benchmark seed outcomes.
7. **Generalization reporting.** Use three non-overlapping splits: train generators/seeds, checkpoint-validation fixed seeds, and untouched benchmark seeds. Under the new from-scratch procedural protocol, S06/D06 can be called *unseen fixed geometry* diagnostics if withheld from training, but they are historically known to researchers. For stronger evidence, add a distinct sealed, procedurally generated test-map split never accessed for model selection or architecture changes.
8. **Domain randomization independent of scene randomization.** Enable random mass/inertia/motor drag or stochastic wind only via explicit separate YAML options, after geometry behavior passes smoke tests. Physics DR is not evidence for obstacle-layout generalization.
9. **No historical reinterpretation.** Preserve all existing accepted results; new runs are versioned and independently initialized. The old cyclic `scene_updates=20` prescription becomes a historical configuration, not the default for the new procedural protocol.

## Engineering phases and verification gates

**A. Freeze and provenance (no GPU).** Check `Scene('S01'..'D06').geometry_hash` and benchmark seed protocol against source records. Snapshot the historical cyclic recipe and mark S06/D06 as `seen_in_training` for old runs.

**B. Train `SceneState`/`SceneDistribution` (CPU-first).** Introduce a JAX PyTree with `[B,N,3]` positions/sizes, `[B,N]` active masks, and per-object trajectory/phase arrays. `Scene` remains an immutable compiled geometry backend; the train-only variant takes a `SceneState`. Random scene state must be part of the episode state and Orbax training checkpoint for exact resumption.

**C. Geometry tests.** Unit-test deterministic repeats for identical seed/world/episode, statistically distinct worlds and episodes, masking, endpoint safety, static path reachability, dynamic phase changes, and identical collision/LiDAR/depth geometry. Compare CPU jit/vmap and GPU throughput against the existing `full` and `policy` LiDAR modes.

**D. Selection isolation.** Train only from randomized layouts. Screen with a fixed training-diagnostic seed partition, validate on a disjoint declared split, and run the frozen S01–D06 benchmark exactly once per accepted seed. Do not use per-scene frozen failures to choose new hyperparameters.

**E. PPO and Racing diagnostics.** Provide explicit `PPO` value explained variance, value-target/advantage standard deviations, KL, clip fraction, entropy/exploration std, and action saturation. Compare `Racing APG/SHAC` on identical *training diagnostic* seeds with recorded gate index, contact time/location, applied low-level commands, and terminal-critic Jacobian norm. Prefer a controlled terminal-value ablation before modifying the Racing task acceptance criteria.

**F. Scheduler safety.** Mark obsolete interrupted ZIP/3-pass and pre-Patience v3 runs as `legacy_retired`/`pre_patience_retired`. Never auto-resume them. New random initialization requires first-full-validation selection and configurable `min_updates=800`, `patience=6`, `min_delta=0.02`; do not use an arbitrary 1,000-update SIGINT. Keep historical checkpoints, run IDs, and frozen reports untouched.

## Non-goals

- No changes to fixed test scenes, official success thresholds or earlier accepted scores.
- No adversarial selection of random maps from the frozen benchmark.
- No new PPO success claim until new independent random seeds and frozen tests actually pass.

## References

- P2M: https://github.com/arclab-hku/P2M (`resources/envs/single/env.py`, `env_utils.py`). Static obstacle terrain generation, dynamic obstacle state sampling/updates, randomized drone reset.
- NavRL: https://github.com/Zhefan-Xu/NavRL (`isaac-training/training/scripts/env.py`). Height/size categories, random obstacle origins/dynamic goal updates, randomized start/target.
- Isaac Lab: https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.managers.html. Explicit `prestartup`, `startup`, `reset`, `interval` randomization boundaries.
- MuJoCo Playground: https://github.com/google-deepmind/mujoco_playground/blob/main/learning/train_jax_ppo.py. Explicit optional `domain_randomization` and JAX PPO settings.
- PPO reference: https://spinningup.openai.com/en/latest/algorithms/ppo.html; https://docs.cleanrl.dev/rl-algorithms/ppo/.
