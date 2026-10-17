# ADR-0035: Prioritize verified OA learning recipes for the release

Status: Accepted, user approved 2026-10-17. Implementation: algorithm-specific
OA training presets and the Command Builder use the historical optimizer and
loss coefficients; ongoing results and source revisions remain immutable.

## Decision

The release target remains from-scratch convergence of the 42 supported Train
combinations. For OA BPTT and SHAC, reuse the historical successful recipe's
loss coefficients, optimizer and short-horizon update settings before attempting
new parameter searches. For PPO, start from the documented 512-environment-step
rollout, finite-horizon `goal_reaching` objective and privileged Critic
configuration used in the successful six-primary-map experiments. Historical
policy weights are never reused to initialize a fresh candidate.

Static and Dynamic OA keep independent `train/procedural` scene distributions,
matched long-flight geometry and separate frozen Benchmark suites. Previously
successful models trained by rotating fixed maps are **parameter evidence, not
proof of generalization** to these new random layouts.

## Historical evidence and explicit limitations

| Sensor / learner | Prior training updates | Historical eight-map score | Source directory |
|---|---:|---:|---|
| Depth APG / current BPTT family | 640 | 200/200 | `results/scratch_depth_apg_s0` |
| Depth SHAC | 800 | 199/200 | `results/scratch_safe_depth_shac_s0` |
| LiDAR APG / current BPTT family | 640 | 198/200 | `results/scratch_safe_lidar_apg_s0` |
| LiDAR SHAC | 1920 | 200/200 | `results/scratch_safe_lidar_shac_s0` |

BPTT Depth restores the original absence of an additional failed-episode cost
(current default is zero), and its recorded initial log standard deviation.
SHAC Depth and both LiDAR algorithms keep their recorded 160 failed-episode
cost. All four preserve LR=0.0003, Critic LR=0.001, horizon=32 and their
sensor-specific perception coefficients. The historical configuration does not
explicitly name the loss implementation; the current `zhang` and `liu` losses
are retained as compatible implementations, **not asserted to be byte-for-byte
identical to old code**. In deterministic BPTT/SHAC, initial Gaussian log std
is not used in the action path. `randomize_navigation_start` is semantically
ignored when the procedural scene supplies an explicit sampled start; the flag
is retained for configuration provenance.

For PPO, the original from-scratch Depth 256-world recipe achieved 149/150 over
six primary scenes, and 64-world variants achieved 150/150. The recipe uses
`gamma=gae_lambda=1`, actor LR=0.0002, Critic LR=0.0005, rollout length 512,
privileged Critic, entropy weight **0.005**, and four minibatches and Critic
epochs. The later entropy-zero improvement was measured in a *continued*
training ablation; it is not preferred over the confirmed from-scratch default.
The native iteration cap is 1,000 with an independent success confirmation.
This does not guarantee success on the matched procedural distribution.

LOTF and PointMassLag have no identical historical OA success run; they use the
same historical task reward and compatible optimizer coefficients as the native
learner. LOTF's body-rate interface does not support the acceleration-specific
named perception loss, which stays disabled. Transfer performance must be
measured rather than claimed.

## Operational switch

Preserve old frozen study source and every result/attempt. Queued candidates
receive explicitly resolved new YAMLs; failed candidates are requeued from
random initialization into distinct directories. Do not silently relabel a
failure as a pass, overwrite a used configuration, restore an old Actor, or
interrupt an unrelated process. Freeze/snapshot the manifest and state before
reconfiguration, and record an amendment mapping old and new cases.

No standalone historical-baseline training series, second Validation command,
or duplicate training loop is authorized. Configurations, losses, and algorithms
belong to the existing public Learning/Simulation stack; local queue mutation
scripts belong only under `results/`.
