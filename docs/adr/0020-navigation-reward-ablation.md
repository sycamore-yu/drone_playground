# ADR-0020 — Navigation PPO reward ablations against Trial 9/10

**Status:** Experimental · 2026-10-11

## Goal and controls

Test whether weak fixed-scene navigation comes from event credit assignment,
workspace-boundary feedback, pre-contact obstacle risk or the short PPO discount.
Historical Trial 9 (collision-free starts) and Trial 10 (stratified starts) remain
immutable baselines. New experiments copy each sensor's measured Trial-9 training
proposal and initialize new network weights.

All arms retain Crazyflow physics, full random obstacle layouts, D435i/MID360
observation rights, action interface, network, PPO settings, original validation
seed cohort, and independent frozen benchmark. Base Task coefficients are read
from the historical Learning YAML through Hydra interpolation, with no numerical
change. Only the declared single term is modified.

| Arm | Sensor | Single logical intervention |
| --- | --- | --- |
| A | Depth | Event surcharge collision=160, OOB=240, timeout=40, numerical=160 |
| A | LiDAR | Event surcharge collision=240, OOB=160, timeout=40, numerical=160 |
| B | Depth | Signed six-face workspace boundary penalty, margin=1.5 m, weight=2 |
| C1 | LiDAR | Pre-contact approach risk, margin=2 m, weight=0.5, speed reference=3 m/s |
| D | Depth / LiDAR | PPO gamma 0.99 → 0.995 per 0.1-s policy step |

Event surcharges **replace** the historical 160-unit generic extra
penalty for the specified outcome; the baseline 20-unit event cost remains.
Continuous B/C terms are scaled by dt, gated after termination, and reported
separately. Approach speed is measured from geometrical clearance across
two consecutive times (including obstacle motion), clipped at 10 m/s and
stop-gradiented; clearance retains its physical gradient.

C1 retains the historical Liu perception/avoidance weights. Calibrate their
balance separately only after measuring the relative magnitudes of
approach_risk_cost and named_avoidance in real rollouts; changing both
simultaneously would not be a single-variable experiment.

## Evaluation and launch

Each checkpoint is screened on S01–S03/D01–D03 × 8 deterministic trials.
The original rule requires 6/8 per scene before testing 25 episodes/scene,
followed by a separate held-out benchmark. Acceptance requires three
independent seeds, not a favorable training-success rate.

Six seed-0 treatments are the bounded pilot budget. Seed 1/2 only start after
their treatment has passed the original full and frozen benchmark. All failures
and invalid configs remain recorded. If no treatment succeeds, report failure
and inspect trajectories before expanding the search space.

Commands in this isolated worktree:

    PYTHONPATH=src SCIPY_ARRAY_API=1 python tools/train_navigation_reward_ablations.py plan
    PYTHONPATH=src SCIPY_ARRAY_API=1 python tools/train_navigation_reward_ablations.py run --max-parallel 2
    PYTHONPATH=src SCIPY_ARRAY_API=1 python tools/train_navigation_reward_ablations.py status

The new experiments use a separate results root,
new_drone_playground/results/ppo_reward_ablation_v01.
Never overwrite Trial 9/10, the old scheduler's accepted Racing results, or
an existing checkpoint.
