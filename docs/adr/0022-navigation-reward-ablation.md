# ADR-0022: Experimental navigation PPO reward ablations

**Current workflow:** [ADR-0026](0026-training-benchmark-release-separation.md) supersedes historical checkpoint validation, screening, patience, and aggregate pass thresholds described below. Original experiment results retain their recorded protocol. Last reviewed 2026-10-16T06:24:13Z against `main@7d59162` and the working tree.

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

## Efficient validation loop (versioned, opt-in)

The first pilot budgets were dominated by repeated identical zero-success
screens across six nominal scenes, not by reward computation. To shorten the
feedback cycle, the A–D experiments alone use:

- **Every 80 PPO updates:** run an eight-episode diagnostic on a rotating pair
  of static/dynamic scenes (S01/D01, S02/D02, S03/D03).
- **Every third checkpoint:** always run the complete six-scene × eight-episode
  screening, even if recent probes were unsuccessful. Full screens therefore
  occur at updates 240, 480, 720 and so on.
- **If any diagnostic episode arrives:** immediately escalate that checkpoint
  to the complete six-scene screen. Do not wait for the next periodic sweep.
- Only a complete screen updates best-score/patience and can initiate the
  25-episode-per-scene full validation. A two-scene probe cannot select or
  accept a policy. A separate held-out benchmark remains the sole final test.
- **Patience = 3 complete screens after minimum update 800:** weak candidates
  are stopped sooner, but short diagnostic failures alone never trigger
  early stopping. The historical Trial 9/10 recipes and run statuses are
  unchanged.
- Once one arm passes the actual frozen benchmark at seed 0, train seeds 1
  and 2 of that arm ahead of other still-untried reward candidates. Do not
  mistake a single successful seed for overall acceptance.

The two-scene probes reuse the **same eight checkpoint-validation initial
conditions for each named scene** as the six-scene screen. Fixed cohort seeds
are now indexed by canonical scene identity instead of the enumeration
position within a temporary subset; benchmark seeds are not modified.

For always-zero probes, the nominal diagnostic episode count per three
checkpoints falls from 144 to 80 (two × 16 plus one × 48), or **44% fewer
screened episodes**. This is an episode-count bound, not a claimed measured
wall-clock speedup; new events record `evaluation_seconds` separately.

Real-shape independent CPU benchmarking (10 physical CPU cores, 16 worlds,
horizon 32, 208 obstacle capacity) measured **19.7 interactions/s** after
compilation. Previously recorded GPU runs achieved hundreds of interactions/s.
Therefore run simulation/PPO on GPU, use the CPU for Hydra preflight, test
execution, outcome aggregation and experiment orchestration; do not schedule
a full CPU PPO training as an equal replacement for a GPU slot.

The design borrows bounded, measured, multi-fidelity allocation from
ASHA/SuccessiveHalving without adding Ray or Optuna as a framework
dependency. Since a delayed-onset good RL policy is possible, every candidate
receives periodic full screens and a generous minimum training budget before
pruning. The search must stop after six pilot arms and explain negative
results rather than silently launching dozens of new random trials.
