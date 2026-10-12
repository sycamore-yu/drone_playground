# Four-architecture Navigation PPO search — 12 October 2026

The search now covers the original Depth CNN/GRU and LiDAR PointNet/GRU,
Fan's historical LiDAR representation, and the VisFly shared visual encoder.
The experiment code stays on `feat/paper-ppo-fan-visfly-v01`.

## Baselines and treatments

| Family | Baseline | First treatment | Second treatment |
| --- | --- | --- | --- |
| Original Depth | Historical Depth A recipe, >=80m starts | Actor LR 2e-4 to 5e-5 | Existing goal-observation option, 10 to 12 state features |
| Original LiDAR | Historical LiDAR D recipe, >=80m starts | Actor LR 1e-4 to 2.5e-5 | Existing goal-observation option, 10 to 12 state features |
| Fan | Existing v04 `fan_margin2_s0` | Actor LR 1e-4 to 2.5e-5 | Initial log standard deviation -2.5 to -1.5 |
| VisFly | Existing v04 `visfly_shared_s0` | Initial log standard deviation 0 to -1 | Thrust range 0–3 to 0–2 times weight |

Each treatment changes one resolved recipe field relative to its family
baseline. The hypotheses are that smaller actor steps reduce excessive KL
updates, explicit goal distance helps long flights, and better-scaled
exploration improves flight before obstacle learning. The experiments test
these hypotheses; their implementation does not establish effectiveness.

The two original baselines preserve the saved network, observation, task
reward and learning options exactly. Their >=80m training starts differ
from the historical short-start distribution. Historical 17/48 (Depth A)
and 3/48 (LiDAR D) results are context, not matched new-population controls.

The current Fan worker is adopted read-only and keeps running. The completed
VisFly shared baseline is adopted without another training run. Old Trials
9/10, A–D, and paper v01–v04 outputs are not overwritten. All new candidates
start from random parameters. The original v04 manager is stopped; its
independent Fan training service is preserved.

## Screening and budgets

New candidates use the same six fixed validation scenes and seed bank.
Every 122,880 environment transitions, evaluate one rotating static/dynamic
pair, eight episodes each. Every third checkpoint, evaluate all six scenes.
Any successful probe triggers a full screen at that checkpoint. A budget
boundary also forces a full screen before preserving the final state.

Only full screens affect ranking and patience. The score remains
`min(scene success rates) + 0.25 * mean(scene success rates)`.
Three non-improving full screens trigger early stopping after the minimum
budget. The improvement threshold is 0.005, small enough to recognize one
additional success when the weakest scene has not changed.

| Family | Minimum transitions before pruning | Maximum transitions per new pilot |
| --- | ---: | ---: |
| Original Depth/LiDAR | 737,280 | 2,211,840 |
| Fan | 4,423,680 | 8,847,360 |
| VisFly | 1,474,560 | 4,423,680 |

Fan's larger minimum retains enough budget to reach the approximately
four-million-transition region where its existing baseline first completed
a fixed scene. Compare candidates at matched interactions within a family;
these are not equal-compute budgets across architectures.

The preset has 12 candidate slots: two adopted baselines and ten possible new
seed-zero runs. It first covers the original architectures, then selects
one-factor trials using parent scores and family coverage. At most two GPU
training processes run concurrently. No per-service CPU, memory or Swap
quota is introduced. Admission checks available memory, VRAM and I/O pressure.

A candidate must pass full validation and the separate frozen benchmark
before seeds 1 and 2 are started. Three accepted seeds of the same recipe
are required for stable acceptance. Successes from different recipes cannot
be combined. Once a sensor modality has one such recipe, competing candidates
for that modality are no longer launched. Missing scores remain unknown.
Budget exhaustion preserves the checkpoint and is not acceptance.

## Execution

The active result root is `new_drone_playground/results/navigation_ppo_search_v01`.
The scheduler writes `study.json`; each new trial stores its command, resolved
configuration, update logs, checkpoints and evaluation reports. It uses
`tools/optimize_paper_ppo.py run --suite navigation --max-parallel 2`.
The user service is `dp-navigation-ppo-search-20261012.service`.

The preset ends when the finite candidate set is exhausted or both sensor
modalities have a three-seed accepted recipe. The scheduler reports remaining
failures; it does not silently expand into another random parameter sweep.
