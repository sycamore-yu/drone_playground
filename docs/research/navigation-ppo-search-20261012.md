# Navigation PPO: matched controls and convergence acceptance

## Current branch and evidence

Checked at `2026-10-14T03:03:09Z`. The integration branch is `v0.2` at
`3345f03`; reward ablations (`c60ab63`) and Scratch Matrix (`6480bc9`) are
already integrated. Their clean worktrees were removed after checking that
neither held results and archiving every temporary evidence file with SHA-256
verification. The manifest is at
`new_drone_playground/tmp/branch-organization-20261014/manifest.json`.
SANDO and the independent Brax branch remain separate.

Paper branch merge `051e455` incorporates the integration snapshot, its canonical
ADR numbering and snapshot sensors. Fan raw history consumes hits at their
acquisition pose rather than removed rolling-scan fields. The merge passed
55 targeted paper/sensor/search tests and two Fan snapshot history tests.
Paper adaptations remain unmerged into `v0.2`; convergence is not established.

The former `navigation_ppo_search_v01/study.json` and `paper_ppo_v04/study.json`
are historical observations. Actual run records identify Depth update1190 and
Fan margin2 update3359 as interrupted; no old trainers were active at audit.
Preserved screens show Depth 1/48 and Fan margin2 8/48, with all remaining
outcomes colliding. VisFly shared is 0/48 with collisions and boundary exits.
Historical Depth 17/48 used shorter training starts and is not a matched >=80m
control. The original four-architecture search did not finish.

Old Fan/VisFly controls also used different screening frequencies, patience and
interaction budgets from their proposed treatments. Their scores cannot establish
single-variable improvements. New runs use fresh matched controls and the current
sensor contract. Historical results and checkpoints remain at their original paths.

## Active experiment

The result root is `new_drone_playground/results/navigation_ppo_convergence_v02`.
Depth and LiDAR seed-zero >=80m controls started from random parameters at
`051e455`. Their actual source hashes and configurations are in each `run.json`
and `config.yaml`. Changes after this commit repair scheduler selection and
interrupted budget-screen recovery; the sensor, reward, networks and PPO update
used by these controls are unchanged.

Both long-start controls finished by `2026-10-14T03:51:26Z`:

| Control | Stop update | Best checkpoint | S01 | S02 | S03 | D01 | D02 | D03 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Depth seed0 | 3600, early stopped | 2160 | 8/8 | 8/8 | 0/8 | 0/8 | 0/8 | 0/8 |
| LiDAR seed0 | 2880, early stopped | 2880, tied zero | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 |

Depth's final screen fell to 0/48 (29 collisions, 19 boundary exits); its
16/48 checkpoint remains retained. LiDAR's final screen has the same outcome
counts. These are diagnostics, not full validation or frozen acceptance.
Mixed-start trials now run independently from random initialization. Their saved
recipes are checked against each control, excluding only output location.

The user approved a training-start-distribution ablation on 14 October 2026,
recorded at `2026-10-14T03:03:09Z`. It uses the existing collision-free mixed
sampling by setting `min_start_goal_distance_m=0`. About 35% of sampling draws
use the near-goal region and the rest use the broad course; collision rejection
and the existing >4m separation still apply. No curriculum framework is added.
Formal scenes, long-distance benchmark starts, success definitions and thresholds
remain fixed. This parameter decision does not approve a new observation or model.

| Family | Matched control | Single-variable candidates |
| --- | --- | --- |
| Depth CNN/GRU | Depth A, >=80m starts, actor LR 2e-4 | Mixed starts; world rotation input; fixed heading; LR 5e-5; existing 12-feature goal input |
| LiDAR PointNet/GRU | LiDAR D, >=80m starts, actor LR 1e-4 | Mixed starts; world rotation input; fixed heading; LR 2.5e-5; existing 12-feature goal input |
| Fan | Margin2 equations, fresh current-sensor control | Actor LR 2.5e-5; initial log std -1.5 |
| VisFly | Shared visual encoder, fresh current-sensor control | Initial log std -1; thrust range 0–2 times weight |

There are 18 seed-zero slots. Original Depth/LiDAR controls run first. Their
mixed-start, world-rotation and fixed-heading diagnostics take priority once their
matched controls finish.
A frozen-benchmark pass promotes the same recipe to independent seeds1 and2.
Once a sensor modality has three accepted seeds of one recipe, no competing
recipe or replication for that modality is launched. Interrupted trials remain
visible and do not prevent independent candidates from running; their saved
state is preserved for explicit contract-checked CLI resumption.

Existing privileged Critic support already includes goal-relative position,
time, clearance and ideal rays. It remains available for a later evidence-based
ablation without introducing another network interface.

The frame diagnostic follows a reproduced ambiguity: the original navigation
state encodes body-frame velocity, target direction and up, while acceleration
is decoded in world coordinates. `heading=target` changes vehicle yaw without
providing world yaw to the Actor. In empty returns, two rotated positions with
matching body-relative targets can therefore have identical instantaneous policy
inputs but need different world actions. The existing `heading=fixed` option
is a narrow test of this issue; it also changes where the camera points and
does not prove that the ambiguity caused convergence failure. This test changes
only heading; it retains >=80m training starts and is separate from mixed starts.

The user approved keeping world-frame actions and adding Actor orientation,
recorded at `2026-10-14T03:53:53Z`; see [ADR-0023](../adr/0023-navigation-world-rotation-observation.md).
The `world_rotation` candidates append the body-to-world rotation matrix through
the existing observation configuration. They retain target heading and >=80m
training starts, so each differs from its original control by only this input.

CPU replay of a copied Depth update2160 and LiDAR update2880 training state
reproduced behavior-policy GRU memory exactly. Replaying the same 32-step rollout
after one PPO update changed the next normalized action by mean/max
0.00157/0.00684 (Depth) and 0.00127/0.00949 (LiDAR). Older boundary memory stayed
fixed, so this does not rule out accumulated drift. It provides no current basis
for changing recurrent-state handling. Raw diagnostics and scripts are in
`new_drone_playground/tmp/navigation-ppo-review-20261014/memory-drift`.

## Screening and budgets

Every 122,880 transitions, evaluate a rotating static/dynamic pair, eight
episodes each. Every third checkpoint, screen all six primary scenes.
Any successful probe triggers a full screen at that checkpoint. The budget
boundary also forces a full screen; interrupted boundary evaluations must be
retried on resume without another training update or substitution by a probe.

Only complete six-scene screens affect ranking and patience. The allocation
score is `min(scene success rates) + 0.25 * mean(scene success rates)`.
Three non-improving full screens trigger early stopping after the minimum
budget, with improvement threshold 0.005. Report comparisons at matched
interactions; a maximum observed score alone is not an ablation conclusion.

| Family | Minimum transitions before pruning | Maximum transitions per pilot |
| --- | ---: | ---: |
| Depth/LiDAR | 737,280 | 2,211,840 |
| Fan | 4,423,680 | 8,847,360 |
| VisFly | 1,474,560 | 4,423,680 |

At most two GPU training processes run concurrently. Admission checks RAM,
VRAM and I/O pressure, without adding per-service CPU, memory or swap quotas.
The scheduler preserves the best complete screening checkpoint and the recent
resumable states through the existing Results policy. Missing scores stay unknown.

New workers also preserve each full screen's per-scene report, score and patience
in the existing TensorBoard event stream before checkpoint retention can remove
its directory. The prior writer recorded only the pass flag and overwrote its
single latest-evaluation index; pruned historical reports cannot be reconstructed
from those flags. The already-running controls keep their original writer;
available reports were copied read-only to `tmp/navigation-ppo-review-20261014/`
for this audit. This recording fix does not change optimization or acceptance.

## Acceptance and execution

A candidate must pass full validation: each S01/S02/S03/D01/D02/D03 scene at
least23/25 safe arrivals. The same selected checkpoint then runs a separate
frozen benchmark using independent seeds. Stable acceptance requires three
independent training seeds of the same recipe. Training reward, probes,
matched-control tests and budget exhaustion are not acceptance. Results from
different checkpoints, recipes or seeds are not spliced into a passing policy.

From the experiment worktree, the supported Pixi task is:

```bash
pixi run navigation-ppo-search status --root /absolute/path/to/results/navigation_ppo_convergence_v02
pixi run navigation-ppo-search run --root /absolute/path/to/results/navigation_ppo_convergence_v02 --max-parallel 2
```

The task accepts `--python` to reuse an existing Pixi interpreter across worktrees;
the launcher explicitly sets this worktree's `src` as its import path. Commands,
resolved configurations, source identity and per-scene outcomes are stored per
run; `study.json` is the scheduler snapshot. The finite queue ends with acceptance
or an explicit negative/incomplete result, not an unbounded parameter sweep.
