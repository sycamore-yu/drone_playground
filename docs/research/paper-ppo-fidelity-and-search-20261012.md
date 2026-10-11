# Paper PPO: source audit and measured search

## Finding

Earlier Fan/VisFly experiments changed policy representations but retained the
project's navigation cost. They did not test the papers' original reward designs.
Commit `f2365f2` corrects the reward paths and several source-level control and
initialization differences. The current experiments remain **long-course transfers**
into Crazyflow; their maps and physics are not the original paper simulators.

Sources:

- Fan et al., *Flying in Highly Dynamic Environments with End-to-end Learning
  Approach*, RA-L 2025, arXiv 2503.14352, equations 5–16 and sections III–IV.
- VisFly official checkout `6ff29bfcbc752da4f9474f6f0ece29d836111a22` at
  `/home/tong/tongworkspace/reference_repos/VisFly`. Relevant files:
  `envs/NavigationEnv.py:85–99`, `envs/base/dynamics.py:610–710`,
  `configs/drone/drone_state.json`, `utils/policies/policies.py`,
  `utils/policies/extractors.py`, and both `cluttered_flight` experiment YAMLs.

## Agreement and remaining differences

| Item | Source | Current source-aligned implementation |
| --- | --- | --- |
| Fan range representation | 36 azimuth bins and 36 stored range frames | Implemented, with valid-only minimum pooling |
| Fan raw history | Aggregate a sliding window of raw point clouds before compression | Implemented in world coordinates; window length 3 is an explicit hypothesis because the paper does not give its numeric value |
| Fan rewards | Progress, velocity bounds, acceleration, jerk, static/dynamic clearance, collision and near-goal hover | Equations implemented and checked term by term; numeric weights are reconstructed and versioned |
| Fan value network | PPO value learning; full architecture not released | Independent reconstructed ResNet value network with the same goal/history inputs |
| Fan map | Random 10–20m arena, five dynamic obstacles | **Different:** project procedural 80m+ tasks and S01–D03 fixed validation |
| VisFly perception | 64×64 depth; CNN 6/12/18, 5/3/3 kernels; state/goal MLPs | Source layer sizes and PyTorch initialization distribution implemented |
| VisFly state | World position, wxyz quaternion, velocity, angular rate and absolute target | Corrected from the earlier xyzw transfer input |
| VisFly actions | Thrust mapped to 0–3g, rates ±2rad/s | Corrected from 0.4–1.6g and ±6rad/s |
| VisFly reward | Exact `NavigationEnv.get_reward` components | Ported; no inherited 20+160 terminal surcharge or Zhang/Liu loss |
| VisFly feature sharing | Shared encoder and joint actor/value Adam update | Added as separate `navigation_paper_visfly2025_shared` configuration |
| VisFly training scene | Original Habitat garage, about 8m start-to-goal | **Different:** 80m+ procedural course; no original garage score is claimed |
| VisFly clock/budget | 0.03s control step, 48 agents, rollout256 | 0.04s (25Hz) control, 500Hz Crazyflow physics, 16 agents, rollout256 |

The Fan 36×36 tensor is **time × azimuth**, not an occupancy map of 36×36
spatial cells. Depth resolution, map geometry, physics and task distance are
recorded separately. The long-course minimum remains at least 80m for all
new candidates; no short episode is counted as an 80m navigation success.

## Implementation checks

Reward definitions belong to `simulation/task_rewards.py` and are selected by
`Task.reward_profile`. Direct environment costs and learning costs use the same
paper functions. Source-aligned profiles bypass the old generic shaping layer.

Fan dynamic-obstacle distances and velocities are obtained from the same scene
geometry used by collision detection. The velocity dilation affects rewards
only; it is not additional privileged policy input. Missing geometry, zero
velocity and already-terminated worlds have tested finite behavior.
The four-dimensional policy state remains planar velocity plus the previous
network acceleration command, as stated in the paper. The configured base
reward is +0.1 before collision and -10 on collision; this collision treatment,
like the unpublished numeric weights, is a declared transfer assumption rather
than a number recovered from the article.

The PyTorch initializer uses uniform bounds `1/sqrt(fan_in)` for both kernel and
bias. This matches the source initialization distribution, not the PyTorch RNG
stream. The shared-encoder PPO keeps one trainable encoder in the optimizer;
actor and value losses both differentiate through it. Old nonshared PPO remains
the default. Old checkpoints, scenes, source records and action bounds retain
their original contracts.

Tests cover exact reward components, fixed/procedural geometry, raw history
expiry, action range, initialization scale, joint gradients, a genuine complete
Crazyflow PPO update, missing-evaluation handling and independent-seed promotion.

## Active and next experiments

All outputs are below:

`/home/tong/tongworkspace/simulation_dev/new_drone_playground/results/paper_ppo_v04`

| Candidate | Experiment | Purpose |
| --- | --- | --- |
| `fan_equations_s0` | `navigation_paper_fan2025_equations` | Reward-equation baseline with declared weight hypotheses |
| `visfly_source_s0` | `navigation_paper_visfly2025_source` | Source reward/action/initialization baseline |
| `fan_margin2_s0` | `navigation_paper_fan2025_equations_margin2` | Only safe distance changes from 1.2 to 2.0m |
| `visfly_shared_s0` | `navigation_paper_visfly2025_shared` | Shared feature learning and the corresponding source joint optimizer |

The two baselines started from random weights on 12 October 2026 and have real
GPU parameter updates. The former v03 25Hz trial was stopped after update2400
with a preserved checkpoint and recorded as interrupted; it was not marked
accepted or erased. Its reallocation record is in `source_alignment_reallocation.json`.

`tools/optimize_paper_ppo.py` implements a finite measured-search loop. It adopts
existing baselines and never starts duplicates. Its scalar objective is the
existing `screening_score`: minimum scene success rate plus one quarter of the
mean scene success rate. Only complete six-scene reports with eight trials per
scene yield an objective value. A missing or two-scene probe result remains
unknown; it is never converted to a zero score.

There are at most four seed-zero recipes. A recipe with a genuinely accepted
frozen benchmark is repeated from scratch with seeds1 and2 before further
search. A failed repeat permits the bounded alternative; one good seed does
not count as stable acceptance. Failed or interrupted runs remain recorded.
If the finite candidate budget ends without acceptance, the final study is a
negative result and requires a new evidence-based design decision.

The original acceptance criteria remain unchanged: six complete scene screens,
then 25 trials per scene with at least23 successes, followed by an independent
frozen benchmark and successful reproduction across three training seeds. Training
reward, Critic explained variance and a single favorable scene do not satisfy it.

The study uses ordinary Python/systemd and the optim-agent skill's measured
propose/evaluate/record procedure. It does not claim execution of the upstream
`optim_agent` Python package. `study.json` stores the actual current objective,
source/configuration identity and best observed report for every run.

No per-service CPU quota, RAM cap or swap cap is reintroduced. At most two
training processes run simultaneously, and the launcher checks available RAM
and VRAM before adding a job. Existing machine/account-level limits are unchanged.

## Reproduce and inspect

```bash
cd /home/tong/tongworkspace/simulation_dev/.worktrees/new_drone_playground/paper-ppo-fan-visfly-v01
PY=/home/tong/tongworkspace/simulation_dev/new_drone_playground/.pixi/envs/default/bin/python
ROOT=/home/tong/tongworkspace/simulation_dev/new_drone_playground/results/paper_ppo_v04
PYTHONPATH=src "$PY" tools/optimize_paper_ppo.py status --root "$ROOT" --python "$PY"
```

The read-only status command is safe while the manager is running. Full training
logs, checkpoints, source hashes and per-scene episode outcomes remain in each
run directory and `.scheduler/`.
