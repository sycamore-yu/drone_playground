# Autoresearch Gym

Autoresearch Gym orchestrates all public training. Its external adapter launches
this project's BPTT, SHAC, or PPO runner, then evaluates the saved Orbax Actor
in a separate process. Candidate comparison stays outside Learning.

## Setup and one recipe

After installing the Pixi environment, put `uv` and `git` on PATH:

```bash
pixi run autoresearch setup
pixi run train experiment=oa_static_depth learning=ppo seed=0
```

Setup clones the sibling `../autoresearch-gym/` at revision
`7fd6391aecc3bf0ab8e8813cf4da7a053baa001f` and applies
`tools/autoresearch/upstream.patch` once. The upstream runner uses its Python 3.10
environment; native training uses this project's Python 3.12 environment. Setup
also links the upstream dashboard assets into the checkout.

`pixi run train` accepts Hydra recipe overrides, creates an ordinary upstream
task JSON and a session-local baseline candidate, registers the session in the
dashboard, and prints the resulting run summary. Autoresearch owns the output
directory and initialization of fresh candidates. An explicit `resume` path
restores a compatible complete training state. See [training](training.md) for
budget accounting and resumption.

## Candidate search

Task JSON fixes the experiment, algorithm, budget, scene cohort, and evaluation
seed base. Candidate `recipe` fields can change `learning.options.*`,
`method.actor.*`, and `simulation.num_envs`. The total interaction budget must
remain divisible by the rollout batch. Candidate search starts from random
weights and cannot override initialization, recovery, or evaluation conditions.

```bash
pixi run autoresearch run \
  --benchmark tools/autoresearch/navigation_depth_smoke.json \
  --tag depth-smoke --no-dashboard
```

This short task completes 16 PPO updates and 8,192 interactions, then evaluates
two episodes in each of six fixed scenes. It checks execution, not convergence.
The ordinary `navigation_depth.json` task uses 409,600 interactions and eight
episodes per scene. Keep the budget and cohort fixed within a search session.

```bash
experiment_session=results/autoresearch/sessions/depth-ppo
mkdir -p "$experiment_session/candidates"
cp tools/autoresearch/seed_trainable.py "$experiment_session/candidates/baseline.py"
pixi run autoresearch run --session-dir "$experiment_session" \
  --candidate "$experiment_session/candidates/baseline.py" \
  --tag baseline --no-dashboard
```

The default public search seed base is 1,600,000. Final Benchmark defaults to
2,000,000. Scene-specific offsets are stable and independent of the training
seed. Repeated candidate comparisons cannot also serve as an untouched final
test. Upstream search instructions live in `../autoresearch-gym/AUTORESEARCH.md`.

## Dashboard through SSH

Start a dashboard for an existing session without starting training:

```bash
pixi run autoresearch session-dashboard ensure \
  --session-dir results/autoresearch/sessions/depth-ppo \
  --host 127.0.0.1 --port 4174 --port-end 4174
```

Forward the port from a local computer, replacing `USER@SERVER` with its SSH target:

```bash
ssh -N -L 4174:127.0.0.1:4174 USER@SERVER
```

Open `http://127.0.0.1:4174/` or `/dashboard/`; the root redirects to the
dashboard. The session list includes registered training runs. Add
`?session=results/autoresearch/sessions/depth-ppo` to select one explicitly.
Use `session-dashboard teardown --session-dir <session>` to stop it.

### Training visuals

Select an active training session and click **Live Frame** for the latest
sampled state, or **Sampled Rollout** for a short trajectory from one training
world. **Off** stops creating new previews. Sampling starts on the next training
update; the optional renderer publishes at most once every eight seconds and
runs in a separate MuJoCo EGL process. Training does not construct a viewer or
pause to execute a second policy rollout. The viewer uses the saved MJCF assets
for fixed scenes and constructs equivalent MuJoCo primitives from the actual
sampled procedural obstacle state during navigation training.

Frames are 640 × 360 JPEG images; a manifest describes sampled trajectory frames.
The Dashboard serves them from the active session's `live/` directory. Renderer
errors appear in `live/preview-render.log`. Camera images and LiDAR points are
not part of this world-camera preview; use a sensor-recording Benchmark or RScope
replay when sensor visualization is required. Runs launched before this feature
cannot acquire training visuals retroactively. The default mode is Off so that
headless throughput is unchanged until a user requests a preview.

The Episode Return chart uses environment steps on the horizontal axis for
native training. Its blue series is the mean sum of per-step rewards over
episodes **completed in each logging window**. The optional green fixed-policy
probe evaluates complete episodes without updating the weights. The historical
`Training Mean Episode Return` card instead averages every completed training
episode since the start, weighted by episode count. Partial episodes do not
contribute. No completed episodes yields `-`, not zero. `Policy Updates` counts
rollout-and-optimizer iterations; PPO epochs are not extra environment steps.

`Best Candidate Score` is the outer-loop Benchmark objective (success rate for
the shipped OA recipes), not the training Return or the within-run Best score.
The `evaluation return` column is unavailable when the Benchmark reports only
flight outcomes. Return values from different reward recipes or auxiliary
objectives should not be compared as if they had the same physical units.

## Results

Standalone runs live under `results/autoresearch/runs/`; session runs live in
that session's `runs/`. Each run's `external/train/` and `external/eval/` contain
native configuration, checkpoint, and report artifacts. `summary.json`,
`results.jsonl`, and `outer_loop_log.md` support candidate comparisons.

`budget_exhausted` means the complete interaction budget was executed. The
selected Best Actor is evaluated when selection has produced one; otherwise the
Latest Actor is evaluated. All failed episodes remain in the report. Live and
final summaries share one native-event projection, with completed-episode counts,
weighted return/length means, and deterministic evaluation curves. TensorBoard
retains the same native measurements. An unmeasured episode mean is null.
