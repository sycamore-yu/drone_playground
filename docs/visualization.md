# Training metrics and flight visualization

Use **Autoresearch Dashboard as the browser entry point**. Training metrics,
short live previews, and the final saved Rollout stay in its existing session
page. The Dashboard opens the MuJoCo WASM viewer only when a user needs
interactive 3D flight playback. RScope and TensorBoard remain optional specialist
tools; they are not separate competing training dashboards.

| Tool | Owns | Entry point | Keep? |
|---|---|---|---|
| Autoresearch Dashboard | Runs, train/eval Return, Loss, status, live sample, final Rollout | `http://127.0.0.1:4174/dashboard/` | Primary browser UI |
| MuJoCo WASM | Interactive native-MJCF replay, exact recorded poses and mesh assets | Dashboard **Interactive 3D replay** or `docs/viewer/dist/` | 3D pane; shares artifact data |
| RScope | Full-resolution offline flight, sensor point clouds, planning overlays | `.mj_unroll` via VS Code / native viewer | Optional detailed analysis |
| TensorBoard | Optimizer trends, histogram/detail views on native `tfevents` | Existing run event files | Optional specialist diagnostics |

No additional physics simulator, training recorder, or checkpoint manager is
introduced by the WASM viewer. All renderers read the same recorded states;
changing the camera or scrubbing playback never mutates training or Benchmark.

## From training to the browser

`pixi run train` creates one Autoresearch session. The Dashboard presents
per-update metrics and optional Live Frame / Sampled Rollout. Training budget
completion automatically publishes a short true final-rollout segment:

- `live/last_training_rollout.webp`: click **Saved final rollout** for a quick
  recording without any browser build.
- `live/browser_replay.json` and `live/browser_scene.xml`: click
  **Interactive 3D replay** for a free-camera MuJoCo view of the same frames.

Only the final recorded fragment is guaranteed: it is the real learning
rollout, not a separate deterministic policy evaluation or an entire Episode.
Older training runs without captured states cannot acquire those states
retroactively. Sensor images and LiDAR hits are outside the training-preview
contract; use a Replay-enabled Benchmark for sensor overlays.

## From Benchmark to the browser

Enable `benchmark.record_replay=true`. Each selected flight is exported as
`replays/<case>/episode.mj_unroll` and `replays/<case>/browser_replay.json` with
the same timestamps, `qpos`, and scene Mocap state. The Benchmark report lists
both under `replays` and `browser_replays`. The browser reuses the exported
`scene.xml`, real Crazyflie 2.x STL meshes, and all referenced MJCF textures.
It does not simulate the saved commands in a fresh environment.

## Single-port Dashboard and WASM

Build the optional browser client once. This is deliberately separate from
`train` so ordinary headless experiments never invoke npm or install a browser.

```bash
pixi run viewer-build
pixi run autoresearch session-dashboard ensure \
  --session-dir results/autoresearch/sessions/<session> \
  --host 127.0.0.1 --port 4174 --port-end 4174
```

From Windows, forward only the Dashboard port:

```bash
ssh -N -L 4174:127.0.0.1:4174 USER@SERVER
```

Open `http://127.0.0.1:4174/dashboard/`, select a finished session, and click
**Interactive 3D replay**. The same local HTTP server serves the WASM client,
the browser record, the MJCF and its supporting assets. With
`docs/viewer/dist/index.html` built, no second long-lived service is required.

## Multiple worktrees and isolated verification

The viewer source is now in local `main`. An older Dashboard process running
from another worktree is **not automatically redeployed** by this Git commit.
Build this version from a mainline worktree and start its Dashboard to use the
single port 4174. During checkout-independent verification, a basic HTTP
server can serve this checkout on port 4175 without touching ongoing training:

```bash
pixi run viewer-build
python3 -m http.server 4175 --bind 127.0.0.1
```

Forward `ssh -N -L 4175:127.0.0.1:4175 USER@SERVER`, then open:

`http://127.0.0.1:4175/docs/viewer/dist/index.html?record=/results/mpcc-full-benchmark-8/replays/racing-0000/browser_replay.json`

The sample record above is present in the Racing feature worktree, not in a
fresh mainline checkout's ignored `results/` directory. For a mainline run,
use a locally produced `browser_replay.json` path instead.
The viewer auto-loads the record from the URL. Play starts playback at recorded
speed (select 0.25×/1×/2×); Pause, Timeline, Follow Drone, Camera Aircraft/Course,
and Flight Path visibility are interactive. **Camera: Aircraft** keeps the actual
8-cm-scale Crazyflie mesh in view rather than increasing its geometric size.
The page ships precompiled WASM via the official `@mujoco/mujoco` package.

The Python standard-library HTTP server is a local SSH development utility;
do not expose the repository root on a public network interface.

## What remains separate

Use TensorBoard only for details that the Dashboard omits, rather than adding
another front page. Preserve native RScope for high-density sensor points and
planner overlays not included in the lightweight WASM format. No separate
RScope-to-TensorBoard conversion pipeline or second live simulation stream is
needed. A browser record, or the existence of a trajectory video, is not a
substitute for a complete Benchmark report.
