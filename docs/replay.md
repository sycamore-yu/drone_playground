# Replay

Simulation exports recorded motion to native RScope `.mj_unroll` files. Replay
uses measured poses and timestamps; exporting does not rerun physics. Each record
includes `scene.xml`, required assets, and `rscope_meta.pkl` in its directory.
When Play or Benchmark enables `benchmark.record_replay=true`, the exporter also
creates `browser_replay.json` beside the RScope record. Reports list both
`replays` and `browser_replays`; the canonical
[visualization guide](visualization.md) explains how to use the browser client.

Enable replay in an actual Play or Benchmark run:

```bash
pixi run play controller=mellinger task.duration=2 benchmark.record_replay=true
pixi run replay replay_path=/absolute/path/to/episode.mj_unroll
```

The report lists exported replay paths. The native viewer requires a desktop
display. The CLI exposes only the selected record in a temporary RScope directory,
preserves the original metadata, and cleans up viewer threads on exit. The
locked RScope/MuJoCo combination includes a narrow overlay-lock handling fix.

With the RScope Viewer VS Code extension installed, open the `.mj_unroll` file
while leaving `rscope_meta.pkl` beside it. A remote workspace may select:

```json
{"rscope.pythonPath": "${workspaceFolder}/.pixi/envs/default/bin/python"}
```

## Export API

`simulation.replay.export_replay(directory, scene, times, positions,
quaternions_xyzw, measurements=None, plans=None)` accepts strictly increasing
times, world positions of shape `[T,3]`, and unit body-to-world quaternions of
shape `[T,4]`. Optional sensor times retain acquisition timing. Recorded dynamic
body positions can be supplied explicitly. Existing nonempty output directories
are rejected.

Sensor points and planner samples are display geometry and never enter collision
physics. Invalid/no-return points are omitted. Plans are displayed only during
their recorded validity interval. The exporter does not silently downsample;
large point clouds increase file and rendering cost.

`simulation.browser_replay.export(directory, max_frames=350)` exports a locally
generated RScope record as the exact `qpos`, Mocap poses and timestamps needed
for MuJoCo WASM. Browser downsampling does not alter the native record or the
Benchmark metrics. The converter loads a Python pickle: give it only the
project's trusted local records, not arbitrary uploaded `.mj_unroll` files.

Display velocity is a finite difference of recorded poses, with the final frame
reusing its predecessor. The required RScope reward channel is zero because the
export API does not receive reward; it is not an evaluation score. Metadata
records these conventions, geometry identity, resource hashes, and frame count.
RScope 0.0.8 uses the first frame interval for playback pacing, so irregular
recording intervals remain correct in the data but do not produce variable native
viewer pacing.

## Verification

`tests/test_replay.py` decodes real files through upstream RScope, checks poses,
geometry, motion, and display resources. CLI tests cover single-record selection,
cleanup, and a clear error without a display. Actual desktop drawing requires a
separate viewer check; current evidence is listed in [validation](validation.md).
