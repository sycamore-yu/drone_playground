# Experiment artifacts

This package owns persistent outputs for immutable runs: resolved configuration, source/code identity, dependencies, metrics, checkpoints and training state, evaluation reports, traces and optional replay bundles.

The on-disk model is:

```text
results/
├── runs/<task>/<method>/<run_id>/   # one real execution
├── selected/                        # lightweight selection manifests
└── scratch/                         # previews, diagnostics and legacy evidence
```

`layout.py` derives the Task/Method scope for new runs and locates an existing run by its stable `run_id`. `record.py` creates one run directory and records its full provenance. Selection tooling never mutates a run and does not duplicate checkpoint or replay bytes.

Replay visualization is an optional artifact, not part of run identity. Evaluation keeps numerical reports and traces regardless; large RScope/MuJoCo bundles are written only when `evaluation.record_replays=true`.

This package does not advance physics or schedule control. Those responsibilities stay in `runtime/` and the environment/execution layers.

`checkpoints.py` stores verified parameter trees and metadata. `training_state.py`
also preserves optimizer state and typed random keys. Neither module constructs
a network or environment; policy reconstruction belongs to `learning/inference.py`.
