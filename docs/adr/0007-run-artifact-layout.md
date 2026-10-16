# ADR-0007: Run artifacts and the Orbax/TensorBoard transition

**Status:** Superseded for new runs by [ADR-0016](0016-results-layout-orbax-tensorboard.md) and [ADR-0026](0026-training-benchmark-release-separation.md).
**Approval:** The supersession was originally recorded as 2026-10-10; exact time was not retained.
**Implementation:** Historical v2 files remain preserved. Current readers use v3 Orbax only.
**Last reviewed:** 2026-10-16T05:28:58Z, working tree based on `7d59162`.

## Historical contract

Layout v2 used `run.json`, `metrics.jsonl`, ZIP training/policy archives, and numbered
`eval/` report directories. Those records retain their original configuration,
selection rules, seed identities, timing, and failed-episode denominators. They
are archival evidence; current code does not resume or reinterpret them.

## Current artifact ownership

A current run stores `config.yaml`, `run.json`, and TensorBoard events. Each
`checkpoints/model_NNNNNN/` contains `metadata.json`, a committed Orbax training
payload, and a frozen actor. Retention removes only old resumable training payloads;
actor snapshots and evaluation evidence remain. Benchmark writes its own
`episodes.csv`, `report.json`, failure records, and requested replay data.

The complete training state includes actor, critic, optimizer, random keys,
environment, delayed commands, and sensor history. Native methods do not create
empty neural checkpoints. `run.json` records lifecycle and provenance;
`report.json` records actual evaluation outcomes and sample counts. TensorBoard
curves cannot replace those records.

## Migration rules

Update every active reader when changing an artifact contract. Never rewrite
active runs or delete results, weights, trajectories, or replays as directory
cleanup. Preserve historical records under their source revision. Verify saving,
exact continuation, frozen inference, and report generation on the integration
branch. A successful format migration does not establish policy convergence.

See [training](../training.md), [Benchmark](../benchmark.md), and
[Orbax](https://orbax.readthedocs.io/en/latest/) for the current interfaces.
