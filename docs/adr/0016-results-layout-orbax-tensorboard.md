# ADR-0016: Results with Orbax and TensorBoard

**Status:** Accepted; current details follow [ADR-0026](0026-training-benchmark-release-separation.md).
**Approval:** Originally recorded as 2026-10-10; exact time was not retained.
**Supersedes:** The new-run proposal in [ADR-0007](0007-run-artifact-layout.md).
**Implementation:** Orbax v3 and TensorBoard are present; ZIP and mixed-format readers have been removed.
**Last reviewed:** 2026-10-16T05:28:58Z, working tree based on `7d59162`.

## Decision

Each run owns one resolved `config.yaml`, `run.json`, and TensorBoard stream.
Model directories are numbered by the actual optimizer update:

```text
results/<run>/
  config.yaml
  run.json
  events.out.tfevents.*
  checkpoints/model_000100/
    metadata.json
    training/
    actor/
```

Training payloads preserve the full Flax state: actor, critic, optimizers, RNG,
physical/sensor state, delayed commands, and recurrent memory. Every current saved
model exports an actor for frozen execution. Both payloads must be committed.
`results.keep_checkpoints=3` bounds resumable state storage; all actor snapshots,
metadata, and evaluation evidence remain. TensorBoard uses update number as step.
No duplicate JSONL metric stream or Hydra log directory is needed.

`run.json` remains necessary for provenance, current lifecycle, counters, and the
latest checkpoint. Benchmark owns separate episode/report files and optional
replays. The earlier selected-actor-only export and screening subdirectories are
historical details superseded by ADR-0026.

## Migration and evidence

Preserve old archives without moving weights or changing active experiments.
Current code loads only v3 Orbax; archival ZIP runs require their original source.
There is no dual-format matrix reader or automatic migration tool.

The original implementation verified Orbax 0.12.7 with Flax PyTrees and typed RNG
keys, and TensorBoardX 2.6.4 event writing. Current retention, payload integrity,
selector, and continuation checks live in `tests/test_orbax_results.py` and
`tests/test_learning.py`. Storage correctness does not imply convergence.
