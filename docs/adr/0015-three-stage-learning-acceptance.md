# ADR-0015: Historical three-stage learning acceptance

**Status:** Superseded for new runs by [ADR-0026](0026-training-benchmark-release-separation.md).
**Approval:** Originally recorded as 2026-10-10; exact time was not retained.
**Implementation:** The historical selection workflow has been removed from current training. Archived evidence retains this protocol.
**Last reviewed:** 2026-10-16T05:28:58Z, working tree based on `7d59162`.

## Historical decision

The approved change replaced three consecutive full evaluations with lightweight
screening, one complete candidate validation, and an independent frozen benchmark.
It retained the then-current task thresholds, eighteen evaluation cells, three
random initialization seeds per recipe, and exclusion of old-project weights.

Screening used eight episodes per scene: six primary OA scenes, or the single
Tracking/Racing case. Trigger thresholds were 6/8 for each OA scene, 7/8 for Racing,
and 7/8 with successful-position RMSE at most 0.35 m for Tracking. A triggered full
validation used 25 episodes per primary OA scene or 100 for state tasks. The first
full pass stopped updates and became the frozen candidate.

Seed bases were 1,000,000 for screening, 1,500,000 for complete candidate validation,
and at least 2,000,000 for final Benchmark. Final OA reports also included S06/D06,
which had no primary-scene gate. Repeatedly using final-test failures to select
weights was outside the protocol. Existing consecutive-pass runs retained their
original rule and saved configuration.

The 2026-10-15 stability study allowed source-versioned historical scratch evidence
and algorithm-specific rewards with shared task goals and thresholds. It also
required at least 900,000 continuation interactions for OA PPO and a declared
previous-success preservation criterion. The implementation was reviewed at
`2026-10-15T14:58:15Z` against `9cc19e3`; that review did not change approval time
or establish integration/convergence.

## Performance measurement retained

Measure actual warm update latency, interactions per second, peak memory, power,
and time to the experiment's quality criterion. Distinguish sensor cost,
evaluation cost, compilation, and steady execution. Keep JAX preallocation off.
Run throughput comparisons without competing training on the same GPU. Hardware
utilization alone is neither efficiency nor policy quality.

## Current interpretation

Current training has no screening, validation, patience, or score-based selection.
Orbax saves periodic and final models; Benchmark evaluates frozen methods. New
experiments use their declared quality protocol. Historical records remain under
their original source and are not rewritten by this supersession.
