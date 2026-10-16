# ADR-0017: Eight acceptance areas

**Status:** Accepted for the eight-area organization. Aggregate thresholds, training selection, and shared OA training are superseded by [ADR-0026](0026-training-benchmark-release-separation.md).
**Approval:** Originally recorded as 2026-10-10; exact time was not retained.
**Implementation:** The areas organize evidence; they are not runtime quality gates.
**Last reviewed:** 2026-10-16T05:28:58Z, working tree based on `7d59162`.

## Decision

Replace the former twenty overlapping C/S/L checklist entries with eight areas.
Historical identifiers remain interpretable in archived reports, without becoming
another active checklist.

| Area | Evidence |
|---|---|
| S1: installation and physics | Installable package, official Crazyflow, physical-state consistency for fixed inputs |
| S2: tasks and sensors | Task events, static/dynamic geometry, calibrated sensing, clocks, and information permissions |
| S3: methods and native planners | Policy/planner/controller composition and actual EGO/SUPER closed-loop execution |
| S4: replay and reproducibility | Recorded scene, sensing, commands, delays, provenance, and complete failure records |
| L1: learning correctness | PPO/BPTT/SHAC updates, gradients, terminal/truncation semantics, memory reset, and continuation |
| L2: perception learning | State/depth/LiDAR observation-to-action execution, including frozen actor isolation |
| L3: training quality | Independent random-initialized runs under a declared convergence protocol |
| L4: efficiency and reproducibility | Interactions, updates, compilation/steady timings, memory, power, total cost, and saved state |

Full 20,000-ray LiDAR scans and direct sampling of 1,024 policy rays are explicit
simulation recipes, not interchangeable performance measurements. S1/S2/L1/L2
are targeted correctness checks; S3/S4 demonstrate platform capabilities. L4 uses
measurements from the same training runs instead of requiring duplicate training.
VRAM occupancy is not a pass criterion.

## Historical numerical protocol

The original L3 rule used twelve training recipes and eighteen static/dynamic
evaluation cells, three seeds each, with multistage selection. It required 95/100
Tracking successes and successful RMSE at most 0.25 m, 90/100 Racing completions,
and 23/25 arrivals per primary OA scene. Original S3 required at least ten static
and ten dynamic episodes per planner and at least one arrival in each primary
static scene. Dynamic and extension outcomes were fully reported.

ADR-0026 removes universal hardcoded aggregate thresholds and approves eighteen
independent training recipes, three seeds each. Episode success definitions remain;
new quality criteria belong to the experiment protocol. See [scope](../spec.md)
and [validation evidence](../validation.md) for current status.
