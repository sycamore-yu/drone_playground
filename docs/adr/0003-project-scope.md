# ADR-0003: Initial research scope and acceptance

**Status:** Accepted for scope. Numbering was superseded by [ADR-0017](0017-eight-acceptance-gates.md) on 2026-10-10; training selection and aggregate thresholds were superseded by [ADR-0026](0026-training-benchmark-release-separation.md).
**Approval:** The original file did not retain a precise approval timestamp.
**Implementation:** The capabilities are present; current release and convergence evidence remain incomplete.
**Last reviewed:** 2026-10-16T05:28:58Z, working tree based on `7d59162`.

## Context

The first release supports reproducible perception, state-control, and native
planning experiments for a single quadrotor.

## Decision

Cover Tracking, Racing, and static/dynamic obstacle avoidance. Preserve the fixed
OA geometry and LSY Level0 racing course. Under ADR-0026, OA training uses random
obstacles while fixed maps belong to Benchmark.

Compare PPO, BPTT, and SHAC across Tracking, Racing, Static OA depth/LiDAR, and
Dynamic OA depth/LiDAR: eighteen combinations. ADR-0026 records the subsequent
approval of independent training and three seeds for each combination.

Use D435i depth and MID360 LiDAR profiles with published device parameters.
Keep method-specific preprocessing explicit. Integrate native EGO-Planner with
depth and SUPER with LiDAR for both OA suites.

Use common actor backbones for controlled algorithm comparisons: a DiffAero-style
MLP with widths `[256,128]`, LayerNorm, and ELU for state tasks; the Zhang CNN/GRU192
for depth; and the Liu PointNet/GRU192 for LiDAR. Critics and learning objectives
remain algorithm-specific. Independent experiments may adjust networks or recipes
using published methods, while preserving source attribution and comparable
execution conditions.

Use available GPU capacity and measure actual training efficiency. No universal
resource cap is part of this decision. Experiment budgets are explicit.

## Consequences

Execution checks and convergence experiments are separate. The former hardcoded
quality thresholds and multistage validation are historical; each new convergence
protocol states its own criteria. Historical frozen weights do not establish
convergence for a new independent initialization.
