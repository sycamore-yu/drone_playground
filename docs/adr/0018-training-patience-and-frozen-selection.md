# ADR-0018: Training patience and one-time frozen selection

**Status:** Accepted on 2026-10-10; the exact approval time was not retained.
**Implementation:** Historical workflow, superseded by [ADR-0026](0026-training-benchmark-release-separation.md). Current training removes patience and first-pass selection.

**Last reviewed:** 2026-10-16T05:37:19Z against `main@7d59162` and the working tree.

## Historical decision

The original workflow used a lightweight diagnostic, full checkpoint validation,
and one independent frozen benchmark. The first checkpoint meeting the full
validation criterion was selected. A successful benchmark ended that seed;
a failed benchmark remained immutable and required a new experimental identity.

Patience began after 800 updates and stopped a recipe after six diagnostic
checks without an improvement of at least 0.02. Each primary scene contributed
eight episodes. Navigation used the worst scene success rate plus 0.25 times the
mean scene rate. Tracking and Racing used success rate; Tracking also used a
bounded successful-episode RMSE bonus. These diagnostic scores were separate
from the former release thresholds. A fixed 1,000-update interruption was not
an equivalent stopping rule.

The run stored the best diagnostic score, stagnation count, termination reason,
optimizer state, and random state. ADR-0024 later added progress and regression
diagnostics. Historical runs retain their saved configuration and interpretation.

## Historical archives

Older ZIP runs could be audited once using their earliest fully passing
checkpoint. The retired `tools/freeze_legacy_candidate.py` wrote a candidate
identity, report, and complete episode denominator under
`results/scratch_v02/.scheduler/frozen_audits/`. Existing audit directories
prevented a second selection attempt on the same frozen cohort.

Current code does not load those ZIP checkpoints or run that audit tool.
The stored weights, configurations, audit records, and failed results remain
local evidence. Current experiments use the bounded training and independent
benchmark workflow in ADR-0026.
