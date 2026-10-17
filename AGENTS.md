# Project collaboration

## Documents and decisions

`GLOSSARY.md` is the domain context document and contains domain definitions.
For action coordinates, body/world conversions, delayed sensors or training
versus policy sampling rates, read ADR-0036/0037 and run `pixi run frame-contracts`.
Do not create a duplicate `CONTEXT.md`. Architecture decisions belong in
`docs/adr/`; `docs/README.md` is the documentation entry point.

Record an approved architecture decision in the current work, including its
reason, approval date, and actual implementation state. `Accepted` means the
decision was approved, not that implementation is complete. Questions, requests
for explanation, comparisons, and evaluations are not approval. Withdraw incorrect
approval records and correct the index. Do not extend an approval to another
model or interface. Ordinary parameters belong in their operating document.

The integration branch, currently `main`, owns the integrated ADR index. Worktree
documents are branch snapshots. Merge documents individually against the current
integration branch, resolve conflicts, check ADR numbers, and synchronize the
index. Unmerged implementation must remain identified as such. Keep long-term
plans and decisions in this existing document set.

Record approval time, implementation state, and review time separately. Use
ISO 8601 timestamps with a timezone, such as `2026-10-10T01:27:02Z`. Support
implementation claims with source or test evidence. A branch change does not
change approval time; a review updates only the review time and source revision.

This division follows Matt Pocock's
[domain-modeling](https://github.com/mattpocock/skills/blob/main/skills/engineering/domain-modeling/SKILL.md)
and [grill-with-docs](https://github.com/mattpocock/skills/blob/main/skills/engineering/grill-with-docs/SKILL.md),
using this project's existing paths.

## Scope before extension

Establish the behavior the user wants to change, then inspect existing code,
upstream public APIs, and mature implementations. Distinguish changes to forward
physics, backward dynamics, controllers, and gradient handling. Backward dynamics
are not merely gradient attenuation. A question does not approve a prohibition
on additional forward models.

Use existing parameters, presets, pure functions, or narrow interfaces where
they close the actual gap. Explain unnecessary maintenance costs before adding
backends, registries, compatibility layers, configuration dimensions, or state
transitions. Mentioning another project is not a request to transplant it.

Choose methods at construction. JAX methods keep explicit parameters, state, and
RNG. Native C++/ROS methods execute on the host. They share environment and
evaluation semantics without an arbitrary-stage generic Pipeline.

Verify facts available in code directly. Ask only unresolved questions that
change interface, data, or experimental meaning, with a recommended option.
Do not request approval again for an accepted decision.

## Changes and verification

Documentation tasks change documentation only. Implementation tasks update
related callers, configuration, and tests together. Keep temporary checks in
`tmp/` and regression tests in `tests/`. Start with the smallest check that
verifies the changed behavior.

Never delete results, weights, or replays as directory cleanup. Artifact layout
changes update actual readers and references under ADR-0007 and its successors.
ADR-0026 makes Orbax v3 the sole current checkpoint and report contract. Do not
restore ZIP/v1/v2 loading, missing-default repair, dual-layout report readers, or
migration tools. Historical `results/` remain read-only evidence; destruction
requires separate authorization. The 75 v1 training archives explicitly removed
on 2026-10-09 do not make the retained historical Actors executable by current code.

Worktrees are development directories, not experiment archives. Before removing
a merged worktree and branch with `git worktree remove` and `git branch -d`,
check uncommitted changes, unique `results/`, and important `tmp/` evidence.
Keep active training and unique data intact. Preserve the source, configuration,
and evaluation provenance of every research result.
