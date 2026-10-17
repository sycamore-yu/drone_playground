# ADR-0025: External training orchestration with Autoresearch Gym

**Status:** Accepted.
**Approval recorded:** 2026-10-15T15:34:00Z, recording the user's instruction to integrate Autoresearch Gym on a feature branch.
**Implementation:** Integrated from `feat/autoresearch-gym` into `main`. [ADR-0026](0026-training-benchmark-release-separation.md) makes this the public training entry and removes the former optional-budget lifecycle.

**Refinement approved:** 2026-10-17T05:54:56Z, the user's instruction to confirm 42 default training recipes through the existing candidate loop.
**Last reviewed:** 2026-10-17T06:35:00Z against `main@dac3fd4`. The fixed-budget task contract and replay propagation are integrated. Release candidates execute from their separately pinned source worktree through the existing GPU queue.

## Decision

Keep the project CLI, Trainer, Simulation, and Orbax implementation. An external
adapter starts native training and reads native results. Autoresearch Gym lives
in a sibling checkout, with its own Python environment. This repository records
the upstream revision and a small patch; it does not vendor that environment.

Candidate comparisons use fixed training iterations, environment count, and
rollout length. The declared Best/early-stop rule selects the Actor. Task JSON
fixes the physical experiment, algorithm, budget, and evaluation cohort.
Candidates may change training options other than rollout length and Actor
configuration. Search evaluation and the final benchmark use distinct seeds.
The backend validates the derived interaction count before launching training.

The initial integration added an optional interaction budget beside the old
selection workflow. ADR-0026 replaces that arrangement: all public training uses
Autoresearch, native training always has a finite budget, and Benchmark runs
outside Learning. Search candidates start from random weights; explicit public
training resumption restores a compatible complete Orbax state.

## Initial integration evidence

The original integration passed 63 targeted tests. A real GPU depth PPO run
completed 16 updates and 8,192 interactions, saved and loaded the Actor, and
finished twelve episodes across six primary scenes. It recorded zero successes
and twelve collisions. This established the execution path, not convergence.
The original evidence remains under
`results/autoresearch/sessions/integration-check/`.

See [Autoresearch](../autoresearch.md) for current commands and result layout.
