# ADR-0006: Physical randomization through Crazyflow pipelines

**Status:** Accepted.
**Approval:** Originally recorded as 2026-10-09; exact time was not retained.
**Implementation:** Present in `simulation/randomization.py` and environment construction.
**Last reviewed:** 2026-10-16T05:28:58Z, working tree based on `7d59162`.

## Context

Robustness experiments need physical parameter error and external disturbances.
Crazyflow already exposes actual flight parameters and pure reset/step builders.
An extra environment wrapper or unrelated MJX parameter copy would duplicate state.

## Decision

Sample actual `SimData.params` through the official reset pipeline before building
the reset function. Use independent random state per world and update only worlds
selected by the reset mask. Sample from nominal parameters each episode, and keep
derived quantities such as inertia inverses consistent. Expose only parameters
used by the selected physical model.

Inject external forces, torques, and gusts at the native load/integration stage of
the step pipeline. Respect its frames and units; advance random processes by
physical simulation time. Assemble the pipeline before building the step function.

Keep the implementation small and within existing state/configuration facilities.
Controller nominal parameters remain separate from randomized plant parameters;
revealing true parameters to a method requires an explicit information contract.

## Consequences and evidence

Nominal recipes remain available. Robustness runs declare distributions and fixed
evaluation seeds, and checkpoint actual parameters plus random state.
[Crazyflow API](https://learnsyslab.github.io/crazyflow/user-guide/oo-api/) and
[pipelines](https://learnsyslab.github.io/crazyflow/user-guide/pipelines/) provide the
underlying mechanism. The pinned dependency was first inspected on 2026-10-09.
