# Documentation

[Environments / Task Preview / Command Builder](command-builder.html) generates
supported Train, Play and Benchmark commands and displays actual recorded
MuJoCo WASM flights beside the selected method. Start with [installation](installation.md), then use
[training](training.md), [Autoresearch](autoresearch.md), [Benchmark](benchmark.md),
[controllers](control-models.md), [native planners](ros_planner.md), or
[visualization](visualization.md). [Replay](replay.md) describes the native
RScope record format and its browser export.

[Public GitHub Pages deployment](pages-deployment.md) explains the standalone
static export of selected, original native flights and its publication checks.

[Scope and acceptance](spec.md) defines the release contract.
[Validation](validation.md) records execution evidence and open checks.
[Experiments](experiments.md) describes reproducible comparisons.
[The glossary](../GLOSSARY.md) contains domain definitions.

[Racing MPC](racing-optimization.md) documents the two integrated controllers,
their source references, and the separate performance acceptance status.
[Controller compatibility](adr/0033-controller-compatibility-and-reference-tracking.md)
defines which tasks provide the reference data required by each method.

## Architecture decisions

`Accepted` records user approval; implementation and verification are separate.
ADR-0026 defines training and evaluation ownership, ADR-0027 defines optional
reward selection and checkpoint publication, and ADR-0028 defines paired
Benchmark cases. ADR-0029 specifies the new MJCF suites and randomization
presets; ADR-0030 covers Racing optimization and persistent training playback.
ADR-0033 records the compatible command catalog and general Full-State NMPC.
Earlier protocols retain historical context.
ADR-0022 remains experimental and is not an approved replacement architecture.

| ADR | Decision |
|---|---|
| [0001](adr/0001-simulation-learning-boundary.md) | Simulation and Learning responsibilities |
| [0002](adr/0002-crazyflow-upstream-and-style.md) | Crazyflow, dependencies, and code conventions |
| [0003](adr/0003-project-scope.md) | Initial research scope and acceptance |
| [0004](adr/0004-native-method-integration.md) | Native C++ and ROS integration |
| [0006](adr/0006-native-crazyflow-randomization.md) | Physical randomization through Crazyflow pipelines |
| [0007](adr/0007-run-artifact-layout.md) | Run artifacts and the Orbax/TensorBoard transition |
| [0008](adr/0008-independent-backward-dynamics-model.md) | Independent backward dynamics |
| [0009](adr/0009-delayed-data-in-episode-state.md) | Delay buffers belong to environment state |
| [0010](adr/0010-training-only-privileged-ppo-critic.md) | Training-only privileged PPO critic |
| [0011](adr/0011-independent-ppo-goal-observation.md) | Goal height and distance in an independent PPO experiment |
| [0012](adr/0012-sensor.md) | Snapshot sensor acquisition and one ray-query implementation |
| [0013](adr/0013-render.md) | Ideal depth generation and explicit camera configuration |
| [0014](adr/0014-rl-environment-and-method-interface.md) | Brax environment contract and trainable methods |
| [0015](adr/0015-three-stage-learning-acceptance.md) | Historical three-stage learning acceptance |
| [0016](adr/0016-results-layout-orbax-tensorboard.md) | Results with Orbax and TensorBoard |
| [0017](adr/0017-eight-acceptance-gates.md) | Eight acceptance areas |
| [0018](adr/0018-training-patience-and-frozen-selection.md) | Historical training patience and frozen selection |
| [0019](adr/0019-procedural-training-fixed-navigation-benchmark.md) | Procedural navigation training and fixed benchmarks |
| [0020](adr/0020-ppo-diagnostics-before-brax-replacement.md) | Diagnose perceptual PPO before replacing the learner |
| [0021](adr/0021-task-owned-reward-contract.md) | Task-owned rewards and Learning-owned auxiliary losses |
| [0022](adr/0022-navigation-reward-ablation.md) | Experimental navigation PPO reward ablations |
| [0023](adr/0023-navigation-world-rotation-observation.md) | World-frame navigation actions with orientation observations |
| [0024](adr/0024-navigation-ppo-progress-regression-screening.md) | Historical navigation progress and policy-regression screening |
| [0025](adr/0025-autoresearch-gym.md) | External training orchestration with Autoresearch Gym |
| [0026](adr/0026-training-benchmark-release-separation.md) | Separate training, simulation, and benchmarking |
| [0027](adr/0027-reward-selection-and-checkpoint-publication.md) | Reward selection and checkpoint publication |
| [0028](adr/0028-paired-benchmark-cases.md) | Paired initial flight cases for policy and planner comparisons |
| [0029](adr/0029-navigation-suites-and-randomization-presets.md) | Native navigation suites and domain randomization |
| [0030](adr/0030-racing-mpc-and-browser-replay.md) | Racing MPC baselines and persistent browser playback |
| [0033](adr/0033-controller-compatibility-and-reference-tracking.md) | Controller reference capabilities and Full-State NMPC |
| [0034](adr/0034-release-trainable-defaults.md) | Trainable defaults, task-matched sampling, and optional confirmation |
| [0035](adr/0035-historical-oa-training-recipes.md) | Prioritize verified OA learner coefficients in release Train presets |
| [0036](adr/0036-navigation-body-frame-action.md) | Egocentric acceleration policy contract and coordinate-system audit |
| [0037](adr/0037-oa-consistent-action-and-time-alignment.md) | Default OA body-frame controls, crossbeam training and sensor timestamp/pose alignment |
| [0038](adr/0038-tracking-racing-body-rate-unification.md) | Unified Tracking/Racing thrust/body-rate policies and retirement of velocity experiments |

Local research notes, transient diagnostics, and historical run archives remain
outside the public source surface. They preserve their original source identity
and protocol. Public operating documents describe the current implementation.
