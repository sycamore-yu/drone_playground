# ADR-0014: Brax environment contract and trainable methods

**Status:** Accepted. Lifecycle ownership is updated by [ADR-0026](0026-training-benchmark-release-separation.md).
**Approval:** Originally recorded as 2026-10-09; exact time was not retained.
**Implementation:** Configurable actors, critics, methods, objectives, concrete environments, and the optional Brax adapter are present. Optional-backend convergence is not established.
**Last reviewed:** 2026-10-16T05:28:58Z, working tree based on `7d59162`.

## Decision

Choose actors, critics, trainable methods, action contracts, and objectives during
construction. Algorithms consume supplied policy/environment operations, rather
than inferring network or action meaning from a task or sensor name. Actors own
output size and recurrent memory; critics have independent information permissions.
PPO retains action-distribution and log-probability operations; BPTT/SHAC retain
the differentiable execution path.

Keep one physical `EnvState` and pure `reset(rng)` / `step(state, action)` behavior.
The Brax adapter owns Brax `State` fields (`pipeline_state`, observations, reward,
done, metrics, and info) plus its size/backend interface. Do not duplicate physical
truth. Crazyflow already batches worlds, so native Brax uses `wrap_env=False`
instead of adding a second vmap wrapper. Verify key shapes and reset only completed
worlds.

Preserve terminated/truncated distinctions and terminal observations. Reset samples
initial states and physical disturbances. Complete training state includes network,
optimizer, RNG, environment, and method histories. Changed checkpoint structures
require a fresh compatible experiment; current policy forbids legacy migration.

Tasks/environments own task reward, physical outcomes, and termination. Learning
objectives own network-specific auxiliary predictions and history. A changed
reward definition is a distinct recipe.

`BaseEnv` owns shared physics, sensing, clocks, delays, and actions. `TrackingEnv`,
`RacingEnv`, and `NavigationEnv` specialize observations and task behavior through
one `make_environment` entry. Existing task references and metadata remain shared;
there is no new environment registry.

Brax PPO and APG remain optional direct upstream comparisons. Upstream APG keeps
its official name; the project's algorithm is BPTT. The adapter does not add another
persistent trainer lifecycle. Native ROS planners remain host methods. A future
learned optimization method owns its differentiable mathematics inside the Method;
the environment does not interpret its private solver variables.

## Recorded checks and limits

The original separate environment (Brax 0.14.2, JAX 0.9.2, Crazyflow 0.3.2) ran
four-world, 64-step PPO and a two-update APG check with finite nonzero gradients.
Those checks establish interoperability, not convergence or performance comparisons.
Brax is optional (`.[brax]`) and its full compatibility with the default newer JAX
stack remains a separate check. The original full CPU attempt received SIGTERM
after about 66 cases; only the completed targeted checks were recorded as passing.

Current roles are bounded training in `learning/runner.py`, updates in `Trainer`,
physical rollout in `simulation/runner.py`, and frozen evaluation in Benchmark.
No additional public Runner class is required.
