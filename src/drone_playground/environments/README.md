# Environment / Task

The environment layer follows the robot learning convention:

```
scene       -> geometry, obstacles, obstacle motion
sensor      -> measurements and declared sample rate
reference   -> desired movement for tracking or racing
controller  -> reference/action to physical control
task        -> task initialization, observation, reward, termination
dynamics    -> native physical transition
environment -> resources, clocks and shared physical-interval execution
```

Training and evaluation construct the same Task implementation. Training selects
scene, command and initial-state distributions; evaluation freezes the policy and
uses reproducible conditions. The environment role is only `train` or `eval`.
Training-time checkpoint selection and the final benchmark both use the same
`eval` environment semantics; a benchmark is identified by its specification,
not by a third environment role.

`factory.py` constructs the six components. `initialization.py` owns shared
simulation and controller setup; a task can supply a native resource initializer
without implicitly completing environment construction in `bind()`.
Environment clocks determine physics substeps. The shared `ActionTransition`
performs scheduled commands and event accumulation; tasks supply event rules.

`randomization.py` handles reset randomization, measurement noise and action
uncertainty. Runtime force/torque evaluation belongs to `dynamics/disturbances.py`.
Dynamics sample supported physical parameters at training reset. Nominal
evaluation uses standard model and sensor conditions without training-only effects.

LOTF supplies `lotf_high_fidelity` and `lotf_simplified` dynamics to the common tasks. Benchmark
protocols live under `benchmarks/`; scene names and acceptance thresholds do not
belong to the environment implementation.
