# Environment / Task

The environment layer follows the robot learning convention:

```
scene       -> geometry, obstacles, obstacle motion
sensor      -> measurement and capture clock
command     -> position goal, velocity command, reference
task        -> observation, reward, termination
dynamics    -> physical transition
environment -> these components running together
```

Training and evaluation construct the same Task implementation. Training selects
scene, command and initial-state distributions; evaluation freezes the policy and
uses reproducible conditions. The environment role is only `train` or `eval`.
Training-time checkpoint selection and the final benchmark both use the same
`eval` environment semantics; a benchmark is identified by its specification,
not by a third environment role.

`randomization.py` separates reset randomization, measurement noise, action
uncertainty and runtime disturbances. Dynamics sample supported physical
parameters at training reset. Nominal evaluation uses standard model and sensor
conditions and disables training-only effects.

LOTF supplies `lotf_high_fidelity` and `lotf_simplified` dynamics to the common tasks. Benchmark
protocols live under `benchmarks/`; scene names and acceptance thresholds do not
belong to the environment implementation.
