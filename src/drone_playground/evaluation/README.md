# Evaluation

`run.py` is the shared entry point. Task implementations are grouped under `tracking/`, `navigation/`, and `racing.py`. Within a task, `policy`, `acceleration`, `recurrent`, and `external` describe actual execution interfaces. `mpc.py` runs the MPC controllers shared by tracking and racing.

`benchmarks.py` reads fixed benchmark definitions, selects checkpoints when explicitly requested by training, and validates benchmark reports. It does not define another environment role. Both training-time evaluation and final benchmark execution use `role=eval` and the same task implementation.

All failed episodes remain in the denominator. Policy and normalization parameters remain frozen. The selected benchmark determines cases, seeds, budgets, and thresholds; parameter/noise sweeps are not a separate framework layer.

Serialization, decisions and trajectory archives live in `artifacts/`; numerical rollout scheduling remains in `runtime/`.
