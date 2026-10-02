# Tasks

Task identity is hovering, tracking, racing, or navigation. Hovering uses the tracking implementation with a constant reference.

`navigation/` groups rigid-body, acceleration-driven, and recurrent implementations of navigation; `tracking/` groups rigid-body and acceleration-driven reference tracking. `racing.py` owns directed gate progress and racing termination. These files name the task and actual state/execution interface, not the paper that selected them.

Observation construction belongs in `environments/observations/`. Reference generators belong in `environments/references.py`. Paper combinations belong in `configs/experiment/`.
The vendored `lsy_upstream/` retains its original license, asset paths, and numerical code.
