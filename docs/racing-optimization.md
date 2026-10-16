# Racing controllers

## Native Racing MPCC

`controller=racing_mpcc` implements an acados nonlinear Model Predictive
Contouring Controller. It augments the nominal Crazyflow `so_rpy` state with
arc progress, and uses a locally linearized arc-length reference of the
gate-ordered Racing spline. Stage costs penalize contour/lag error, attitude
and control effort, while rewarding feasible progress. The controller imposes
altitude, tilt, thrust and progress-speed limits and uses SQP-RTI.

It draws on [the LSY acados MPCC course implementation](https://github.com/lihengrongfei/lsy_drone_racing_ws26)
and the literature on quadrotor MPCC, but is a separately authored local
controller rather than a line-for-line reproduction of either.

```bash
pixi run setup-acados
export ACADOS_SOURCE_DIR="$PWD/.cache/acados"
export ACADOS_PYTHON_PATH="$PWD/.cache/acados-python"
pixi run play task=racing scene=racing sensor=state method=controller \
  controller=racing_mpcc simulation.num_envs=1
```

The fixed Level0 Racing Benchmark completed **8/8 matched cases**, each with
the required five directional gate crossings. Mean completion time was
**14.682 s**, with sample standard deviation **0.045 s**. The observed CPU
controller solve measurement was **7.31 ms** mean, **8.51 ms** P95 over 20
synchronized decisions. These results apply only to that fixed track and
physical parameter setup; full randomized track and dynamics results remain
unmeasured.

## Experimental JAX perception-aware sampling MPC

`controller=perception_mpc` predicts attitude-controlled rotor-drag Crazyflow
dynamics over multiple sampled action sequences. It uses known Racing gate
centers as projection landmarks and penalizes view-cone violations and
image-plane motion alongside reference tracking and control effort. It is
inspired by [RPG PAMPC](https://github.com/uzh-rpg/rpg_mpc) and
[DeepMind MJPC](https://github.com/google-deepmind/mujoco_mpc), without
importing the original ACADO solver or any project-specific ROS dependency.

```bash
pixi run play task=racing scene=racing sensor=state method=controller \
  controller=perception_mpc controller.samples=32 \
  controller.horizon=4 controller.prediction_seconds=0.16
```

Short Crazyflow Play and one-episode Benchmark were executable. With 32 samples
and four stages, CPU decision time measured **83.46 ms** mean, exceeding the
50 Hz method period of 20 ms. This is an experimental, known-landmark
controller: do not interpret its objectives as measurements from D435i or
onboard vision. Full-course Racing success and hard real-time behavior are
not established.

## Excluded original RPG research adapters

The original RPG Time-Optimal + Attitude MPC adapter and the original RPG
ROS1/ACADO PAMPC controller are intentionally excluded from local `main`.
They remain isolated in the `feature/racing-mpc-implementation` research
branch. Their numerical feasibility, collision checks, and model differences
are documented there, not exposed as runnable mainline controller presets.

## Future controller comparisons

The current classical tracking controllers are Sampling MPC and acados LSY
Attitude MPC, both independent of Racing's new contouring formulation. Planner
baselines should supply a common time-stamped position/velocity/acceleration
reference, then be paired with the *same* Tracking Controller to distinguish
planning performance from tracking quality.

Potential next methods, not yet installed or benchmarked:

| Repository | Actual role | Integration priority |
|---|---|---|
| [RPS98/mpc](https://github.com/RPS98/mpc) | acados full-state quadrotor NMPC with explicit intermediate and terminal trajectory references | First choice if a separate full-state tracking MPC is needed; align its state and actuator conventions with Crazyflow |
| [UZH RPG Quadrotor Control](https://github.com/uzh-rpg/rpg_quadrotor_control) | Differential-flatness/rotor-drag-aware trajectory tracking and body-rate control | Valuable classical high-speed tracker; it is **not an MPC** |
| [UZH RPG High-MPC](https://github.com/uzh-rpg/high_mpc) | Learned high-level decision variables wrapped around MPC for moving-gate flight | Next Racing-specific research comparison, requires separate policy training |
| [CA-AC-MPC](https://github.com/prisma-lab/CA-AC-MPC) | PyTorch actor-critic with differentiable MPC and C++/CUDA accelerated iLQR | Defer: a new learning/solver stack, disproportionate for a baseline |
| [ChanJoon/mpcc](https://github.com/ChanJoon/mpcc) | ACADO/ROS-based multirotor contouring MPC | Redundant with the existing native acados MPCC; not a direct current-system port |

The public [MIGHTY](https://github.com/mit-acl/mighty) repository's separate
`mpc` package tracks **ground robots**. The public
[SANDO](https://github.com/mit-acl/sando) fake simulator interpolates planner
position/velocity/acceleration goals rather than solving quadrotor NMPC.
Neither provides a verified UAV MPC component that could simply replace our
Tracking Controller. Adapt the *reference contract*, not those simulators.
