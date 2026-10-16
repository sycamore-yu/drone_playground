# Controllers and dynamics

Methods are constructed from explicit Actor, action, planner, and controller
configuration. JAX methods retain explicit parameters, state, and RNG; native
solvers execute on the host. Both use the same task and physical outcomes.

| Configuration | Implementation | Execution |
|---|---|---|
| `controller=mellinger` | Crazyflow position-control conversion | JAX |
| `controller=so3` | EGO SO3Control outer-loop equations | JAX |
| `controller=sampling_mpc` | Sampling prediction and elite-mean update | Host with JAX prediction |
| `controller=attitude_mpc` | LSY nonlinear OCP with acados | Host |
| `controller=full_state_nmpc` | RPS98-inspired 10-state quaternion NMPC, body-rate/thrust output | Host, acados |
| `controller=racing_mpcc` | Arc-length contouring/lag/progress optimization for ordered gates | Host, acados |
| `controller=perception_mpc` | Known-gate image-visibility and trajectory costs (experimental) | Host with JAX prediction |
| `controller=ideal` | Ideal trajectory tracking | Host, direct state update |

Policy acceleration actions use Crazyflow's Mellinger conversion to attitude
and thrust. `controller` selects trajectory tracking for Planner/Controller
methods; it does not replace the Policy action conversion. Ideal tracking is
explicitly reported and does not simulate motor dynamics.

```bash
pixi run play controller=so3 task.reference=hover task.duration=0.2
pixi run play controller=sampling_mpc task.reference=hover task.duration=0.2
pixi run play task=tracking scene=empty sensor=state method=controller \
  controller=full_state_nmpc simulation.num_envs=1
pixi run train experiment=tracking_lotf learning=shac seed=0
pixi run train experiment=tracking_point_mass learning=bptt seed=0
```

Attitude MPC, Full-State NMPC and Racing MPCC require the optional acados
setup in [installation](installation.md). General trajectory trackers can be
used for Tracking, Racing, or behind an OA Planner; Racing MPCC cannot solve a
generic obstacle-avoidance trajectory problem. Controller applicability is
defined in [ADR-0033](adr/0033-controller-compatibility-and-reference-tracking.md).
Controllers retain their source equations, bounds, and attribution. Trajectory
samples remain host NumPy data so MPC can inspect the future horizon.

The Full-State NMPC builds its nominal predictor from the 10-state
position/quaternion/velocity and thrust/body-rate formulation of
[RPS98/mpc](https://github.com/RPS98/mpc). Its Crazyflow output is a real
BodyRate setpoint; it derives desired attitude and thrust from supplied
position/velocity/acceleration references. This is a local adaptation, not a
copy of the upstream model, and does not perform obstacle search.
The Mellinger/SO3 Hydra options remain usable as classical non-MPC trackers;
they are intentionally outside the Builder's selected MPC comparison catalog.

## Backward dynamics

`learning.backward_model=null` uses derivatives of the forward plant.
`point_mass_lag` supplies position, velocity, and acceleration derivatives for
acceleration actions. `lotf` supplies position, orientation, velocity, and derived
acceleration derivatives for body-rate/thrust actions. Other outputs retain
native derivatives. The forward plant remains the selected Crazyflow dynamics.

The backward model uses the command actually delivered at each physical step;
it cannot anticipate delayed actions. Point-mass lag accepts `time_constant`;
LOTF accepts an optional `mass`, defaulting to nominal vehicle mass. These are
physical surrogate models, separate from temporal gradient attenuation. PPO
does not differentiate through physics.

## Delay and randomization

`simulation.action_delay_s` and `sensor.config.latency` accept fixed seconds or
an episode-sampled `[low, high]` range. Commands follow the physics clock;
measurements preserve acquisition pose and acquisition/availability times.
Per-world buffers reset with their episode and are saved in complete checkpoints.

```bash
pixi run train experiment=tracking_point_mass learning=bptt \
  'simulation.action_delay_s=[0.025,0.05]' \
  '+simulation.randomization.mass=[0.9,1.1]' \
  '+simulation.randomization.motor_strength=[0.95,1.05]'
```

Crazyflow reset hooks sample physical parameters; step hooks apply external
forces and gusts. Controllers and action scaling use nominal parameters while
the plant uses sampled values. Disturbance options include `force_world_n`,
`torque_body_nm`, `gust_std_n`, and `gust_period_s`.
