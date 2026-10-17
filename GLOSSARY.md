# Domain glossary

Terms for quadrotor learning, planning, control, and experimental comparison.
Architecture decisions are recorded in [the ADR index](docs/README.md).

## Tasks and experiments

**Task**: The objective of a flight episode, including its success, failure, and
progress rules.

**Tracking**: Following a time-indexed motion reference. Tracking error measures
how closely the vehicle follows that reference.

**Racing**: Passing gates in their prescribed order and direction. Flight time
is evaluated after establishing a valid completion of the course.

**Navigation**: Reaching a goal safely while satisfying obstacle and task
constraints.

**Obstacle Avoidance (OA)**: Navigation through an environment containing
obstacles that the vehicle must avoid.

**Static Obstacle Avoidance (Static OA)**: An obstacle-avoidance case whose
obstacles remain fixed throughout the episode.

**Dynamic Obstacle Avoidance (Dynamic OA)**: An obstacle-avoidance case containing
moving obstacles. Static obstacles may also be present.

**Scene**: The spatial geometry, obstacles, gates, and prescribed motion of an
environment.

**Procedural Scene**: A scene generated from a declared distribution and random
seed. Different samples can vary obstacle layouts, motion, start states, and goals.

**Episode**: One flight from an independently initialized state to its first
success, failure, or deadline. Each retry is a separate episode.

**Training Seed**: The initial random condition for an independent parameter
training run. One trained policy can be evaluated over many episode seeds.

**Benchmark Case**: A predefined scene, initial state, goal, external conditions,
and evaluation protocol.

**Benchmark Suite**: A declared set of benchmark cases evaluated under a common
measurement protocol.

## Physics, information, and motion

**Dynamics**: The physical transition from a vehicle state and physical input
to its next state.

**Backward Dynamics Model**: A dynamics model used to calculate sensitivities
of states and actions during training. Its state, inputs, and transition equations
may differ from those used to advance the physical simulation.

**Derivative Rule**: The relationship used to propagate input and output
sensitivities. It can use the forward equations' derivatives or an explicitly
declared approximation.

**Gradient Decay**: Scaling a propagated gradient by a specified factor while
preserving the corresponding forward value.

**Transport Delay**: The time between producing a measurement or command and
making it available to its recipient. Sampling periods and actuator response
times describe separate effects.

**Sensor Measurement**: A physical observation acquired at a specified time in
a defined coordinate system.

**World Frame (W)**: The simulation's inertial coordinate frame. Navigation
positions, planned trajectories and the state estimator use world coordinates.

**Body Frame (B)**: The rigid-body FLU frame (x forward, y left, z up). The
xyzw attitude quaternion represents body-to-world orientation.

**Frame Transport**: Converting a vector or point between frames with a
specified attitude and, for points, a position at a specified timestamp.
Scalar collective thrust is frame-independent; angular body rates are body
vectors; world-reference Euler angles are an orientation, not a world vector.

**Sensor Age**: Policy decision time minus the last available measurement's
acquisition time. Nonzero age can arise from a lower sensor sampling frequency
even when communication latency is zero. Motion compensation changes the
coordinate representation of stale data, not the time when it was measured.

**Local Yaw Offset**: A policy's heading command relative to the vehicle's
current world yaw at command issuance; unlike an absolute world yaw angle,
zero offset requests keeping the current heading.

**Snapshot Acquisition**: Acquiring all measurements in a frame at the same
sensor and scene pose. Acquisition and delivery can occur at different times.

**Observation**: The measurements, states, goals, and representations a method
is allowed to use.

**Privileged Information**: Simulation truth, maps, or future information outside
a method's declared observation permissions for an experiment.

**Path**: An ordered, parameterized geometric route.

**Trajectory**: A motion reference with an explicit time relationship and validity
interval. It can include position, velocity, acceleration, and other prescribed
derivatives.

**Waypoint**: A spatial target or an element of an ordered target sequence.
Timing is specified separately when needed.

**Setpoint**: A requested control target with defined units, coordinates, and
time semantics.

**Control Interface**: The accepted control quantities, their ordering, units,
coordinate systems, and valid ranges. Attitude, angular-rate, and acceleration
interfaces represent different physical inputs.

## Methods and learning

**Planner**: A method that uses permitted information to compute a path,
trajectory, or other motion plan. Its computation can include learning and
optimization.

**Controller**: A method that converts vehicle state, references, or setpoints
into lower-level control requests. Its computation can include optimization or
learning.

**Trajectory-Tracking Controller**: A controller that follows an externally
specified, time-indexed trajectory. It does not, by that ability alone,
compute a collision-free path or determine gate-passing order.

**Task-Specific Controller**: A controller whose optimization problem depends
on task-specific state, constraints or reference semantics; its applicability
cannot be inferred from its shared physical output interface alone.

**Controller Compatibility**: A match between the task's available reference,
the controller's required information and the physical output contract.
Compatibility does not establish convergence, safety or real-time performance.

**Policy**: A mapping from permitted observations and optional internal memory
to decisions.

**Method**: A decision process used in closed-loop flight. It can combine a
planner, controller, and policy.

**Training Algorithm**: An update rule that adjusts learnable parameters using
samples, losses, or demonstrations.

**Physics Step**: One numerical integration step of the physical dynamics. A
policy/environment step can contain several physics steps.

**Environment Step**: One decision-and-transition cycle of an environment. It
can include multiple physics steps and does not necessarily end an episode.

**Rollout Length**: The number of environment steps collected per parallel
environment in one training iteration. A rollout may cross episode boundaries.

**Training Iteration (Update)**: One collection of rollouts followed by the
declared algorithm's parameter update. Repeated policy/critic optimization
epochs over that batch belong to the same iteration.

**Iteration Budget**: The maximum number of training iterations in a run,
independent of the number of environments or the rollout length.

**Environment Samples**: The observed count of environment transitions in a
run. It depends on the environment count, rollout length, iteration count, and
any explicitly declared auxiliary sampling.

**Backpropagation Through Time (BPTT)**: Differentiating an unrolled sequence of
state transitions to update policy parameters from its trajectory objective.

**Forward Simulation**: Advancing physical flight, scene motion, and measurements
from a given state and decision to obtain the next state and events.

**Ideal Tracking**: A response model whose position and velocity equal the
reference, with attitude derived from reference acceleration and heading.

**Checkpoint**: Saved parameters and state associated with a particular training
update. A resumable checkpoint contains the state needed to continue that run.

**Latest Checkpoint**: The most recent successfully saved checkpoint in a run,
ordered by training update.

**Best Checkpoint**: The checkpoint designated best by the training method's
declared criterion. Reward-based selection compares the mean undiscounted return
of completed episodes from a declared training or fixed evaluation distribution.
A selector may resolve to the latest checkpoint when the
method records no best checkpoint; that fallback does not establish higher quality.

**Candidate**: One proposed training configuration evaluated within a search
experiment under its declared budget and comparison protocol.

## Research claims

**Controlled Comparison**: A comparison that holds experimental conditions fixed
except for the variable being studied.

**Named Adaptation**: Applying an identifiable published method to new conditions
while recording which parts were retained and which were changed.

**Convergence Acceptance**: Establishing that a training method stably reaches
predefined task quality under a declared evaluation protocol.

**From-scratch Training**: A run whose actor, critic, and optimizer start from
the run's own random initialization. Resuming that same run preserves its
from-scratch origin.

**Pretrained Initialization**: Starting a new run from learned parameters produced
by another run or source. Its provenance includes the source training cost.

## Relationships

- A task can be exercised in several scenes or benchmark cases.
- A benchmark suite fixes the cases and measurement protocol for a comparison.
- A sensor measurement is a physical observation; a method observation specifies
  which information that method receives.
- A training algorithm changes parameters; a frozen method executes those
  parameters independently of training.
- A trajectory adds timing to a motion reference; a path describes its geometry.
- A candidate identifies a training proposal; a checkpoint identifies saved
  state within one training run.


**Collective Thrust and Body Rates (CTBR)**: A control command containing scalar
collective thrust and three desired angular velocities about the vehicle body
axes. This project expresses thrust in newtons and rates in rad/s; the Actor
order is `[T, wx, wy, wz]`. A desired attitude angle is a different control
quantity and requires an attitude controller before the rate controller.
