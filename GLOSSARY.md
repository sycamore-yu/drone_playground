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
