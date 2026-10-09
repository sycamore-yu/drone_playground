"""Named perception objectives and state-only temporal gradient decay.

Trajectories are time-major [T,B,3] in consistent coordinates and SI units.
Clearance is obstacle-surface distance minus vehicle radius, in meters, with
shape [T,B] or [T,B,K]. Positive approach_speed has that same shape, in m/s.
Supply relative approach speed for moving obstacles, not absolute vehicle speed.
Losses average valid samples, not their sum. Call on contiguous episode segments;
the trainer owns terminal masking, reset, and aggregation across segments.

Zhang references DiffPhysDrone 271936190b5c/main_cuda.py. Liu follows equations
4--8 of Learning to Fly from Point Clouds via Differentiable Simulation; its
beta/jerk defaults are the reconstruction settings in learning-methods.md.
Unlike the upstream Zhang shifted full-window slice, velocity smoothing here is
explicitly causal, includes the current sample and uses shorter prefix windows.
All numeric recipe arguments are fixed hyperparameters (close over them for JIT).
"""

import jax
import jax.numpy as jnp


def _norm(vector: jax.Array) -> jax.Array:
    """Euclidean norm with the zero subgradient at zero, without an epsilon bias."""
    squared = jnp.sum(jnp.square(vector), axis=-1)
    return jnp.where(squared > 0, jnp.sqrt(jnp.where(squared > 0, squared, 1)), 0)


def causal_velocity_average(velocity: jax.Array, *, window: int = 30) -> jax.Array:
    """Mean of v[max(0,t-window+1):t+1], preserving [T,B,3] shape."""
    if velocity.ndim != 3 or velocity.shape[-1] != 3 or velocity.shape[0] == 0:
        raise ValueError("Expected a nonempty velocity trajectory [T,B,3]")
    if window < 1:
        raise ValueError("window must be positive")
    cumulative = jnp.concatenate((jnp.zeros_like(velocity[:1]), jnp.cumsum(velocity, axis=0)))
    end = jnp.arange(1, velocity.shape[0] + 1)
    start = jnp.maximum(0, end - window)
    return (cumulative[end] - cumulative[start]) / (end - start)[:, None, None]


def velocity_huber_loss(
    velocity: jax.Array,
    target_velocity: jax.Array,
    *,
    window: int = 30,
    delta: float = 1.0,
    norm_weight: float = 1.0,
    component_weight: float = 0.0,
) -> jax.Array:
    """mean(norm_weight*H_delta(||vbar-target||) + component_weight*sum_i H_delta(e_i)).

    H_delta(e) = e²/2 for |e|<=delta, else delta*(|e|-delta/2).
    Targets align with the current time, have shape [T,B,3], and are not detached.
    Zhang uses weights (1,0); Liu equation 5 uses (0.8,0.6).
    """
    if velocity.shape != target_velocity.shape or delta <= 0:
        raise ValueError("Velocity/target shapes must match and Huber delta must be positive")
    error = causal_velocity_average(velocity, window=window) - target_velocity
    squared = jnp.sum(error**2, axis=-1)
    radial = jnp.where(
        squared <= delta**2,
        0.5 * squared,
        delta * (jnp.sqrt(jnp.maximum(squared, delta**2)) - 0.5 * delta),
    )
    absolute = jnp.abs(error)
    component = jnp.where(absolute <= delta, 0.5 * error**2, delta * (absolute - 0.5 * delta)).sum(
        axis=-1
    )
    return jnp.mean(norm_weight * radial + component_weight * component)


def approach_weighted_clearance(
    clearance: jax.Array,
    approach_speed: jax.Array,
    *,
    mask: jax.Array | None = None,
    margin: float = 1.0,
    collision_beta: float = 32.0,
    minimum_approach_speed: float = 1.0,
) -> tuple[jax.Array, jax.Array]:
    """Return (mean(w*relu(margin-c)²), mean(w*softplus(-collision_beta*c))).

    w = stop_gradient(max(approach_speed, minimum_approach_speed)). There is no
    division by collision_beta. Zhang uses a floor of 1; Liu's nonnegative
    approach weighting uses 0. Masked no-hit entries may contain NaN/Inf; they
    contribute neither value nor gradient, and an entirely masked batch costs 0.
    """
    if clearance.shape != approach_speed.shape:
        raise ValueError("clearance and approach_speed must have identical shapes")
    if mask is None:
        mask = jnp.ones(clearance.shape, dtype=jnp.bool_)
    if mask.shape != clearance.shape or mask.dtype != jnp.bool_:
        raise ValueError("clearance mask must be boolean and match clearance shape")
    if collision_beta <= 0 or margin < 0 or minimum_approach_speed < 0:
        raise ValueError("Expected positive collision_beta and nonnegative margin/approach floor")
    distance = jnp.where(mask, clearance, 0)
    weight = jax.lax.stop_gradient(
        jnp.maximum(jnp.where(mask, approach_speed, 0), minimum_approach_speed)
    )
    weight = jnp.where(mask, weight, 0)
    count = jnp.maximum(jnp.sum(mask), 1)
    barrier = jnp.sum(weight * jax.nn.relu(margin - distance) ** 2) / count
    collision = jnp.sum(weight * jax.nn.softplus(-collision_beta * distance)) / count
    return barrier, collision


def velocity_auxiliary_loss(prediction: jax.Array, velocity: jax.Array) -> jax.Array:
    """Component-mean squared velocity error; the observed velocity target is detached."""
    if prediction.shape != velocity.shape:
        raise ValueError("Velocity prediction and target shapes must match")
    return jnp.mean((prediction - jax.lax.stop_gradient(velocity)) ** 2)


def _jerk(acceleration: jax.Array, dt: float) -> jax.Array:
    if acceleration.ndim != 3 or acceleration.shape[-1] != 3 or acceleration.shape[0] == 0:
        raise ValueError("Expected a nonempty acceleration trajectory [T,B,3]")
    if dt <= 0:
        raise ValueError("dt must be positive, in seconds")
    if acceleration.shape[0] == 1:
        return jnp.zeros_like(acceleration)
    return jnp.diff(acceleration, axis=0) / dt


def zhang_loss(
    velocity: jax.Array,
    target_velocity: jax.Array,
    acceleration: jax.Array,
    clearance: jax.Array,
    approach_speed: jax.Array,
    *,
    dt: float,
    velocity_prediction: jax.Array | None = None,
    clearance_mask: jax.Array | None = None,
    velocity_window: int = 30,
    huber_delta: float = 1.0,
    velocity_weight: float = 1.0,
    clearance_weight: float = 1.5,
    collision_weight: float = 2.0,
    acceleration_weight: float = 0.01,
    jerk_weight: float = 0.001,
    velocity_aux_weight: float = 0.0,
    clearance_margin: float = 1.0,
    collision_beta: float = 32.0,
    minimum_approach_speed: float = 1.0,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Zhang adapted objective; returns (weighted total, unweighted scalar terms).

    Velocity: Huber of the norm of causally smoothed velocity error. Clearance and
    collision: approach_weighted_clearance. Acceleration: mean(||a||²). Jerk:
    mean(||diff(a)/dt||²), zero for T=1; prepend prior acceleration if required.
    Auxiliary velocity: component MSE against detached velocity. It is disabled
    by default and requires predictions when its weight is nonzero (upstream
    uses 2). Acceleration is the caller's commanded net acceleration, without
    gravity; physical conversion stays in Simulation. No unused legacy costs.
    """
    if acceleration.shape != velocity.shape:
        raise ValueError("Acceleration and velocity trajectories must have matching shapes")
    if velocity_aux_weight != 0 and velocity_prediction is None:
        raise ValueError("velocity_prediction is required for a nonzero velocity_aux_weight")
    barrier, collision = approach_weighted_clearance(
        clearance,
        approach_speed,
        mask=clearance_mask,
        margin=clearance_margin,
        collision_beta=collision_beta,
        minimum_approach_speed=minimum_approach_speed,
    )
    terms = {
        "velocity": velocity_huber_loss(
            velocity, target_velocity, window=velocity_window, delta=huber_delta
        ),
        "clearance": barrier,
        "collision": collision,
        "acceleration": jnp.mean(jnp.sum(acceleration**2, axis=-1)),
        "jerk": jnp.mean(jnp.sum(_jerk(acceleration, dt) ** 2, axis=-1)),
        "velocity_aux": (
            jnp.zeros((), dtype=velocity.dtype)
            if velocity_prediction is None
            else velocity_auxiliary_loss(velocity_prediction, velocity)
        ),
    }
    total = (
        velocity_weight * terms["velocity"]
        + clearance_weight * terms["clearance"]
        + collision_weight * terms["collision"]
        + acceleration_weight * terms["acceleration"]
        + jerk_weight * terms["jerk"]
        + velocity_aux_weight * terms["velocity_aux"]
    )
    return total, terms


def liu_loss(
    velocity: jax.Array,
    target_velocity: jax.Array,
    acceleration: jax.Array,
    clearance: jax.Array,
    approach_speed: jax.Array,
    *,
    dt: float,
    clearance_mask: jax.Array | None = None,
    velocity_window: int = 30,
    huber_delta: float = 1.0,
    velocity_norm_weight: float = 0.8,
    velocity_component_weight: float = 0.6,
    velocity_weight: float = 1.0,
    collision_weight: float = 1.5,
    acceleration_weight: float = 0.01,
    jerk_weight: float = 0.001,
    beta1: float = 4 / 3,
    beta2: float = 32.0,
    jerk_mean_weight: float = 1.0,
    jerk_variance_weight: float = 0.1,
    clearance_margin: float = 1.0,
    minimum_approach_speed: float = 0.0,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Liu equations 4--8 with documented reconstruction defaults; (total, terms).

    L_v = .8*mean(H_delta(||vbar-target||)) + .6*mean(sum_i H_delta(e_i)).
    L_c = mean(w*(relu(margin-clearance)² + beta1*softplus(-beta2*clearance))).
    L_a = mean(||a||²), the squared norm in equation 7. For j=diff(a)/dt,
    L_j = jerk_mean_weight*mean(||j||) + jerk_variance_weight*mean_B(var_T(||j||)).
    Variance uses population normalization, is per trajectory, and never mixes
    environments. T=1 gives zero jerk. Zero norms have a finite zero subgradient.
    """
    if acceleration.shape != velocity.shape:
        raise ValueError("Acceleration and velocity trajectories must have matching shapes")
    barrier, collision = approach_weighted_clearance(
        clearance,
        approach_speed,
        mask=clearance_mask,
        margin=clearance_margin,
        collision_beta=beta2,
        minimum_approach_speed=minimum_approach_speed,
    )
    jerk_magnitude = _norm(_jerk(acceleration, dt))
    terms = {
        "velocity": velocity_huber_loss(
            velocity,
            target_velocity,
            window=velocity_window,
            delta=huber_delta,
            norm_weight=velocity_norm_weight,
            component_weight=velocity_component_weight,
        ),
        "clearance": barrier,
        "collision": collision,
        "acceleration": jnp.mean(jnp.sum(acceleration**2, axis=-1)),
        "jerk_mean": jnp.mean(jerk_magnitude),
        "jerk_variance": jnp.mean(jnp.var(jerk_magnitude, axis=0)),
    }
    terms["avoidance"] = barrier + beta1 * collision
    terms["jerk"] = (
        jerk_mean_weight * terms["jerk_mean"] + jerk_variance_weight * terms["jerk_variance"]
    )
    total = (
        velocity_weight * terms["velocity"]
        + collision_weight * terms["avoidance"]
        + acceleration_weight * terms["acceleration"]
        + jerk_weight * terms["jerk"]
    )
    return total, terms


@jax.custom_jvp
def _decay_identity(value: jax.Array, factor: jax.Array) -> jax.Array:
    return value


@_decay_identity.defjvp
def _decay_identity_jvp(primals, tangents):
    value, factor = primals
    value_tangent, _ = tangents
    return value, factor * value_tangent


def temporal_gradient_decay(state, *, alpha: float, dt: float):
    """Exact forward identity on a state PyTree, with Jacobian exp(-alpha*dt)*I.

    Use ``action = policy(state); next_state = step(temporal_gradient_decay(state,
    alpha=alpha, dt=dt), action)``. This scales the dynamics' partial derivative
    with respect to incoming state while leaving its direct action derivative
    intact. Never wrap next_state: doing so would also damp direct action gradients.
    Floating leaves are transformed; integer/bool bookkeeping leaves pass through.
    Alpha [1/s] and dt [s] are fixed hyperparameters with zero gradients.
    """
    factor = jnp.exp(-jnp.asarray(alpha) * dt)
    return jax.tree.map(
        lambda value: (
            _decay_identity(value, factor)
            if jnp.issubdtype(jnp.asarray(value).dtype, jnp.inexact)
            else value
        ),
        state,
    )
