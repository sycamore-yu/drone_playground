"""Task references and first-event episode semantics in SI units."""

from enum import IntEnum

import jax.numpy as jnp
import numpy as np
from flax import struct
from jax import Array
from scipy.interpolate import CubicSpline


class Event(IntEnum):
    """The first terminal event is retained until an explicit reset."""

    RUNNING = 0
    COLLISION = 1
    NUMERICAL_FAILURE = 2
    OUT_OF_BOUNDS = 3
    SUCCESS = 4
    TIMEOUT = 5
    MISSED_GATE = 6
    METHOD_FAILURE = 7


@struct.dataclass
class TaskState:
    """Per-world progress and error integrals for one independent episode."""

    event: Array
    gates: Array
    error_sum: Array
    samples: Array
    min_clearance: Array


def first_event(collision, finite, in_bounds, success, timeout):
    """Apply the Navigation event priority, including simultaneous events."""
    return jnp.select(
        [collision, ~finite, ~in_bounds, success, timeout],
        [
            Event.COLLISION,
            Event.NUMERICAL_FAILURE,
            Event.OUT_OF_BOUNDS,
            Event.SUCCESS,
            Event.TIMEOUT,
        ],
        default=Event.RUNNING,
    ).astype(jnp.int32)


class Task:
    """Task-specific references and rules, independent of the flight method."""

    def __init__(
        self,
        name: str,
        scene,
        reference: str = "figure_eight",
        duration: float | None = None,
        reference_seed: int = 0,
        gate_order=None,
    ):
        """Configure the task reference, duration and terminal-event contract."""
        if name not in {"tracking", "racing", "navigation"}:
            raise ValueError(f"Unknown task: {name}")
        self.name = name
        self.reference = reference
        self.gate_order = jnp.zeros(0, jnp.int32)
        self.radius = 0.07
        self.goal_radius = 0.5
        self.duration = (
            {"tracking": 20.0, "racing": 60.0, "navigation": 300.0}[name]
            if duration is None
            else duration
        )
        if not np.isfinite(self.duration) or self.duration <= 0:
            raise ValueError("Episode duration must be finite and positive")
        self.start = jnp.array([0.0, 0.0, 1.5])
        self.goal = self.start
        self.low = jnp.array([-5.0, -5.0, 0.07])
        self.high = jnp.array([5.0, 5.0, 5.0])
        if name == "navigation":
            self.start, self.goal = jnp.array([2.0, 0.0, 3.0]), jnp.array([98.0, 0.0, 3.0])
            self.low, self.high = jnp.array([0.0, -20.0, 0.5]), jnp.array([100.0, 20.0, 6.0])
        elif name == "racing":
            self.start = jnp.array([-1.5, 0.75, 0.5])
            self.low, self.high = jnp.array([-2.5, -1.5, 0.07]), jnp.array([2.5, 1.5, 2.0])
            self.gate_positions, self.gate_rotations = scene.gate_positions, scene.gate_rotations
            order = np.asarray(scene.gate_order if gate_order is None else gate_order, int) - 1
            if len(order) == 0 or np.any(order < 0) or np.any(order >= len(self.gate_positions)):
                raise ValueError("Racing needs a nonempty order referencing existing scene gates")
            self.gate_order = jnp.asarray(order)
            detours = np.asarray(scene.numeric.get("reference_detours", [])).reshape(-1, 4)
            speed = float(scene.numeric.get("reference_speed", [0.5])[0])
            spacing = float(scene.numeric.get("reference_spacing", [0.3])[0])
            if speed <= 0 or spacing <= 0:
                raise ValueError("Racing reference speed and gate spacing must be positive")
            points = [np.asarray(self.start)]
            for i in order:
                center = np.asarray(self.gate_positions[i])
                normal = np.asarray(self.gate_rotations[i, :, 0])
                points.extend([center - spacing * normal, center, center + spacing * normal])
                points.extend(detours[detours[:, 0] == i + 1, 1:])
            points = np.array(points)
            times = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1)) / speed]
            self._set_spline(times, points)
            self.goal = jnp.asarray(points[-1])
        elif reference == "spline":
            rng = np.random.default_rng(reference_seed)
            points = rng.uniform([-1.0, -1.0, 1.0], [1.0, 1.0, 2.0], (9, 3))
            points[0] = np.asarray(self.start)
            self._set_spline(np.linspace(0, self.duration, 9), points)
        elif reference not in {"hover", "figure_eight"}:
            raise ValueError(f"Unknown tracking reference: {reference}")

    @property
    def contract(self) -> dict:
        """Return the instantiated task rules for durable evaluation records."""
        result = {
            "name": self.name,
            "duration_seconds": float(self.duration),
            "body_radius_metres": self.radius,
            "start_metres": np.asarray(self.start).tolist(),
            "bounds_metres": [np.asarray(self.low).tolist(), np.asarray(self.high).tolist()],
            "event_priority": [
                "COLLISION",
                "NUMERICAL_FAILURE",
                "OUT_OF_BOUNDS",
                "SUCCESS",
                "TIMEOUT",
            ],
        }
        if self.name == "navigation":
            result.update(
                goal_metres=np.asarray(self.goal).tolist(), goal_radius_metres=self.goal_radius
            )
        elif self.name == "racing":
            result["gate_order"] = (np.asarray(self.gate_order) + 1).tolist()
        else:
            result["reference"] = self.reference
        return result

    def _set_spline(self, times, points):
        spline = CubicSpline(times, points, bc_type=((1, np.zeros(3)), (1, np.zeros(3))))
        self.knots, self.coefficients = jnp.asarray(times), jnp.asarray(spline.c)

    def target(self, time: Array) -> tuple[Array, Array, Array]:
        """Return world position, velocity and acceleration at absolute episode time."""
        time = jnp.asarray(time)
        if self.name == "navigation":
            goal = jnp.broadcast_to(self.goal, (*time.shape, 3))
            return goal, jnp.zeros_like(goal), jnp.zeros_like(goal)
        if self.name == "racing" or self.reference == "spline":
            t = jnp.clip(time, self.knots[0], self.knots[-1])
            i = jnp.clip(jnp.searchsorted(self.knots, t, side="right") - 1, 0, len(self.knots) - 2)
            d = (t - self.knots[i])[..., None]
            a, b, c, e = self.coefficients[:, i]
            p, v, acc = (
                ((a * d + b) * d + c) * d + e,
                (3 * a * d + 2 * b) * d + c,
                6 * a * d + 2 * b,
            )
            return p, jnp.where((time < self.knots[-1])[..., None], v, 0.0), acc
        if self.reference == "hover":
            p = jnp.broadcast_to(self.start, (*time.shape, 3))
            return p, jnp.zeros_like(p), jnp.zeros_like(p)
        w = 2 * jnp.pi / self.duration
        phase = w * time
        z = jnp.zeros_like(time)
        p = self.start + jnp.stack([jnp.sin(phase), 0.5 * jnp.sin(2 * phase), z], -1)
        v = jnp.stack([w * jnp.cos(phase), w * jnp.cos(2 * phase), z], -1)
        a = jnp.stack([-(w**2) * jnp.sin(phase), -2 * w**2 * jnp.sin(2 * phase), z], -1)
        return p, v, a

    def initial(self, batch: int) -> TaskState:
        """Create fresh counters for a batch of independently initialized episodes."""
        return TaskState(
            jnp.zeros(batch, jnp.int32),
            jnp.zeros(batch, jnp.int32),
            jnp.zeros(batch),
            jnp.zeros(batch, jnp.int32),
            jnp.full(batch, jnp.inf),
        )

    def update(self, state: TaskState, before, after, time, clearance) -> TaskState:
        """Judge every physics substep, retaining the first event and gate crossings."""
        p = after.pos[:, 0]
        finite = jnp.all(
            jnp.isfinite(
                jnp.concatenate([p, after.vel[:, 0], after.quat[:, 0], after.ang_vel[:, 0]], -1)
            ),
            -1,
        )
        collision = (clearance <= self.radius) | (p[:, 2] <= self.radius)
        in_bounds = jnp.all((p >= self.low) & (p <= self.high), -1)
        gates = state.gates
        missed = jnp.zeros_like(finite)
        if self.name == "racing":
            total_gates = len(self.gate_order)
            gate = self.gate_order[jnp.minimum(gates, total_gates - 1)]
            center, rot = self.gate_positions[gate], self.gate_rotations[gate]
            prev = jnp.einsum("bij,bj->bi", jnp.swapaxes(rot, -1, -2), before.pos[:, 0] - center)
            curr = jnp.einsum("bij,bj->bi", jnp.swapaxes(rot, -1, -2), p - center)
            crossed = (prev[:, 0] < 0) & (curr[:, 0] >= 0) & (gates < total_gates)
            fraction = -prev[:, 0] / jnp.maximum(curr[:, 0] - prev[:, 0], 1e-8)
            hit = prev + fraction[:, None] * (curr - prev)
            aperture = jnp.all(jnp.abs(hit[:, 1:]) <= 0.2 - self.radius, -1)
            gates = gates + (crossed & aperture).astype(jnp.int32)
            # A far-away plane crossing is not a gate attempt.
            missed = crossed & ~aperture & jnp.all(jnp.abs(hit[:, 1:]) < 0.36, -1)
            success = gates == total_gates
        elif self.name == "navigation":
            success = jnp.linalg.norm(p - self.goal, axis=-1) <= self.goal_radius
        else:
            success = time >= self.duration - 1e-5
        event = first_event(collision, finite, in_bounds, success, time >= self.duration - 1e-5)
        event = jnp.where((event == Event.RUNNING) & missed, Event.MISSED_GATE, event)
        active = state.event == Event.RUNNING
        target, _, _ = self.target(time)
        error = jnp.sum((p - target) ** 2, -1)
        return TaskState(
            event=jnp.where(active, event, state.event),
            gates=jnp.where(active, gates, state.gates),
            error_sum=state.error_sum + jnp.where(active, error, 0.0),
            samples=state.samples + active.astype(jnp.int32),
            min_clearance=jnp.where(
                active,
                jnp.minimum(state.min_clearance, clearance - self.radius),
                state.min_clearance,
            ),
        )
