"""Explicit navigation-domain adaptation of the recurrent point-cloud method.

The actor still receives only the paper point cloud and ten proprioceptive
fields. Geometry is used for physical evaluation, differentiable loss, and
collision-free initial-state sampling, never for a hidden route or controller.
"""

import jax
import jax.numpy as jnp

from drone_playground.environments.scenes.navigation import clearance_and_collision, euclidean_norm
from drone_playground.environments.tasks.pointcloud import PointCloudTask
from drone_playground.models.point_mass import PointMassState, acceleration_attitude


def _select(mask, new, old):
    return jax.tree.map(
        lambda a, b: jnp.where(mask.reshape(mask.shape + (1,) * (a.ndim - mask.ndim)), a, b),
        new,
        old,
    )


class PointCloudNavigationTask(PointCloudTask):
    physics_engine = "PointMassLag JAX 500Hz with randomized transport delay; MuJoCo replay only"

    def __init__(self, config):
        super().__init__(config)
        self.settings = config["env"]["task"]
        self.physics_dt = 1 / self.physics_freq
        self.substeps = self.physics_freq // self.freq
        self.bank, self.manifest = self.scene.build()
        self.physics_engine = type(self).physics_engine

    def select_bank(self, indices):
        names = (
            "kind",
            "size",
            "origin",
            "motion",
            "params",
            "active",
            "start",
            "goal",
            "difficulty",
            "subtype",
            "rotations",
        )
        return self.bank.replace(
            **{
                name: getattr(self.bank, name)[indices]
                for name in names
                if getattr(self.bank, name) is not None
            }
        )

    def clearance(self, bank, state, time):
        centre = state.pos + jnp.einsum("...ij,j->...i", state.rotation, jnp.array([0, 0, 0.005]))
        return jax.vmap(
            lambda i, p, t: clearance_and_collision(bank, i, t, p, self.body_radius)[0]
        )(jnp.arange(bank.num_instances), centre, jnp.broadcast_to(time, (bank.num_instances,)))

    def observation(self, bank, state, time, speeds):
        points, valid = jax.vmap(lambda i, p, r, t: self.sensor.sample(bank, i, p, r, t))(
            jnp.arange(bank.num_instances),
            state.pos,
            state.rotation,
            jnp.broadcast_to(time, (bank.num_instances,)),
        )
        proprio, target = self.observer.proprioception(state, bank.goal, speeds, self.body_radius)
        return points, valid, proprio, target

    def delays(self, key, count):
        low, high = self.config["runtime"]["action_delay_ms"]
        milliseconds = jax.random.uniform(key, (count,), minval=low, maxval=high)
        return jnp.ceil(milliseconds / (1000 * self.physics_dt) - 1e-6).astype(jnp.int32)

    def training_initial(self, key, count):
        ik, pk, tk, sk, vk, dk = jax.random.split(key, 6)
        ids = jax.random.randint(ik, (count,), 0, self.bank.num_instances)
        bank = self.select_bank(ids)
        clocks = jax.random.uniform(tk, (count,), maxval=self.duration)
        clocks = clocks.at[: count // 4].set(0.0)
        n = int(self.settings["training_position_candidates"])
        unit = jax.random.uniform(pk, (count, n, 3))
        low, high = (
            bank.world_low + jnp.array([2, 1, 0.5]),
            bank.world_high - jnp.array([2, 1, 0.5]),
        )
        proposals = low + unit * (high - low)
        # Include departure and near-goal states in every batch. Remaining
        # states cover the free course; no global path is computed or supplied.
        proposals = proposals.at[: count // 4].set(bank.start[: count // 4, None, :])
        near = bank.goal[:, None, :] + (unit - 0.5) * jnp.array([4.0, 4.0, 1.0])
        near = jnp.clip(near, low, high)
        proposals = proposals.at[count // 4 : count // 2].set(near[count // 4 : count // 2])
        clearance = jax.vmap(
            lambda p: self.clearance(bank, PointMassState.create(p), clocks), in_axes=1, out_axes=1
        )(proposals)
        safe = clearance >= self.settings["initial_clearance_m"]
        first = jnp.argmax(safe, axis=1)
        pos = proposals[jnp.arange(count), first]
        # The finite proposal batch has a safe departure fallback. The training
        # metric independently checks all actual sampled states before updating.
        found = jnp.any(safe, axis=1)
        pos = jnp.where(found[:, None], pos, bank.start)
        clocks = jnp.where(found, clocks, 0.0)
        speeds = jax.random.uniform(
            sk,
            (count,),
            minval=self.settings["command_speed_range"][0],
            maxval=self.settings["command_speed_range"][1],
        )
        delta = bank.goal - pos
        velocity = delta / jnp.maximum(euclidean_norm(delta), 1e-6)[:, None] * speeds[:, None]
        velocity += jax.random.normal(vk, (count, 3)) * self.settings["training_velocity_noise"]
        velocity = velocity.at[: count // 2].set(0.0)
        state = PointMassState.create(pos).replace(vel=velocity)
        rotation = acceleration_attitude(state.acc, state.vel, state.rotation)
        return bank, state.replace(rotation=rotation), clocks, speeds, self.delays(dk, count)

    def advance_checked(self, bank, state, command, previous, ticks, timestamp, outcome):
        """Same 500Hz integration as training, with exact first-event freezing."""
        initial_time = timestamp

        def step(carry, tick):
            physical, clock, result, minimum = carry
            active = result == 0
            due = jnp.where((tick < ticks)[:, None], previous, command)
            candidate = self.model.step(physical, due, self.physics_dt)
            now = initial_time + (tick + 1) * self.physics_dt
            finite = jnp.all(jnp.isfinite(candidate.vector()), axis=-1)
            clearance = self.clearance(bank, candidate, now)
            collision = clearance < 0
            outside = jnp.any(
                (candidate.pos < bank.world_low) | (candidate.pos > bank.world_high), -1
            )
            arrived = euclidean_norm(bank.goal - candidate.pos) <= self.goal_radius
            ended = jnp.where(
                ~finite, 4, jnp.where(collision, 2, jnp.where(outside, 3, jnp.where(arrived, 1, 0)))
            ).astype(jnp.int32)
            return (
                _select(active & finite, candidate, physical),
                jnp.where(active, now, clock),
                jnp.where(active, ended, result),
                jnp.where(active & finite, jnp.minimum(minimum, clearance), minimum),
            ), None

        end, _ = jax.lax.scan(
            step,
            (state, timestamp, outcome, jnp.full_like(timestamp, jnp.inf)),
            jnp.arange(self.substeps),
        )
        physical, clock, result, minimum = end
        return physical, clock, result, jnp.where(jnp.isfinite(minimum), minimum, 0.0)


def validate_navigation_adaptation(config):
    task, algo, settings = config["env"]["task"], config["algorithm"], config["training"]
    implementations = {
        "pointcloud_recurrent": "pointcloud_navigation_bptt",
        "depth_recurrent": "depth_navigation_bptt",
    }
    implementation = config["method"]["implementation"]
    if implementation not in implementations:
        raise ValueError("Navigation adaptation requires a qualified recurrent flight policy")
    if (
        config["method"]["output"],
        config["method"].get("network_output_frame"),
        config["method"].get("command_units"),
    ) != ("world_acceleration", "body", "m/s^2"):
        raise ValueError("The policy emits body-frame acceleration converted to world m/s^2")
    if config["env"]["execution"]["controller"]["name"] != "acceleration_passthrough":
        raise ValueError("Navigation adaptation cannot substitute an implicit controller")
    if (
        task["freq"],
        task["physics_freq"],
        task["duration"],
        task["goal_radius"],
        task["body_radius"],
    ) != (10, 500, 300, 0.5, 0.07):
        raise ValueError(
            "Navigation adaptation retains 10/500Hz, 300s, 0.5m arrival and 0.07m body"
        )
    if algo["name"] != implementations[implementation]:
        raise ValueError("Use the explicit navigation adaptation trainer")
    if config["env"]["execution"]["dynamics"]["forward"] != "point_mass_lag":
        raise ValueError("The qualified adapter uses the point-mass lag model")
    if config["env"]["sensor"]["source_rate_hz"] != task["freq"]:
        raise ValueError(
            "Recurrent policy and exteroceptive measurements must have matching clocks"
        )
    delay = config["runtime"].get("action_delay_ms")
    if (
        delay is None
        or delay[1] > 1000 / task["freq"]
        or config["runtime"].get("action_delay_steps", 0)
    ):
        raise ValueError("Choose millisecond transport delay within one policy interval")
    if min(settings["num_envs"], settings["policy_updates"], algo["horizon_length"]) < 1:
        raise ValueError("Positive training counts are required")
    expected = settings["num_envs"] * settings["policy_updates"] * algo["horizon_length"]
    if settings.get("num_timesteps") not in (None, expected):
        raise ValueError("Declared interaction and update budgets disagree")
    if settings.get("resume") and settings.get("warm_start"):
        raise ValueError("Choose exact continuation or parameter warm start")
    lower, upper = task["command_speed_range"]
    if not 0 < lower <= upper <= task["max_speed"] == 20.0:
        raise ValueError("Curriculum command speeds must fit within the 20m/s nominal maximum")
    if algo["gradient"]["transition"] not in ("direct", "exponential"):
        raise ValueError("Use the declared point-mass derivative contract")
