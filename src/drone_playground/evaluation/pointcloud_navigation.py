"""Actual first-event navigation evaluation of the explicitly trained adaptation."""

import copy
import time
from collections import Counter
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from hydra.utils import instantiate

from drone_playground.environments.scenes.navigation import euclidean_norm
from drone_playground.evaluation.pointcloud import export_case, summarize_trace
from drone_playground.evaluation.tracking import save_report, tree_digest


def navigation_development_selection(report):
    """Frozen 64-case lexicographic objective; exact ties keep the earlier snapshot.

    The integer scalar preserves worst-scene priority: one additional success in
    the worst scene outweighs every possible change in total successes (0..64).
    It is an optimization objective, not a success rate.
    """
    rows = report["episodes"]
    expected = {prefix + suffix for prefix in ("S", "D") for suffix in ("01", "02", "03", "06")}
    counts = Counter(row["scene_id"] for row in rows)
    seeds = [row["seed"] for row in rows]
    reset = report.get("initial_conditions")
    if (report["num_trials"] != 64 or len(rows) != 64
            or counts != dict.fromkeys(expected, 8) or len(set(seeds)) != 64
            or not reset or reset["seeds"] != seeds):
        raise ValueError("navigation-development-v2 requires eight independent resets per Navigation8 scene")
    successes = {scene: sum(bool(row["arrived"]) for row in rows if row["scene_id"] == scene)
                 for scene in expected}
    worst, total = min(successes.values()), sum(successes.values())
    return dict(score=[worst, total], pilot_objective=(65 * worst + total) / 584,
                worst_scene_success_rate=worst / 8, overall_success_rate=total / 64,
                selection_rule="navigation-development-v2")


class PointCloudNavigationEvaluator:
    def __init__(self, task, network, seed_start=20000, repeats=1, commanded_speed=4.0, initial_conditions=None):
        if repeats < 1 or not 0 < commanded_speed <= task.settings["max_speed"]:
            raise ValueError("Choose positive repeats and an admissible commanded cruise speed")
        self.task = copy.copy(task)
        indices = jnp.tile(jnp.arange(task.bank.num_instances), repeats)
        self.task.bank = task.select_bank(indices)
        self.ids = list(task.manifest["scene_ids"]) * repeats
        self.seeds = list(range(seed_start, seed_start + len(self.ids)))
        self.speed = commanded_speed
        self.repeats = repeats
        self.network = network
        self.initial_conditions = None
        self.initial_velocity = jnp.zeros((len(self.ids), 3))
        self.initial_rotation = jnp.broadcast_to(jnp.eye(3), (len(self.ids),3,3))
        if initial_conditions:
            from .navigation_resets import navigation_resets

            reset = navigation_resets(self.task.bank, np.arange(len(self.ids)), self.seeds,
                                      initial_conditions, task.body_radius)
            self.task.bank = self.task.bank.replace(start=jnp.asarray(reset['position']))
            self.initial_velocity = jnp.asarray(reset['velocity'])
            self.initial_rotation = jnp.asarray(reset['rotation'])
            self.initial_conditions = reset['record']
        self._run = jax.jit(self._rollout)

    def _rollout(self, params, speeds, delays):
        task, network, bank = self.task, self.network, self.task.bank
        count, length = bank.num_instances, task.episode_length
        physical = task.initial_state(bank).replace(vel=self.initial_velocity, rotation=self.initial_rotation)
        hidden = jnp.zeros((count, network.hidden_size))
        last = jnp.zeros((count, 3))
        clock = jnp.zeros(count)
        outcome = jnp.zeros(count, jnp.int32)

        def row(old, new, proprio, command, active, minimum, index):
            state, _, _, timestamp, result = new
            return dict(
                pos=state.pos,
                velocity=state.vel,
                rotation=state.rotation,
                observation_pos=old[0].pos,
                observation_rotation=old[0].rotation,
                observation_time=jnp.full((count,), index * task.dt),
                obs=proprio,
                time=timestamp,
                actions=jnp.where(active[:, None], command, 0.0),
                reward=jnp.where(
                    active,
                    euclidean_norm(bank.goal - old[0].pos) - euclidean_norm(bank.goal - state.pos),
                    0.0,
                ),
                active=active,
                done=result != 0,
                outcome=result,
                metrics=dict(
                    clearance=minimum,
                    goal_distance=euclidean_norm(bank.goal - state.pos),
                    speed=euclidean_norm(state.vel),
                ),
            )

        initial = (physical, hidden, last, clock, outcome)
        proprio, _ = task.observer.proprioception(physical, bank.goal, speeds, task.body_radius)
        template = row(initial, initial, proprio, last, jnp.zeros(count, bool), jnp.zeros(count), 0)
        archive = jax.tree.map(lambda v: jnp.zeros((length, *v.shape), v.dtype), template)

        def advance(carry):
            current, index, buffers = carry
            state, memory, previous, timestamp, result = current
            active = result == 0
            points, valid, proprio, _ = task.observation(bank, state, timestamp, speeds)
            action, next_memory = network.apply(params, points, valid, proprio, memory)
            command = task.command(action, state)
            nxt, now, status, minimum = task.advance_checked(
                bank, state, command, previous, delays, timestamp, result
            )
            status = jnp.where((index == length - 1) & (status == 0), 5, status)
            end = (
                nxt,
                jnp.where(active[:, None], next_memory, memory),
                jnp.where(active[:, None], command, previous),
                now,
                status,
            )
            record = row(current, end, proprio, command, active, minimum, index)
            buffers = jax.tree.map(
                lambda storage, value: storage.at[index].set(value), buffers, record
            )
            return end, index + 1, buffers

        final, stop, archive = jax.lax.while_loop(
            lambda carry: (carry[1] < length) & jnp.any(carry[0][4] == 0),
            advance,
            (initial, jnp.int32(0), archive),
        )
        proprio, _ = task.observer.proprioception(final[0], bank.goal, speeds, task.body_radius)
        padding = row(final, final, proprio, last, jnp.zeros(count, bool), jnp.zeros(count), 0)
        padding = jax.tree.map(
            lambda value: jnp.broadcast_to(value, (length, *value.shape)), padding
        )
        padding["observation_time"] = jnp.broadcast_to(
            jnp.arange(length)[:, None] * task.dt, (length, count)
        )
        return jax.tree.map(
            lambda a, b: jnp.where(
                (jnp.arange(length) < stop).reshape((length,) + (1,) * (a.ndim - 1)), a, b
            ),
            archive,
            padding,
        )

    def run(self, parameters, *, commanded_speed=None, delay_ticks=None):
        speed = self.speed if commanded_speed is None else float(commanded_speed)
        if not np.isfinite(speed) or not 0 < speed <= self.task.settings["max_speed"]:
            raise ValueError("Commanded speed must be positive and within the nominal maximum")
        count = self.task.bank.num_instances
        explicit = delay_ticks is not None
        if delay_ticks is None:
            delays = jax.vmap(lambda seed: self.task.delays(jax.random.PRNGKey(seed), 1)[0])(
                jnp.asarray(self.seeds, jnp.int32)
            )
        else:
            values = np.asarray(delay_ticks)
            low, high = self.task.config["runtime"]["action_delay_ms"]
            bounds = np.ceil(np.array([low, high]) / (1000 * self.task.physics_dt) - 1e-6)
            if (
                values.shape != (count,)
                or not np.issubdtype(values.dtype, np.integer)
                or np.any(values < bounds[0])
                or np.any(values > bounds[1])
            ):
                raise ValueError(
                    "Explicit delay ticks must match the batch and declared physical-delay range"
                )
            delays = jnp.asarray(values, jnp.int32)
        before = tree_digest(parameters)
        started = time.monotonic()
        trace = jax.tree.map(
            np.asarray, self._run(parameters, jnp.full((count,), speed, jnp.float32), delays)
        )
        report = summarize_trace(trace, self.ids, speed, self.task.duration, self.task.bank.start)
        rates = {
            name: np.mean([r["arrived"] for r in report["episodes"] if r["scene_id"] == name])
            for name in sorted(set(self.ids))
        }
        after = tree_digest(parameters)
        if before != after:
            raise RuntimeError("Frozen navigation evaluation modified the policy")
        report.update(
            scene_success_rates={name: float(value) for name, value in rates.items()},
            collision_rate=report["collision"] / report["num_trials"],
            failure_rate=(
                report["collision"] + report["out_of_bounds"] + report["numerical_failure"]
            )
            / report["num_trials"],
            parameters_frozen=True,
            parameter_sha256=before,
            reset_seeds=self.seeds,
            quality_passed=bool(report["success_rate"] >= 0.9 and min(rates.values()) >= 0.8),
            quality_rule="mean arrival >=0.9 and each named scene >=0.8; evaluate independent seeds after selection",
            elapsed_seconds=time.monotonic() - started,
            catalog_sha256=self.task.manifest["catalog_sha256"],
            evaluation_scope=("fixed Navigation8 geometry; independently seeded initial pose/velocity and transport delays"
                              if self.initial_conditions else
                              "fixed geometry and start pose; independent seeded transport delays only"),
            initial_conditions=self.initial_conditions,
            dynamics_transition_hz=self.task.physics_freq,
            policy_hz=self.task.freq,
            action_delay_ms=self.task.config["runtime"]["action_delay_ms"],
            nominal_max_speed_m_s=self.task.settings["max_speed"],
            delay_ticks=np.asarray(delays).tolist(),
            delay_source="explicit development grid"
            if explicit
            else "seeded uniform milliseconds rounded to physics ticks",
        )
        for index, episode in enumerate(report['episodes']):
            episode.update(seed=self.seeds[index], delay_ticks=int(delays[index]),
                           initial_position_m=np.asarray(self.task.bank.start[index]).tolist(),
                           initial_velocity_mps=np.asarray(self.initial_velocity[index]).tolist())
        return report, trace

    def export(self, trace, directory, all_repeats=False):
        paths = []
        seen = set()
        for case, name in enumerate(self.ids):
            if not all_repeats and name in seen:
                continue
            seen.add(name)
            paths.append(
                str(export_case(self.task, trace, case, Path(directory) / f"{case:03d}-{name}"))
            )
        return paths


def evaluate(config, root, run_id):
    from drone_playground.composition import build_environment
    from drone_playground.runs.pointcloud import load_training_state
    from drone_playground.runs.record import RunRecorder

    state, metadata = load_training_state(config["checkpoint"])
    trained = metadata["config"]
    if trained["env"]["task"]["name"] not in ("pointcloud_navigation", "depth_navigation"):
        raise ValueError(
            "Use parameter warm start to adapt a source-paper checkpoint, not a relabeled evaluation"
        )
    cfg = copy.deepcopy(trained)
    cfg["mode"] = "eval"
    cfg["evaluation"] = copy.deepcopy(config["evaluation"])
    cfg["runtime"]["device"] = config["runtime"]["device"]
    task = build_environment(cfg, cfg["runtime"]["device"])
    network = instantiate(cfg["network"], _convert_="all")
    start = cfg["evaluation"].get("seed_start")
    start = (
        start if start is not None else (20000 if cfg["evaluation"]["split"] == "dev" else 30000)
    )
    evaluator = PointCloudNavigationEvaluator(
        task, network, start, cfg["evaluation"]["episodes"], cfg["evaluation"]["commanded_speed"],
        cfg["evaluation"].get("initial_conditions")
    )
    with RunRecorder(
        root, run_id, cfg, task_id="navigation-convergence/pointcloud-evaluation"
    ) as rec:
        report, trace = evaluator.run(state.params)
        report.update(
            checkpoint=str(Path(config["checkpoint"]).resolve()),
            trained_adaptation_updates=int(state.updates),
            split=cfg["evaluation"]["split"],
        )
        save_report(rec.path / "eval/report.json", report)
        if cfg["evaluation"].get("release_validation"):
            from .navigation_resets import validate_navigation_report

            if cfg["evaluation"]["release_validation"] != "navigation-v1":
                raise ValueError("Unknown navigation release validation")
            release = validate_navigation_report(report)
            save_report(rec.path / "eval/release-validation.json", release)
            report["quality_passed"] = release["passed"]
            report["quality_rule"] = "release-navigation-v1: >=100 each static/dynamic; each task arrival >=90%"
            save_report(rec.path / "eval/report.json", report)
        arrays = {name: value for name, value in trace.items() if name != "metrics"}
        arrays.update({"metric_" + name: value for name, value in trace["metrics"].items()})
        (rec.path / "traces").mkdir(exist_ok=True)
        np.savez_compressed(rec.path / "traces/navigation.npz", **arrays)
        paths = evaluator.export(trace, rec.path / "rollouts", all_repeats=True)
        save_report(rec.path / "rollouts/index.json", {"paths": paths})
        rec.finish(
            "completed",
            quality_passed=report["quality_passed"],
            actual_trials=report["num_trials"],
            arrived=report["arrived"],
            parameters_frozen=True,
        )
    task.close()
    return report
