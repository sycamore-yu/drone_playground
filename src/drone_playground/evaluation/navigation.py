"""Independent navigation rollouts with per-difficulty cells and full failure accounting.

Every evaluation cell is one (unit, difficulty) pair with ``per_difficulty``
episodes taken from the frozen scene bank slice for that difficulty. Failed,
collided, out-of-bounds and invalid episodes stay in the denominator, and each
episode keeps its scenario identity so a result can be traced back to geometry.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from drone_playground.tasks.navigation import (
    OUTCOME_COLLISION,
    OUTCOME_NAMES,
    OUTCOME_TIMEOUT,
)
from drone_playground.tasks.scenes.navigation import DIFFICULTIES


def summarize_cell(trace: dict, labels: list[dict], dt: float, duration: float) -> dict:
    """Aggregate one (unit, difficulty) cell, counting every episode exactly once."""
    done = np.asarray(trace["done"]).astype(bool)
    outcome = np.asarray(trace["outcome"]).astype(np.int32)
    steps_total, batch = done.shape
    any_done = done.any(axis=0)
    first = np.argmax(done, axis=0)
    rows = np.arange(batch)
    steps = np.where(any_done, first + 1, steps_total)
    final_outcome = np.where(any_done, outcome[first, rows], OUTCOME_TIMEOUT)

    arrived = final_outcome == 1
    collision = final_outcome == OUTCOME_COLLISION
    out_of_bounds = final_outcome == 3
    numerical = final_outcome == 4
    timeout = ~any_done

    times = steps * dt
    constrained = float(np.where(arrived, times, duration).mean())
    success_times = times[arrived]
    clearance = np.asarray(trace["metrics"]["clearance"])
    goal_distance = np.asarray(trace["metrics"]["goal_distance"])
    returns = np.where(np.asarray(trace["active"], dtype=bool), trace["reward"], 0).sum(axis=0)

    episodes = []
    for index in range(batch):
        episodes.append(
            {
                "case": index,
                **labels[index],
                "outcome": OUTCOME_NAMES[int(final_outcome[index])],
                "arrived": bool(arrived[index]),
                "collision": bool(collision[index]),
                "out_of_bounds": bool(out_of_bounds[index]),
                "numerical_failure": bool(numerical[index]),
                "timeout": bool(timeout[index]),
                "steps": int(steps[index]),
                "arrival_time_s": float(times[index]) if arrived[index] else None,
                "min_clearance_m": float(clearance[: steps[index], index].min()),
                "final_goal_distance_m": float(goal_distance[steps[index] - 1, index]),
                "return": float(returns[index]),
            }
        )
    return {
        "num_trials": int(batch),
        "arrived": int(arrived.sum()),
        "collision": int(collision.sum()),
        "out_of_bounds": int(out_of_bounds.sum()),
        "numerical_failure": int(numerical.sum()),
        "timeout": int(timeout.sum()),
        "success_rate": float(arrived.mean()),
        "collision_rate": float(collision.mean()),
        "failure_rate": float((collision | out_of_bounds | numerical).mean()),
        "timeout_rate": float(timeout.mean()),
        "constrained_time_mean_s": constrained,
        "success_time_mean_s": float(success_times.mean()) if arrived.any() else None,
        "success_time_samples": int(arrived.sum()),
        "min_clearance_m": float(clearance.min()),
        "return_mean": float(returns.mean()),
        "episodes": episodes,
    }


def combine_cells(cells: dict[str, dict]) -> dict:
    """Macro-average over difficulties, the protocol's primary selection quantity."""
    totals = sum(cell["num_trials"] for cell in cells.values())
    macro_success = float(np.mean([cell["success_rate"] for cell in cells.values()]))
    macro_collision = float(np.mean([cell["collision_rate"] for cell in cells.values()]))
    macro_constrained = float(
        np.mean([cell["constrained_time_mean_s"] for cell in cells.values()])
    )
    valid = [cell["success_time_mean_s"] for cell in cells.values() if cell["success_time_mean_s"]]
    return {
        "num_trials": totals,
        "arrived": sum(cell["arrived"] for cell in cells.values()),
        "collision": sum(cell["collision"] for cell in cells.values()),
        "out_of_bounds": sum(cell["out_of_bounds"] for cell in cells.values()),
        "numerical_failure": sum(cell["numerical_failure"] for cell in cells.values()),
        "timeout": sum(cell["timeout"] for cell in cells.values()),
        "success_rate": macro_success,
        "collision_rate": macro_collision,
        "failure_rate": float(np.mean([cell["failure_rate"] for cell in cells.values()])),
        "timeout_rate": float(np.mean([cell["timeout_rate"] for cell in cells.values()])),
        "constrained_time_mean_s": macro_constrained,
        "success_time_mean_s": float(np.mean(valid)) if valid else None,
        "return_mean": float(np.mean([cell["return_mean"] for cell in cells.values()])),
        "cells": cells,
    }


class NavigationEvaluator:
    """Reuse one compiled rollout per difficulty for every snapshot of a run."""

    def __init__(self, env, make_policy, reset_seeds: list[int]):
        self.env = env
        self.reset_seeds = list(reset_seeds)
        per_difficulty = env.bank.num_instances // len(DIFFICULTIES)
        if per_difficulty < 1:
            raise ValueError("The evaluation bank has no per-difficulty instances")
        seeds = jnp.asarray([jax.random.PRNGKey(seed) for seed in self.reset_seeds])
        self.groups: dict[str, list[int]] = {}
        self._run = {}
        for index, difficulty in enumerate(DIFFICULTIES):
            scenario_ids = jnp.asarray(
                np.arange(index * per_difficulty, (index + 1) * per_difficulty), jnp.int32
            )
            self.groups[difficulty] = [int(value) for value in scenario_ids]
            keys = seeds[: per_difficulty] if len(seeds) >= per_difficulty else jnp.tile(
                seeds, (per_difficulty // len(seeds) + 1, 1)
            )[:per_difficulty]
            self._run[difficulty] = jax.jit(self._make_rollout(make_policy, keys, scenario_ids))

    def _make_rollout(self, make_policy, keys, scenario_ids):
        env = self.env
        length = env.episode_length

        def run(params):
            policy = make_policy(params, deterministic=True)
            state = jax.vmap(env.reset)(keys, scenario_ids)

            def one_step(carry, index):
                current, alive = carry
                action = policy(current.obs, jax.random.PRNGKey(0))[0]
                proposed = jax.vmap(env.step)(current, action)

                def freeze(old, new):
                    mask = alive.reshape(alive.shape + (1,) * (new.ndim - alive.ndim))
                    return jnp.where(mask, new, old)

                proposed = jax.tree.map(freeze, current, proposed)
                failed = (~alive) | proposed.done.astype(bool)
                trace = {
                    "pos": proposed.pipeline_state.sim_data.states.pos[:, 0, 0],
                    "quat": proposed.pipeline_state.sim_data.states.quat[:, 0, 0],
                    "obs": proposed.obs,
                    "time": jnp.full((keys.shape[0],), (index + 1) * env.dt),
                    "actions": action,
                    "reward": proposed.reward,
                    "metrics": proposed.metrics,
                    "active": alive,
                    "failed": failed,
                    "done": proposed.done,
                    "outcome": proposed.info["outcome"],
                }
                return (proposed, ~failed), trace

            _, trace = jax.lax.scan(
                one_step, (state, jnp.ones(keys.shape[0], dtype=bool)), jnp.arange(length)
            )
            return trace

        return run

    def labels(self) -> dict[str, list[dict]]:
        labels = {}
        for difficulty, scenario_ids in self.groups.items():
            labels[difficulty] = [
                {"scenario_id": scenario_id, **self.env.bank.labels(scenario_id)}
                for scenario_id in scenario_ids
            ]
        return labels

    def run(self, params, quality_threshold: float | None = None):
        from drone_playground.evaluation.tracking import tree_digest

        before = tree_digest(params)
        traces = {}
        labels = self.labels()
        for difficulty in DIFFICULTIES:
            trace = jax.tree.map(np.asarray, self._run[difficulty](params))
            traces[difficulty] = trace
        after = tree_digest(params)
        if before != after:
            raise RuntimeError("Evaluation changed policy or normalization parameters")
        cells = {
            difficulty: summarize_cell(
                traces[difficulty], labels[difficulty], self.env.dt, self.env.duration
            )
            for difficulty in DIFFICULTIES
        }
        report = combine_cells(cells)
        report.update(
            parameter_sha256=before,
            parameters_frozen=True,
            task=self.env.task,
            dynamics=self.env.dynamics,
            drone=self.env.drone,
            scene_bank_sha256=_bank_digest(self.env),
            scenario_groups={key: value for key, value in self.groups.items()},
        )
        if quality_threshold is None:
            report["quality_passed"] = None
            report["quality_rule"] = (
                "pending the P5 protocol freeze; macro success rate, collision rate and "
                "constrained completion time are reported per difficulty"
            )
        else:
            report["quality_passed"] = bool(report["success_rate"] >= quality_threshold)
            report["quality_rule"] = f"macro success rate >= {quality_threshold}"
        return report, traces


def _bank_digest(env) -> str:
    return env.bank.digest()


def select_episodes(report: dict, count: int = 4) -> dict[str, list[int]]:
    """One replay per difficulty: the first failure if any, otherwise the first case."""
    selection = {}
    for difficulty, cell in report["cells"].items():
        episodes = cell["episodes"]
        failed = [row["case"] for row in episodes if not row["arrived"]]
        fixed = list(range(min(count, len(episodes))))
        chosen = sorted({*fixed, *(failed[:1] or [episodes[0]["case"]])})
        selection[difficulty] = chosen
    return selection


def export_navigation_replays(env, traces: dict, directory: Path, count: int = 4,
                              case_indices: dict | None = None) -> list[dict]:
    """Write one self-contained rscope replay per difficulty cell."""
    from drone_playground.runs.navigation_scene import (
        active_indices,
        create_replay_model,
        obstacle_track,
    )
    from drone_playground.runs.rscope_io import export_rollout

    per_difficulty = env.bank.num_instances // len(DIFFICULTIES)
    published = []
    for index, difficulty in enumerate(DIFFICULTIES):
        if difficulty not in traces:
            continue
        trace = traces[difficulty]
        cases = (case_indices[difficulty] if case_indices is not None
                 else list(range(min(count, trace["pos"].shape[1]))))
        for case in cases:
            scenario_id = index * per_difficulty + case
            stop = trace["pos"].shape[0]
            if "active" in trace:
                live = np.flatnonzero(np.asarray(trace["active"])[:, case])
                if not len(live):
                    raise ValueError(f"Replay case {case} has no active transition")
                # Include the terminal transition; omit the batch rollout's
                # padding so dynamic geometry cannot move after the episode.
                stop = int(live[-1]) + 1
            single = jax.tree.map(lambda value: np.asarray(value)[:stop, case : case + 1], trace)
            times = np.asarray(single["time"])[:, 0]
            active = active_indices(env.bank, scenario_id)
            single["obstacle_pos"] = obstacle_track(env.bank, scenario_id, times)[:, active][
                :, None
            ]
            single = {
                key: value
                for key, value in single.items()
                if key not in ("done", "outcome", "active", "failed")
            }
            target = directory / difficulty / f"case-{case:03d}"
            replay_model = create_replay_model(env, scenario_id)
            path = export_rollout(replay_model, target, single)
            published.append(
                {
                    "difficulty": difficulty,
                    "case": case,
                    "scenario_id": scenario_id,
                    "frames": stop,
                    "replay": str(path.relative_to(directory)),
                }
            )
    return published


def evaluate_navigation(config: dict, root: Path, run_id: str):
    """Independent evaluation entry for a frozen navigation policy."""
    import time

    from drone_playground.composition import build_environment
    from drone_playground.policies.neural import NeuralPolicy
    from drone_playground.runs.console import capture_console
    from drone_playground.runs.record import RunRecorder

    from .execution import resolve_evaluation_config
    from .tracking import save_report

    policy = NeuralPolicy.load(config["checkpoint"])
    maker, params, metadata = policy.make_policy, policy.parameters, policy.metadata
    resolved = resolve_evaluation_config(config, metadata)
    split = resolved["evaluation"]["split"]
    per_difficulty = int(resolved["evaluation"]["episodes"])
    rec = RunRecorder(root, run_id, resolved, task_id="p5-navigation-evaluation")
    env = None
    with capture_console(rec.path / "console.log"):
        try:
            rec.phase("initializing")
            start = resolved["evaluation"].get("seed_start")
            start = start if start is not None else (20000 if split == "dev" else 30000)
            env = build_environment(
                resolved, resolved["training"]["device"], split, per_difficulty
            )
            if (
                env.observation_size != metadata["observation_size"]
                or env.action_size != metadata["action_size"]
            ):
                raise ValueError(
                    "Selected environment dimensions differ from the frozen policy contract"
                )
            save_report(rec.path / "scene-manifest.json", env.scene_manifest)
            evaluator = NavigationEvaluator(env, maker, list(range(start, start + per_difficulty)))
            rec.phase("evaluating")
            tic = time.monotonic()
            threshold = resolved["evaluation"].get("quality_threshold")
            report, traces = evaluator.run(params, threshold)
            report.update(
                split=split,
                episodes_per_difficulty=per_difficulty,
                checkpoint=str(Path(config["checkpoint"]).resolve()),
                elapsed_seconds=time.monotonic() - tic,
            )
            save_report(rec.path / "eval/report.json", report)
            published = export_navigation_replays(
                env, traces, rec.path / "rollouts", case_indices=select_episodes(report)
            )
            save_report(rec.path / "rollouts/index.json", {"replays": published})
            rec.log(
                0,
                {
                    "eval/success_rate": report["success_rate"],
                    "eval/collision_rate": report["collision_rate"],
                    "eval/constrained_time_s": report["constrained_time_mean_s"],
                },
            )
            rec.finish(
                "completed",
                quality_passed=report["quality_passed"],
                full_budget_completed=True,
                num_trials=report["num_trials"],
                arrived=report["arrived"],
            )
            return report
        except BaseException as exc:
            rec.finish("failed", error=repr(exc))
            raise
        finally:
            if env is not None:
                env.close()
