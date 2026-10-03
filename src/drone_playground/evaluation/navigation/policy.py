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

from drone_playground.environments.scenes.geometry import DIFFICULTIES
from drone_playground.evaluation.navigation.metrics import combine_cells, summarize_cell


class NavigationEvaluator:
    """Reuse one compiled rollout per difficulty for every snapshot of a run."""

    def __init__(
        self,
        env,
        make_policy,
        reset_seeds: list[int],
        *,
        cases=None,
        initial_conditions=None,
    ):
        self.env = env
        self.make_policy, self.seeds = make_policy, list(reset_seeds)
        self.reset_seeds = list(reset_seeds)
        self.initial_conditions = initial_conditions
        self._labels = {}
        per_difficulty = env.bank.num_instances // len(DIFFICULTIES)
        if per_difficulty < 1:
            raise ValueError("The evaluation bank has no per-difficulty instances")
        self.groups: dict[str, list[int]] = {}
        self._run = {}
        for index, difficulty in enumerate(DIFFICULTIES):
            rows = (
                cases[difficulty]
                if cases is not None
                else [
                    dict(
                        scenario_id=scenario,
                        seed=reset_seeds[i % len(reset_seeds)],
                    )
                    for i, scenario in enumerate(
                        range(index * per_difficulty, (index + 1) * per_difficulty)
                    )
                ]
            )
            if not rows:
                continue
            scenario_ids = jnp.asarray([row["scenario_id"] for row in rows], jnp.int32)
            self.groups[difficulty] = [int(value) for value in scenario_ids]
            keys = jnp.stack([jax.random.PRNGKey(row["seed"]) for row in rows])
            initials = (
                jax.tree.map(
                    lambda *values: jnp.stack(values),
                    *[row["initial_state"] for row in rows],
                )
                if cases is not None
                else None
            )
            self._labels[difficulty] = [
                {
                    **env.bank.labels(row["scenario_id"]),
                    **{key: value for key, value in row.items() if key != "initial_state"},
                }
                for row in rows
            ]
            self._run[difficulty] = jax.jit(
                self._make_rollout(make_policy, keys, scenario_ids, initials)
            )

    def _make_rollout(self, make_policy, keys, scenario_ids, initial_states):
        from drone_playground.runtime.jax_runner import policy_rollout

        return lambda params: policy_rollout(
            self.env,
            make_policy,
            params,
            keys,
            reference_ids=scenario_ids,
            initial_states=initial_states,
            kind="navigation",
        )

    def labels(self) -> dict[str, list[dict]]:
        return self._labels

    def run(self, params):
        from drone_playground.artifacts.reporting import tree_digest

        before = tree_digest(params)
        traces = {}
        labels = self.labels()
        for difficulty in self.groups:
            trace = jax.tree.map(np.asarray, self._run[difficulty](params))
            traces[difficulty] = trace
        after = tree_digest(params)
        if before != after:
            raise RuntimeError("Evaluation changed policy or normalization parameters")
        cells = {
            difficulty: summarize_cell(
                traces[difficulty],
                labels[difficulty],
                self.env.dt,
                self.env.duration,
            )
            for difficulty in self.groups
        }
        report = combine_cells(cells)
        report.update(
            parameter_sha256=before,
            parameters_frozen=True,
            task=self.env.task.name,
            dynamics=self.env.dynamics.forward,
            drone=self.env.drone,
            scene_bank_sha256=_bank_digest(self.env),
            scenario_groups={key: value for key, value in self.groups.items()},
            episodes=[row for cell in cells.values() for row in cell["episodes"]],
            initial_conditions=self.initial_conditions,
        )
        report["scene_success_rates"] = {
            scene: float(
                np.mean(
                    [row["arrived"] for row in report["episodes"] if row.get("scene_id") == scene]
                )
            )
            for scene in {row["scene_id"] for row in report["episodes"] if "scene_id" in row}
        }
        from drone_playground.runtime.timing import measure_policy

        initial = self.env.reset(jax.random.PRNGKey(self.seeds[0]), jnp.int32(0))
        report.update(measure_policy(self.env, self.make_policy, params, initial.obs, initial))
        report["sensor_timing"] = getattr(self.env, "sensor_timing", None)
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


def export_navigation_replays(
    env,
    traces: dict,
    directory: Path,
    count: int = 4,
    case_indices: dict | None = None,
    scenario_groups: dict | None = None,
    decision_root: Path | None = None,
) -> list[dict]:
    """Write one self-contained rscope replay per difficulty cell."""
    from drone_playground.visualization.navigation_scene import (
        active_indices,
        create_replay_model,
        obstacle_track,
    )
    from drone_playground.visualization.rscope_io import export_rollout

    per_difficulty = env.bank.num_instances // len(DIFFICULTIES)
    published = []
    for index, difficulty in enumerate(DIFFICULTIES):
        if difficulty not in traces:
            continue
        trace = traces[difficulty]
        cases = (
            case_indices[difficulty]
            if case_indices is not None
            else list(range(min(count, trace["pos"].shape[1])))
        )
        for case in cases:
            scenario_id = (
                scenario_groups[difficulty][case]
                if scenario_groups is not None
                else index * per_difficulty + case
            )
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
            if decision_root is not None:
                from drone_playground.artifacts.decisions import load_native_decisions
                from drone_playground.visualization.layers import layers_from_decisions

                replay_model.replay_visualization = layers_from_decisions(
                    load_native_decisions(
                        decision_root / difficulty / str(case) / "decision-trace"
                    ),
                    sensor=replay_model.replay_visualization.sensor,
                )
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

    from drone_playground.artifacts.console import capture_console
    from drone_playground.artifacts.record import RunRecorder
    from drone_playground.artifacts.reporting import save_report
    from drone_playground.environments.environment import build_environment
    from drone_playground.evaluation.run import resolve_evaluation_config
    from drone_playground.networks.policies import NeuralPolicy

    policy = NeuralPolicy.load(config["checkpoint"])
    maker, params, metadata = (
        policy.make_policy,
        policy.parameters,
        policy.metadata,
    )
    resolved = resolve_evaluation_config(config, metadata)
    role = resolved["evaluation"]["role"]
    per_difficulty = int(resolved["evaluation"]["episodes"])
    rec = RunRecorder(root, run_id, resolved, task_id="p5-navigation-evaluation")
    env = None
    with capture_console(rec.path / "console.log"):
        try:
            rec.phase("initializing")
            start = resolved["evaluation"].get("seed_start")
            start = start if start is not None else 30000
            env = build_environment(resolved, resolved["runtime"]["device"], role, per_difficulty)
            rec.record_environment(env)
            if (
                env.observation_size != metadata["observation_size"]
                or env.action_size != metadata["action_size"]
            ):
                raise ValueError(
                    "Selected environment dimensions differ from the frozen policy contract"
                )
            save_report(rec.path / "scene-manifest.json", env.scene_manifest)
            from drone_playground.evaluation.run import make_evaluator

            evaluator = make_evaluator(env, maker, list(range(start, start + per_difficulty)))
            rec.phase("evaluating")
            tic = time.monotonic()
            report, traces = evaluator.run(params)
            from drone_playground.benchmarks import apply_quality

            apply_quality(report, resolved)
            report.update(
                role=role,
                episodes_per_scene=per_difficulty if report["initial_conditions"] else None,
                episodes_per_difficulty=None if report["initial_conditions"] else per_difficulty,
                checkpoint=str(Path(config["checkpoint"]).resolve()),
                elapsed_seconds=time.monotonic() - tic,
            )
            from drone_playground.benchmarks import (
                benchmark_id,
                load_protocol,
                validate_navigation_report,
            )

            criterion = benchmark_id(resolved)
            if criterion:
                protocol = load_protocol(resolved["evaluation"]["protocol"])
                scenes = {row["scene_id"] for row in report["episodes"]}
                tasks = [name for name, group in protocol["scenes"].items() if set(group) <= scenes]
                validation = validate_navigation_report(
                    report, tasks=tasks, criterion=criterion, protocol=protocol
                )
                report.update(
                    quality_passed=validation["passed"],
                    quality_rule=validation["protocol"],
                )
                save_report(rec.path / "eval/benchmark-validation.json", validation)
            save_report(rec.path / "eval/report.json", report)
            from drone_playground.artifacts.traces import save_navigation_traces

            save_navigation_traces(env, traces, rec.path / "traces", report["scenario_groups"])
            published = []
            if resolved["evaluation"].get("record_replays", False):
                published = export_navigation_replays(
                    env,
                    traces,
                    rec.path / "rollouts",
                    case_indices=select_episodes(report),
                    scenario_groups=report["scenario_groups"],
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
