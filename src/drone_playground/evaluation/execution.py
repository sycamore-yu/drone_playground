"""One evaluation entry for frozen neural policies and online optimization."""

from __future__ import annotations

import copy
import time
from pathlib import Path

from drone_playground.composition import build_environment
from drone_playground.policies.neural import NeuralPolicy
from drone_playground.runs.console import capture_console
from drone_playground.runs.record import RunRecorder
from drone_playground.runs.rscope_io import export_rollout

from .tracking import PolicyEvaluator, save_report, select_replays


def make_evaluator(env, make_policy, seeds):
    if env.task.startswith("lotf_"):
        from .lotf import LOTFEvaluator

        return LOTFEvaluator(env, make_policy, seeds)
    if env.task == "navigation":
        from .navigation import NavigationEvaluator

        return NavigationEvaluator(env, make_policy, seeds)
    if env.task == "racing":
        from .racing import RaceEvaluator

        return RaceEvaluator(env, make_policy, seeds)
    return PolicyEvaluator(env, make_policy, seeds)


def resolve_evaluation_config(config, metadata):
    selection = config["evaluation"].get("environment", "checkpoint")
    if selection == "checkpoint":
        resolved = copy.deepcopy(metadata["config"])
    elif selection == "experiment":
        resolved = copy.deepcopy(config)
        if resolved["policy"]["output"] != metadata["config"]["policy"]["output"]:
            raise ValueError("Frozen policy output and requested execution command contract differ")
        # Network and training algorithm identify the loaded policy, while
        # task/controller/model/scene/observations identify the selected evaluation.
        for group in ("network", "algorithm"):
            resolved[group] = copy.deepcopy(metadata["config"][group])
    else:
        raise ValueError("evaluation.environment must be checkpoint or experiment")
    resolved["mode"] = config["mode"]
    resolved["evaluation"] = copy.deepcopy(config["evaluation"])
    resolved["training"]["device"] = config["training"]["device"]
    return resolved


def evaluate_experiment(config, root, run_id):
    if config["policy"]["name"] in ("native_ego", "native_super"):
        from .native_planners import evaluate_native

        return evaluate_native(config, root, run_id)
    if config["task"]["name"] == "navigation":
        from .navigation import evaluate_navigation

        return evaluate_navigation(config, root, run_id)
    if config["controller"]["name"] in ("attitude_mpc", "sampling_mpc") and not config.get(
        "checkpoint"
    ):
        from .optimization import evaluate_optimization

        return evaluate_optimization(config, root, run_id)
    if not config.get("checkpoint"):
        raise ValueError("Frozen neural execution requires checkpoint=<path>")
    policy = NeuralPolicy.load(config["checkpoint"])
    maker, params, metadata = policy.make_policy, policy.parameters, policy.metadata
    resolved = resolve_evaluation_config(config, metadata)
    rec = RunRecorder(root, run_id, resolved, task_id="composable-flight/05-task-evaluation")
    env = None
    with capture_console(rec.path / "console.log"):
        try:
            rec.phase("initializing")
            split = resolved["evaluation"]["split"]
            count = resolved["evaluation"]["episodes"]
            start = resolved["evaluation"].get("seed_start")
            start = start if start is not None else (20000 if split == "dev" else 30000)
            env = build_environment(resolved, resolved["training"]["device"], split, count)
            if (
                env.observation_size != metadata["observation_size"]
                or env.action_size != metadata["action_size"]
            ):
                raise ValueError(
                    "Selected environment dimensions differ from the frozen policy contract"
                )
            evaluator = make_evaluator(env, maker, list(range(start, start + count)))
            rec.phase("evaluating")
            tic = time.monotonic()
            report, trace = evaluator.run(params)
            report.update(
                split=split,
                checkpoint=str(Path(config["checkpoint"]).resolve()),
                elapsed_seconds=time.monotonic() - tic,
            )
            save_report(rec.path / "eval/report.json", report)
            export_rollout(env.sim, rec.path / "rollouts", select_replays(trace, report))
            rec.log(
                0,
                {
                    "eval/completion_rate": report["completion_rate"],
                    "eval/rmse_all_m": report["rmse_all_mean"],
                },
            )
            rec.finish(
                "completed",
                quality_passed=report["quality_passed"],
                full_budget_completed=True,
                num_trials=count,
                completed=report["completed"],
            )
            return report
        except BaseException as exc:
            rec.finish("failed", error=repr(exc))
            raise
        finally:
            if env is not None:
                env.close()
