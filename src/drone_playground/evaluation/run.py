"""One evaluation entry for frozen neural policies and online optimization."""

from __future__ import annotations

import copy
import time
from pathlib import Path

from drone_playground.artifacts.console import capture_console
from drone_playground.artifacts.record import RunRecorder
from drone_playground.artifacts.reporting import save_report
from drone_playground.composition import build_environment
from drone_playground.evaluation.tracking.metrics import select_replays
from drone_playground.networks.policies import NeuralPolicy
from drone_playground.visualization.rscope_io import export_rollout


def make_evaluator(env, make_policy, seeds):
    from hydra.utils import get_class

    target = env.experiment_config["env"]["task"]["policy_evaluator"]
    arguments = {}
    if env.task == "navigation" and env.experiment_config["evaluation"].get(
        "protocol"
    ):
        import jax.numpy as jnp

        from drone_playground.benchmarks import load_protocol
        from drone_playground.evaluation.navigation.cases import navigation_cases, navigation_resets

        config = env.experiment_config
        protocol = load_protocol(config["evaluation"]["protocol"])
        cases = navigation_cases(
            env.bank, len(seeds), seeds[0], True, protocol=protocol
        )
        ordered = [case for group in cases.values() for case in group]
        initial_spec = (
            config["training"].get("checkpoint_eval_initial_conditions")
            if config.get("mode") == "train"
            else None
        )
        initials = navigation_resets(
            env.bank,
            [c["scenario_id"] for c in ordered],
            [c["seed"] for c in ordered],
            initial_spec
            or config["evaluation"].get("initial_conditions")
            or protocol["initial_conditions"],
            env.body_radius,
        )
        for index, case in enumerate(ordered):
            case["initial_state"] = {
                field: jnp.asarray(initials[field][index])
                for field in ("position", "velocity", "quaternion")
            }
            case["initial_position_m"] = initials["record"]["position_m"][index]
        arguments.update(cases=cases, initial_conditions=initials["record"])
    return get_class(target)(env, make_policy, seeds, **arguments)


def resolve_evaluation_config(config, metadata):
    selection = config["evaluation"].get("environment", "checkpoint")
    if selection == "checkpoint":
        resolved = copy.deepcopy(metadata["config"])
    elif selection == "config":
        resolved = copy.deepcopy(config)
        if (
            resolved["method"]["output"]
            != metadata["config"]["method"]["output"]
        ):
            raise ValueError(
                "Frozen policy output and requested execution command contract differ"
            )
        # Network and training algorithm identify the loaded policy, while
        # task/controller/model/scene/observations identify the selected evaluation.
        for group in ("network", "algorithm", "method"):
            resolved[group] = copy.deepcopy(metadata["config"][group])
    else:
        raise ValueError("evaluation.environment must be checkpoint or config")
    resolved["mode"] = config["mode"]
    resolved["evaluation"] = copy.deepcopy(config["evaluation"])
    resolved["runtime"]["device"] = config["runtime"]["device"]
    return resolved


def evaluate_experiment(config, root, run_id):
    if config["method"]["implementation"] in (
        "native_ego",
        "native_super",
        "native_service",
        "pipeline",
    ):
        if config["env"]["task"]["name"] != "navigation":
            from drone_playground.evaluation.tracking.external import evaluate_native_control

            return evaluate_native_control(config, root, run_id)
        from drone_playground.evaluation.navigation.external import evaluate_native

        return evaluate_native(config, root, run_id)
    if config["env"]["task"]["name"] == "navigation":
        from drone_playground.evaluation.navigation.policy import evaluate_navigation

        return evaluate_navigation(config, root, run_id)
    if config["method"]["implementation"] in (
        "attitude_mpc",
        "sampling_mpc",
    ) and not config.get("checkpoint"):
        from drone_playground.evaluation.mpc import evaluate_optimization

        return evaluate_optimization(config, root, run_id)
    if not config.get("checkpoint"):
        raise ValueError("Frozen neural execution requires checkpoint=<path>")
    policy = NeuralPolicy.load(config["checkpoint"])
    maker, params, metadata = (
        policy.make_policy,
        policy.parameters,
        policy.metadata,
    )
    resolved = resolve_evaluation_config(config, metadata)
    rec = RunRecorder(
        root, run_id, resolved, task_id="composable-flight/05-task-evaluation"
    )
    env = None
    with capture_console(rec.path / "console.log"):
        try:
            rec.phase("initializing")
            role = resolved["evaluation"]["role"]
            count = resolved["evaluation"]["episodes"]
            start = resolved["evaluation"].get("seed_start")
            start = start if start is not None else 30000
            env = build_environment(
                resolved, resolved["runtime"]["device"], role, count
            )
            rec.record_environment(env)
            if (
                env.observation_size != metadata["observation_size"]
                or env.action_size != metadata["action_size"]
            ):
                raise ValueError(
                    "Selected environment dimensions differ from the frozen policy contract"
                )
            evaluator = make_evaluator(
                env, maker, list(range(start, start + count))
            )
            rec.phase("evaluating")
            tic = time.monotonic()
            report, trace = evaluator.run(params)
            from drone_playground.benchmarks import apply_quality

            apply_quality(report, resolved)
            report.update(
                role=role,
                checkpoint=str(Path(config["checkpoint"]).resolve()),
                elapsed_seconds=time.monotonic() - tic,
            )
            from drone_playground.benchmarks import benchmark_id

            if benchmark_id(resolved) in ("tracking-v1", "racing-v1"):
                from drone_playground.benchmarks import validate_control_report

                validation = validate_control_report(report, env.task)
                save_report(
                    rec.path / "eval/benchmark-validation.json", validation
                )
                report.update(
                    quality_passed=validation["passed"],
                    quality_rule=validation["protocol"],
                )
            save_report(rec.path / "eval/report.json", report)
            if resolved["evaluation"].get("record_replays", False):
                export_rollout(
                    env.sim, rec.path / "rollouts", select_replays(trace, report)
                )
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
