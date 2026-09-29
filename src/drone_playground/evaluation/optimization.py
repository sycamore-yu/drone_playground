"""Genuine model-predictive control execution through the composed task interface."""

from __future__ import annotations

import hashlib
import inspect
import json
import time
from pathlib import Path
from types import SimpleNamespace

import crazyflow  # noqa: F401
import jax
import numpy as np

from drone_playground.composition import build_environment
from drone_playground.evaluation.racing import summarize_race
from drone_playground.evaluation.tracking import save_report, select_replays, summarize_trials
from drone_playground.methods.optimal_control.factory import build_controller
from drone_playground.runs.console import capture_console
from drone_playground.runs.record import RunRecorder
from drone_playground.runtime.host_runner import run_steps
from drone_playground.visualization.rscope_io import export_rollout


def evaluate_optimization(config, root, run_id):
    args = SimpleNamespace(
        controller=config["method"]["decision"]["name"],
        episodes=config["evaluation"]["episodes"],
        split=config["evaluation"]["split"],
        seed_start=config["evaluation"].get("seed_start"),
        run_id=run_id,
    )
    rec = RunRecorder(root, run_id, config, task_id="composable-flight/02-policy-control")
    console = capture_console(rec.path / "console.log")
    console.__enter__()
    seed_start = (
        args.seed_start
        if args.seed_start is not None
        else (20000 if args.split == "dev" else 30000)
    )
    seeds = list(range(seed_start, seed_start + args.episodes))
    env = None
    ctrl = None
    began = time.monotonic()
    try:
        rec.phase("initializing")
        env = build_environment(config, config["runtime"]["device"], args.split, args.episodes)
        step_fn = jax.jit(env.step_physical)
        reset = jax.jit(env.reset)
        state = reset(jax.random.PRNGKey(seeds[0]))
        execution_devices = [str(device) for device in state.obs.devices()]
        ctrl = build_controller(config, env, state, rec.path / "acados-code")
        all_traces = []
        timings = []
        statuses = []
        finished_flags = []
        reset_details = []
        rec.phase("evaluating", step=0)
        for case, seed in enumerate(seeds):
            state = reset(jax.random.PRNGKey(seed))
            reset_details.append({key: float(state.info[key]) for key in
                                  ("delay_requested_ms", "delay_effective_ms") if key in state.info})
            if args.controller == "attitude_mpc":
                ctrl.episode_callback()
                # Reset solver memory between trials to make each seed independent.
                ctrl.native._acados_ocp_solver.reset(reset_qp_solver_mem=1)
            else:
                ctrl.reset(seed)
            saved = []
            finished = False

            def decide(current, tick):
                if args.controller == "attitude_mpc":
                    action = ctrl.compute_control(env.controller_observation(current), {})
                else:
                    action = ctrl.compute_control(env.controller_observation(current), tick)
                return action, dict(ctrl.last_diagnostics)

            def after_step(transition):
                if args.controller == "attitude_mpc":
                    current = transition.after
                    return ctrl.step_callback(
                        transition.command,
                        env.controller_observation(current),
                        float(current.reward),
                        bool(current.done),
                        False,
                        {},
                    )
                return False

            for tick, transition, finished in run_steps(
                state, env.episode_length, decide, step_fn, after_step
            ):
                old, state = transition.before, transition.after
                action, diag = transition.command, transition.diagnostics
                host = state
                env_seconds = transition.environment_seconds
                statuses.append(diag.get("status", 0))
                timings.append(dict(case=case, step=tick, **diag, environment_seconds=env_seconds))
                x = host.pipeline_state.sim_data.states
                normalized = np.asarray((action - env.low) / (env.high - env.low) * 2 - 1)
                record = dict(
                    pos=np.asarray(x.pos[0, 0]),
                    quat=np.asarray(x.quat[0, 0]),
                    obs=np.asarray(old.obs),
                    actions=normalized,
                    reward=float(host.reward),
                    time=(tick + 1) * env.dt,
                    metrics={name: float(value) for name, value in host.metrics.items()},
                    active=True,
                    failed=bool(host.metrics["failure"] > 0),
                )
                record["metrics"]["solver_status"] = diag.get("status", 0)
                record["metrics"]["decision_seconds"] = diag["decision_seconds"]
                saved.append(record)
            finished_flags.append(
                bool(finished and not host.metrics.get("success", not bool(host.done)))
            )
            # Fixed-size, full-duration recording with a separate active mask.
            length = len(saved)
            if not length:
                raise AssertionError("Controller produced no physical transitions")
            last = saved[-1]
            for tick in range(length, env.episode_length):
                saved.append({**last, "active": False, "reward": 0.0, "time": (tick + 1) * env.dt})
            trial = jax.tree.map(lambda *xs: np.asarray(xs), *saved)
            all_traces.append(trial)
            rec.phase("evaluating", step=case + 1, completed_trials=case + 1)
            rec.log(
                case + 1,
                {
                    "eval/gates_passed": float(host.metrics.get("gates_passed", 0)),
                    "eval/completed": float(host.metrics.get("success", not bool(host.done))),
                    "eval/collision": float(host.metrics.get("collision", host.metrics["failure"])),
                },
            )
            print(
                json.dumps(
                    dict(
                        run_id=args.run_id,
                        case=case,
                        seed=seed,
                        steps=length,
                        gates=float(host.metrics.get("gates_passed", 0)),
                        success=bool(host.metrics.get("success", not bool(host.done))),
                        collision=bool(host.metrics.get("collision", host.metrics["failure"])),
                        controller_finished=finished,
                    )
                ),
                flush=True,
            )
        trace = jax.tree.map(lambda *xs: np.stack(xs, axis=1), *all_traces)
        report = (summarize_race if env.task == "racing" else summarize_trials)(
            trace, seeds, env.dt
        )
        for i, flag in enumerate(finished_flags):
            report["episodes"][i].update(reset_details[i])
            if flag:
                report["episodes"][i]["controller_finished"] = True
                report["episodes"][i]["time_out"] = False
        report["timeouts"] = sum(x.get("time_out", False) for x in report["episodes"])
        decision = np.array([x["decision_seconds"] for x in timings])
        stable = decision[1:] if len(decision) > 1 else decision
        runtime_files = [Path(inspect.getfile(type(ctrl))), Path(inspect.getfile(build_controller))]
        if args.controller == "attitude_mpc":
            runtime_files += [Path(inspect.getfile(type(ctrl.native)))]
            if ctrl.delay_predictor is not None:
                runtime_files += [Path(inspect.getfile(type(ctrl.delay_predictor)))]
            runtime_files += [ctrl.source / "lib" / name for name in
                              ("libacados.so", "libblasfeo.so", "libhpipm.so")]
        runtime_identity = {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
                            for path in runtime_files}
        report.update(
            config=config,
            split=args.split,
            parameter_sha256=hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
            parameter_identity_kind="resolved optimization configuration",
            runtime_identity=runtime_identity,
            parameters_frozen=True,
            controller=args.controller,
            actual_environment_devices=execution_devices,
            prediction_device=str(getattr(ctrl, "prediction_device", "cpu/native-acados")),
            actual_steps=len(timings),
            elapsed_seconds=time.monotonic() - began,
            nonzero_solve_status_count=int(np.count_nonzero(statuses)),
            first_decision_seconds=float(decision[0]),
            decision_p50_ms=float(np.median(stable) * 1000),
            decision_p95_ms=float(np.quantile(stable, 0.95) * 1000),
            deadline_miss_fraction=float(np.mean(stable > env.dt)),
            source="LSY AttitudeMPC"
            if args.controller == "attitude_mpc"
            else "Crazyflow sampling.py",
            timing_protocol="synchronous simulation; controller latency measured, not injected as control delay",
        )
        if args.controller == 'attitude_mpc' and ctrl.delay_predictor is not None:
            report['execution_adaptation'] = dict(
                name='issued_command_delay_prediction',
                estimated_delay_ms=ctrl.delay_predictor.delay_seconds * 1000,
                integration_step_ms=ctrl.delay_predictor.max_step_seconds * 1000,
                input='current observed state, model and previously issued commands only',
                reference='fixed task reference advanced by estimated delay',
            )
        save_report(rec.path / "eval/report.json", report)
        if config["evaluation"].get("release_validation") == "control-v1":
            from drone_playground.evaluation.release_control import validate_control_report

            validation = validate_control_report(report, "racing" if env.task == "racing" else "tracking")
            save_report(rec.path / "eval/release-validation.json", validation)
            report.update(quality_passed=validation["passed"], quality_rule=validation["protocol"])
            save_report(rec.path / "eval/report.json", report)
        save_report(rec.path / "eval/solver-steps.json", dict(steps=timings))
        export_rollout(env.sim, rec.path / "rollouts", select_replays(trace, report))
        rec.finish(
            "completed",
            engineer_passed=True,
            full_budget_completed=True,
            quality_passed=report["quality_passed"],
            report="eval/report.json",
            completed=report["completed"],
            num_trials=args.episodes,
        )
        return report
    except BaseException as exc:
        rec.finish("failed", error=repr(exc))
        raise
    finally:
        if ctrl is not None:
            ctrl.close()
        if env is not None:
            env.close()
        console.__exit__(None, None, None)
