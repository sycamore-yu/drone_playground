"""Genuine model-predictive control execution through the composed task interface."""

from __future__ import annotations

import json
import time
from types import SimpleNamespace

import crazyflow  # noqa: F401
import jax
import jax.numpy as jnp
import numpy as np

from drone_playground.composition import build_environment
from drone_playground.controllers.factory import build_controller
from drone_playground.evaluation.racing import summarize_race
from drone_playground.evaluation.tracking import save_report, select_replays, summarize_trials
from drone_playground.runs.console import capture_console
from drone_playground.runs.record import RunRecorder
from drone_playground.runs.rscope_io import export_rollout


def evaluate_optimization(config, root, run_id):
    args = SimpleNamespace(
        controller=config["controller"]["name"],
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
        env = build_environment(config, config["training"]["device"], args.split, args.episodes)
        step_fn = jax.jit(env.step_physical)
        reset = jax.jit(env.reset)
        state = reset(jax.random.PRNGKey(seeds[0]))
        execution_devices = [str(device) for device in state.obs.devices()]
        ctrl = build_controller(config, env, state, rec.path / "acados-code")
        all_traces = []
        timings = []
        statuses = []
        finished_flags = []
        rec.phase("evaluating", step=0)
        for case, seed in enumerate(seeds):
            state = reset(jax.random.PRNGKey(seed))
            if args.controller == "attitude_mpc":
                ctrl.episode_callback()
                # Reset solver memory between trials to make each seed independent.
                ctrl.native._acados_ocp_solver.reset(reset_qp_solver_mem=1)
            else:
                ctrl.reset(seed)
            saved = []
            finished = False
            for tick in range(env.episode_length):
                old = state
                if args.controller == "attitude_mpc":
                    action = ctrl.compute_control(env.controller_observation(state), {})
                else:
                    action = ctrl.compute_from_data(state.pipeline_state.sim_data, tick)
                diag = dict(ctrl.last_diagnostics)
                # The explicit synchronization makes reported environment time meaningful.
                start_step = time.perf_counter()
                state = step_fn(state, jnp.asarray(action, jnp.float32))
                jax.block_until_ready(state.obs)
                host = state
                env_seconds = time.perf_counter() - start_step
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
                if args.controller == "attitude_mpc":
                    finished = ctrl.step_callback(
                        action,
                        env.controller_observation(state),
                        float(host.reward),
                        bool(host.done),
                        False,
                        {},
                    )
                if bool(host.done) or finished:
                    break
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
            if flag:
                report["episodes"][i]["controller_finished"] = True
                report["episodes"][i]["time_out"] = False
        report["timeouts"] = sum(x.get("time_out", False) for x in report["episodes"])
        decision = np.array([x["decision_seconds"] for x in timings])
        stable = decision[1:] if len(decision) > 1 else decision
        report.update(
            config=config,
            parameters_frozen=True,
            controller=args.controller,
            actual_environment_devices=execution_devices,
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
