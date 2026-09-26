"""Execute genuine optimization controllers on the frozen native LSY course."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import crazyflow  # noqa: F401
import jax
import jax.numpy as jnp
import numpy as np

from drone_playground.evaluation.racing import summarize_race
from drone_playground.evaluation.tracking import save_report, select_replays
from drone_playground.runs.console import capture_console
from drone_playground.runs.record import RunRecorder
from drone_playground.runs.rscope_io import export_rollout
from drone_playground.tasks.racing import RacingEnv, load_config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--controller", choices=["attitude_mpc", "sampling_mpc"], required=True)
    p.add_argument("--episodes", type=int, default=128)
    p.add_argument("--split", choices=["dev", "heldout"], default="heldout")
    p.add_argument("--run-id", required=True)
    p.add_argument("--seed-start", type=int)
    p.add_argument("--samples", type=int, default=2000)
    p.add_argument("--prediction-device", choices=["cpu", "gpu"], default="cpu")
    args = p.parse_args()
    config = dict(
        task="racing",
        controller=args.controller,
        drone="cf21B_500",
        dynamics="first_principles",
        course="lsy-level0",
        split=args.split,
        episodes=args.episodes,
        seed_start=args.seed_start,
        frequency=50,
        episode_length=1500,
        samples=args.samples,
        prediction_device=args.prediction_device,
        lsy_commit="b1f5b36adb8e08e8e2adea85de790bd0e0a1d118",
        scope="preset-reference tracking with native gate completion, not free minimum-time planning",
    )
    config["native_disturbances"] = load_config().env.disturbances.to_dict()
    rec = RunRecorder(Path(__file__).resolve().parents[1], args.run_id, config, task_id="06")
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
        env = RacingEnv(device="cpu", dynamics="first_principles", reference_seed=seeds[0])
        step_fn = jax.jit(env.step_physical)
        reset = jax.jit(env.reset)
        state = reset(jax.random.PRNGKey(seeds[0]))
        execution_devices = [str(device) for device in state.obs.devices()]
        if args.controller == "attitude_mpc":
            from drone_playground.controllers.lsy_mpc import LSYAttitudeMPC

            ctrl = LSYAttitudeMPC(
                env.controller_observation(state), {}, env.config, workdir=rec.path / "acados-code"
            )
        else:
            from drone_playground.controllers.sampling import SamplingMPC

            ctrl = SamplingMPC(
                drone=env.drone,
                reference=np.asarray(env.trajectories[0]),
                frequency=50,
                samples=args.samples,
                device=args.prediction_device,
                obstacles=np.asarray(env.default.obstacles_pos[0]),
            )
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
            finished_flags.append(bool(finished and not host.metrics["success"]))
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
                    "eval/gates_passed": float(host.metrics["gates_passed"]),
                    "eval/completed": float(host.metrics["success"]),
                    "eval/collision": float(host.metrics["collision"]),
                },
            )
            print(
                json.dumps(
                    dict(
                        run_id=args.run_id,
                        case=case,
                        seed=seed,
                        steps=length,
                        gates=float(host.metrics["gates_passed"]),
                        success=bool(host.metrics["success"]),
                        collision=bool(host.metrics["collision"]),
                        controller_finished=finished,
                    )
                ),
                flush=True,
            )
        trace = jax.tree.map(lambda *xs: np.stack(xs, axis=1), *all_traces)
        report = summarize_race(trace, seeds, env.dt)
        for i, flag in enumerate(finished_flags):
            if flag:
                report["episodes"][i]["controller_finished"] = True
                report["episodes"][i]["time_out"] = False
        report["timeouts"] = sum(x["time_out"] for x in report["episodes"])
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
        print(
            json.dumps({k: v for k, v in report.items() if k != "episodes"}, indent=2), flush=True
        )
    except BaseException as exc:
        rec.finish("failed", error=repr(exc))
        raise
    finally:
        if ctrl is not None:
            ctrl.close()
        if env is not None:
            env.close()
        console.__exit__(None, None, None)


if __name__ == "__main__":
    main()
