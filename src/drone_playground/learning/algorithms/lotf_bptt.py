"""LOTF native BPTT update with durable checkpoints and external evaluation.

The update is adapted from lotf/algos/bptt.py (GPLv3, pinned submodule), preserving
the time scan, reward sum, RNG splits and Adam update. Outer host orchestration
adds snapshots without reinitializing optimizer, environment, or random state.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
import traceback
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from brax.io import model as storage
from flax.training.train_state import TrainState
from lotf.algos.bptt import RunnerState

from drone_playground.composition import build_environment
from drone_playground.evaluation.lotf import LOTFEvaluator
from drone_playground.evaluation.tracking import save_report, select_replays, tree_digest
from drone_playground.networks.policies import lotf_inference_factory, make_lotf_network
from drone_playground.runs.console import capture_console
from drone_playground.runs.record import RunRecorder
from drone_playground.visualization.rscope_io import export_rollout


def initialize(task, config):
    training = config["training"]
    key_init, key_bptt = jax.random.split(jax.random.key(training["seed"]))
    network = make_lotf_network(
        task.observation_size, task.action_size, config["network"], task.hover_action
    )
    params = network.initialize(key_init)
    schedule = optax.cosine_decay_schedule(
        config["algorithm"]["learning_rate"], training["policy_updates"]
    )
    state = TrainState.create(apply_fn=network.apply, params=params, tx=optax.adam(schedule))
    key_bptt, key_reset = jax.random.split(key_bptt)
    initial, obs = task.training_env.reset(jax.random.split(key_reset, training["num_envs"]), None)
    return RunnerState(state, initial, obs, key_bptt, 0), network


def make_update(task, config):
    env = task.training_env
    count = config["training"]["num_envs"]
    horizon = task.episode_length

    @jax.jit
    def update(runner):
        def objective(params, runner):
            def step(old, _):
                action = old.train_state.apply_fn(params, old.last_obs)
                key, key_step = jax.random.split(old.key)
                keys = jax.random.split(key_step, count)
                env_state, obs, reward, terminated, truncated, _ = env.step(
                    old.env_state, action, None, keys
                )
                return old._replace(env_state=env_state, last_obs=obs, key=key), (
                    reward,
                    jnp.mean(terminated.astype(jnp.float32)),
                    jnp.mean(truncated.astype(jnp.float32)),
                )

            end, (rewards, terminated, truncated) = jax.lax.scan(step, runner, None, length=horizon)
            return -rewards.sum() / count, (end, jnp.mean(terminated), jnp.mean(truncated))

        (loss, (end, terminal, timeout)), gradient = jax.value_and_grad(objective, has_aux=True)(
            runner.train_state.params, runner
        )
        trained = runner.train_state.apply_gradients(grads=gradient)
        end = end._replace(train_state=trained, epoch_idx=runner.epoch_idx + 1)
        norm = optax.tree.norm(gradient)
        return end, {
            "loss": loss,
            "gradient_norm": norm,
            "terminated_fraction": terminal,
            "truncated_fraction": timeout,
        }

    return update


def save_training_state(path, runner, config):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(
        params=runner.train_state.params,
        opt_state=runner.train_state.opt_state,
        optimizer_step=runner.train_state.step,
        env_state=runner.env_state,
        last_obs=runner.last_obs,
        key=runner.key,
        epoch_idx=runner.epoch_idx,
    )
    typed = []

    def to_host(keypath, leaf):
        if hasattr(leaf, "dtype") and jax.dtypes.issubdtype(leaf.dtype, jax.dtypes.prng_key):
            typed.append(jax.tree_util.keystr(keypath))
            return np.asarray(jax.random.key_data(leaf))
        return np.asarray(leaf)

    host = jax.tree_util.tree_map_with_path(to_host, payload)
    temp = path.with_suffix(".tmp")
    storage.save_params(str(temp), host)
    temp.replace(path)
    save_report(
        path.with_suffix(".json"),
        dict(
            config=config,
            typed_key_paths=typed,
            updates=int(runner.epoch_idx),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        ),
    )


def load_training_state(path, template, config):
    path = Path(path)
    meta = json.loads(path.with_suffix(".json").read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest() != meta["sha256"]:
        raise ValueError("LOTF continuation digest mismatch")
    from drone_playground.runs.migration import require_current

    meta["config"] = require_current(meta["config"])
    for group in ("method", "env", "objective", "network"):
        if config[group] != meta["config"][group]:
            raise ValueError(f"LOTF continuation configuration differs: {group}")
    # The first persisted configuration named an unused duration field
    # horizon_length. Both schemas execute the task's complete episode window.
    # Compare the actual algorithm semantics and retain immutable old artifacts.
    for name in ("name", "learning_rate", "schedule", "gradient"):
        if config["algorithm"].get(name) != meta["config"]["algorithm"].get(name):
            raise ValueError(f"LOTF continuation configuration differs: algorithm.{name}")
    for recipe in (config, meta["config"]):
        if recipe["algorithm"].get("rollout", "full_episode") != "full_episode":
            raise ValueError("LOTF continuation requires the declared full-episode rollout")
    for name in ("num_envs", "policy_updates", "seed"):
        if config["training"][name] != meta["config"]["training"][name]:
            raise ValueError(f"LOTF continuation training setting differs: {name}")
    typed = set(meta["typed_key_paths"])
    data = jax.tree_util.tree_map_with_path(
        lambda p, x: (
            jax.random.wrap_key_data(jnp.asarray(x))
            if jax.tree_util.keystr(p) in typed
            else jnp.asarray(x)
        ),
        storage.load_params(str(path)),
    )
    return template._replace(
        train_state=template.train_state.replace(
            params=data["params"], opt_state=data["opt_state"], step=data["optimizer_step"]
        ),
        env_state=data["env_state"],
        last_obs=data["last_obs"],
        key=data["key"],
        epoch_idx=data["epoch_idx"],
    )


def save_policy(path, params, task, config, step):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    storage.save_params(str(temp), jax.tree.map(np.asarray, params))
    temp.replace(path)
    metadata = dict(
        config_version=3,
        config=config,
        step=step,
        policy_family="lotf_mlp",
        observation_size=task.observation_size,
        action_size=task.action_size,
        action_bias=np.asarray(task.hover_action).tolist(),
        checkpoint_kind="inference-parameters",
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        parameter_sha256=tree_digest(params),
    )
    save_report(path.with_suffix(".json"), metadata)
    return path


def load_policy(path):
    path = Path(path)
    meta = json.loads(path.with_suffix(".json").read_text())
    from drone_playground.runs.migration import require_current

    meta["config"] = require_current(meta["config"])
    if hashlib.sha256(path.read_bytes()).hexdigest() != meta["sha256"]:
        raise ValueError("LOTF policy file digest mismatch")
    params = storage.load_params(str(path))
    if tree_digest(params) != meta["parameter_sha256"]:
        raise ValueError("LOTF parameter content changed")
    network = make_lotf_network(
        meta["observation_size"],
        meta["action_size"],
        meta["config"]["network"],
        jnp.asarray(meta["action_bias"]),
    )
    return lotf_inference_factory(network), params, meta


def development_score(task_name, report):
    error = (
        report["last_second_rmse_mean_m"] if task_name == "lotf_hover" else report["rmse_all_mean"]
    )
    return report["completed"], -error


def inherit_development_best(resume_path, destination, task_name):
    """Carry model selection across continuation, restricted to the saved epoch.

    A standalone state restores numerical training. Complete experiment
    continuation additionally needs its sibling development reports and policies.
    Reading the source's latest best.json would leak evaluations from future
    updates when resuming an earlier checkpoint, so reconstruct the eligible set.
    """
    resume_path, destination = Path(resume_path).resolve(), Path(destination)
    meta = json.loads(resume_path.with_suffix(".json").read_text())
    cutoff = int(meta["updates"])
    source = resume_path.parent.parent
    candidates = []
    for report_path in sorted((source / "eval").glob("step-*.json")):
        report = json.loads(report_path.read_text())
        if report.get("split") == "dev" and int(report.get("updates", cutoff + 1)) <= cutoff:
            candidates.append((development_score(task_name, report), report_path, report))
    if not candidates:
        if cutoff == 0:
            return (-1, -float("inf")), None
        raise ValueError(
            "Exact experiment continuation requires the source run's development reports "
            "and checkpoints up to the resumed update"
        )
    score, report_path, best = max(candidates, key=lambda row: row[0])
    policy_path = (source / best["checkpoint"]).resolve()
    if not policy_path.is_relative_to(source):
        raise ValueError("Development checkpoint must belong to its source run")
    # Check both file and parameter digests before importing the selected policy.
    _, _, policy_meta = load_policy(policy_path)
    if int(policy_meta["step"]) != int(best["step"]):
        raise ValueError("Development report and checkpoint step disagree")
    checkpoints = destination / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    for item in (policy_path, policy_path.with_suffix(".json")):
        shutil.copyfile(item, checkpoints / item.name)
    inherited = {
        **best,
        "checkpoint": f"checkpoints/{policy_path.name}",
        "inherited_from": str(source),
        "source_report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
    }
    save_report(destination / "eval" / f"step-{best['step']:010d}.json", inherited)
    save_report(
        checkpoints / "best.json",
        dict(
            path=policy_path.name,
            step=best["step"],
            score=list(score),
            selection_split="dev",
            inherited_from=str(source),
            resumed_update=cutoff,
            checkpoint_sha256=policy_meta["sha256"],
        ),
    )
    save_report(
        destination / "resume-selection.json",
        dict(
            source_run=str(source),
            source_state=str(resume_path),
            source_state_sha256=meta["sha256"],
            resumed_update=cutoff,
            eligible_steps=sorted(int(row[2]["step"]) for row in candidates),
            selected_step=int(best["step"]),
            selected_score=list(score),
            selected_checkpoint_sha256=policy_meta["sha256"],
            source_report=str(report_path),
            source_report_sha256=inherited["source_report_sha256"],
            future_evaluations_excluded=True,
        ),
    )
    return score, inherited


def train(config, root, run_id):
    rec = RunRecorder(root, run_id, config, task_id="composable-flight/04-training")
    task = evaluation = None
    start = time.monotonic()
    with capture_console(rec.path / "console.log"):
        try:
            rec.phase("initializing", step=0)
            task = build_environment(config, device=config["runtime"]["device"])
            evaluation = build_environment(config, device=config["runtime"]["device"], split="dev")
            runner, network = initialize(task, config)
            initial = jax.tree.map(np.asarray, runner.train_state.params)
            if config["training"].get("resume"):
                runner = load_training_state(config["training"]["resume"], runner, config)
            initial_updates = int(runner.epoch_idx)
            make_policy = lotf_inference_factory(network)
            evaluator = LOTFEvaluator(evaluation, make_policy, list(range(20000, 20032)))
            update = make_update(task, config)
            budget = config["training"]["policy_updates"]
            transitions = config["training"]["num_envs"] * task.episode_length
            milestones = {int(x) for x in np.linspace(0, budget, config["training"]["num_evals"])}
            milestones.add(budget)
            best_score = (-1, -float("inf"))
            best = None
            if config["training"].get("resume"):
                best_score, best = inherit_development_best(
                    config["training"]["resume"], rec.path, task.task
                )
            timings = []

            def snapshot(runner):
                nonlocal best_score, best
                epoch = int(runner.epoch_idx)
                step = epoch * transitions
                rec.phase("checkpoint", step=step, updates=epoch)
                save_training_state(
                    rec.path / "training-state" / f"update-{epoch:06d}.pkl", runner, config
                )
                params = (None, runner.train_state.params)
                path = save_policy(
                    rec.path / "checkpoints" / f"step-{step:010d}.pkl", params, task, config, step
                )
                rec.phase("evaluation", step=step, updates=epoch)
                report, trace = evaluator.run(params)
                report.update(
                    step=step,
                    updates=epoch,
                    checkpoint=str(path.relative_to(rec.path)),
                    split="dev",
                )
                save_report(rec.path / "eval" / f"step-{step:010d}.json", report)
                replay = select_replays(trace, report)
                replay["metrics"]["training_step"] = np.full_like(replay["reward"], step)
                export_rollout(evaluation.sim, rec.path / "rollouts" / f"step-{step:010d}", replay)
                score = development_score(task.task, report)
                if score > best_score:
                    best_score, best = score, report
                    save_report(
                        rec.path / "checkpoints/best.json",
                        dict(path=path.name, step=step, score=list(score), selection_split="dev"),
                    )
                rec.log(
                    step,
                    {
                        "eval/return": report["return_mean"],
                        "eval/rmse_all_m": report["rmse_all_mean"],
                        "eval/last_second_rmse_m": report["last_second_rmse_mean_m"],
                        "eval/completion_rate": report["completion_rate"],
                        "eval/quality_passed": float(report["quality_passed"]),
                    },
                )
                print(
                    json.dumps(
                        {
                            "run_id": run_id,
                            "updates": epoch,
                            "steps": step,
                            "completed": report["completed"],
                            "rmse_m": report["rmse_all_mean"],
                            "settled_rmse_m": report["last_second_rmse_mean_m"],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

            snapshot(runner)
            for epoch in range(int(runner.epoch_idx) + 1, budget + 1):
                rec.phase(
                    "compiling" if epoch == initial_updates + 1 else "training",
                    step=(epoch - 1) * transitions,
                    updates=epoch - 1,
                )
                tic = time.monotonic()
                runner, metrics = update(runner)
                scalar = {k: float(v) for k, v in metrics.items()}
                timings.append(time.monotonic() - tic)
                if not all(np.isfinite(v) for v in scalar.values()):
                    save_training_state(
                        rec.path / "training-state/failed-update.pkl", runner, config
                    )
                    raise FloatingPointError(f"Nonfinite LOTF update {epoch}: {scalar}")
                rec.log(
                    epoch * transitions,
                    {
                        **{"training/" + k: v for k, v in scalar.items()},
                        "training/update_seconds": timings[-1],
                        "training/updates": epoch,
                    },
                )
                if epoch in milestones:
                    snapshot(runner)
                if epoch % 10 == 0:
                    print(
                        json.dumps(
                            {
                                "run_id": run_id,
                                "updates": epoch,
                                "loss": scalar["loss"],
                                "update_seconds": timings[-1],
                            }
                        ),
                        flush=True,
                    )
                if time.monotonic() - start > config["training"]["max_wall_seconds"]:
                    save_training_state(
                        rec.path / "training-state/budget-exhausted.pkl", runner, config
                    )
                    raise TimeoutError("Declared LOTF wall-clock budget reached")
            final = jax.tree.map(np.asarray, runner.train_state.params)
            delta = float(
                np.sqrt(
                    sum(
                        np.sum((a - b) ** 2)
                        for a, b in zip(jax.tree.leaves(initial), jax.tree.leaves(final))
                    )
                )
            )
            result = dict(
                actual_steps=budget * transitions,
                initial_updates=initial_updates,
                new_training_interactions=(budget - initial_updates) * transitions,
                updates=budget,
                actor_parameter_delta_l2=delta,
                full_budget_completed=True,
                engineer_passed=np.isfinite(delta).item() and delta > 0,
                quality_passed=best["quality_passed"],
                best_development_result=best,
                compile_and_first_update_seconds=timings[0] if timings else 0.0,
                net_update_seconds=sum(timings[1:]),
                elapsed_seconds=time.monotonic() - start,
                source_commit="cba6e5370773ace8a08107f02810eecabf16c793",
                scope="high-fidelity forward / analytical backward / native BPTT; learned residual and online adaptation disabled",
            )
            rec.finish("completed", **result)
            return result
        except BaseException as exc:
            traceback.print_exc()
            rec.finish("failed", error=repr(exc), elapsed_seconds=time.monotonic() - start)
            raise
        finally:
            if task is not None:
                task.close()
            if evaluation is not None:
                evaluation.close()
