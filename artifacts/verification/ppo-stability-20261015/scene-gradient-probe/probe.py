"""Separate reproducible scene-gradient conflict from conditional sampling variability."""

import inspect
import json
import subprocess
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree
from omegaconf import OmegaConf

from drone_playground.learning import checkpoint, ppo
from drone_playground.learning.trainer import Trainer
from drone_playground.simulation.evaluation import create_environment, create_method
from drone_playground.simulation.records import atomic_json


def cosine(left, right):
    """Compare nonzero row vectors without inferring a statistical population effect."""
    numerator = left @ right.T
    denominator = np.linalg.norm(left, axis=1)[:, None] * np.linalg.norm(right, axis=1)
    return numerator / np.maximum(denominator, 1e-30)


def main():
    """Measure two copied behavior rollouts; never update or evaluate the source learner."""
    np.testing.assert_allclose(cosine(np.eye(2), np.eye(2)), np.eye(2))
    np.testing.assert_allclose(cosine(np.array([[1.0, 0]]), np.array([[-2.0, 0]])), -1)
    output = Path(__file__).resolve().parent
    design = json.loads(
        (output.parent / "candidate-configs/depth-scene-gradient-probe-design.json").read_text()
    )
    checkout = Path(inspect.getfile(Trainer)).resolve().parents[3]
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=checkout, check=True, capture_output=True, text=True
    ).stdout.strip()
    assert revision == design["runtime_commit"]
    assert not subprocess.run(
        ["git", "status", "--porcelain", "--", "src"],
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    source = Path(design["source_checkpoint"])
    metadata = json.loads((source / "metadata.json").read_text())
    control = json.loads(Path(design["control_evidence"]).read_text())
    prior = next(row for row in control["checks"] if row["update"] == 180)
    assert prior["successes"] == 150 and prior["passed"]
    config = OmegaConf.create(metadata["config"]["experiment"])
    scenes = list(config.learning.scenes)
    assert scenes == ["S01", "S02", "S03", "D01", "D02", "D03", "S06", "D06"]
    started = time.monotonic()
    trainer = Trainer(
        create_environment(config, scene=scenes),
        kind="depth",
        algorithm="ppo",
        seed=config.seed,
        config=OmegaConf.to_container(config.learning.options, resolve=True),
        actor_config=OmegaConf.to_container(config.method.actor, resolve=True),
        method=create_method(config),
        loss=config.learning.loss,
        backward_model=config.learning.backward_model,
        backward_options=OmegaConf.to_container(config.learning.backward_options),
    )
    assert trainer.resolved_config == metadata["config"]["learning"]
    assert not trainer.config["shared_visual_encoder"]
    state, _ = checkpoint.load_model_state(source, trainer.initialize())
    assert int(state.updates) == 180 and int(state.interactions) == 5898240
    variables = ppo.policy_variables(state)
    flat_variables, unravel = ravel_pytree(variables)
    groups = np.arange(64).reshape(16, 4)
    initial_rng = np.asarray(jax.random.key_data(state.rng))
    result = {
        "status": "running",
        "source_checkpoint": str(source),
        "runtime_commit": revision,
        "runner_git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=output.parents[3],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "setup_seconds": time.monotonic() - started,
        "scenes": scenes,
        "sampling_interactions": 0,
        "new_sampling_interactions": 0,
        "gradient_numerics": "highest matmul precision; deterministic compiler options "
        "scoped to gradient/loss; collection unchanged",
        "replicates": [],
        "scope": design["scope"],
    }
    atomic_json(output / "report.json", result)
    collect = jax.jit(lambda current: ppo.collect_rollout(trainer, current))
    all_scene_gradients = []
    all_actor_gradients = []

    def loss(parameters, indices, rollout, advantages, noise):
        with jax.default_matmul_precision("highest"):
            values = unravel(parameters)
            data = rollout
            memory = state.recurrent_memory
            _, (mean, auxiliary) = ppo.recurrent_policy(
                trainer,
                values["params"],
                data.observations,
                memory,
                data.done,
            )
            log_prob = trainer.method.log_probability(
                data.pre_tanh,
                mean,
                values["log_std"],
                data.observations,
            )
            ratio = jnp.exp(log_prob - data.log_prob)
            clipped = jnp.clip(
                ratio,
                1 - trainer.config["clip_epsilon"],
                1 + trainer.config["clip_epsilon"],
            )
            surrogate = -jnp.minimum(
                ratio * advantages,
                clipped * advantages,
            )
            sample = mean + jnp.exp(values["log_std"]) * noise
            negative_entropy = trainer.method.log_probability(
                sample,
                mean,
                values["log_std"],
                data.observations,
            )
            auxiliary_loss = trainer.objective.auxiliary_cost(auxiliary, data.observations)
            costs = surrogate + trainer.config["entropy_weight"] * negative_entropy + auxiliary_loss
            return jnp.mean(costs[:, indices])

    compiler = {"xla_gpu_exclude_nondeterministic_ops": True, "xla_gpu_autotune_level": 0}
    value_gradient = jax.jit(jax.value_and_grad(loss), compiler_options=compiler)
    value_only = jax.jit(loss, compiler_options=compiler)
    apply_actor = jax.jit(trainer._apply_actor)

    try:
        for replicate in range(2):
            before = time.monotonic()
            reused = replicate in design.get("reuse_saved_rollouts", [])
            if reused:
                with np.load(
                    output / f"replicate_{replicate}_rollout.npz", allow_pickle=False
                ) as data:
                    rollout = ppo.Rollout(
                        observations={
                            key[5:]: jnp.asarray(data[key])
                            for key in data.files
                            if key.startswith("obs__")
                        },
                        **{
                            key: jnp.asarray(data[key])
                            for key in (
                                "pre_tanh",
                                "log_prob",
                                "rewards",
                                "values",
                                "next_values",
                                "terminated",
                                "done",
                            )
                        },
                    )
                    advantages, noise = (
                        jnp.asarray(data["advantages"]),
                        jnp.asarray(data["entropy_noise"]),
                    )
                    completed_successes = (
                        int(data["completed_successes"]) if "completed_successes" in data else None
                    )
            else:
                rng = state.rng if replicate == 0 else jax.random.fold_in(state.rng, 810002)
                after, rollout, terms = collect(state.replace(rng=rng))
                jax.block_until_ready(rollout.rewards)
                advantages, _ = ppo.generalized_advantage_estimate(
                    rollout.rewards,
                    rollout.values,
                    rollout.next_values,
                    rollout.terminated,
                    rollout.done,
                    gamma=trainer.config["gamma"],
                    gae_lambda=trainer.config["gae_lambda"],
                )
                advantages = ppo.normalize_advantages(advantages)
                noise = jax.random.normal(
                    jax.random.fold_in(after.rng, 810003),
                    rollout.pre_tanh.shape,
                    dtype=rollout.pre_tanh.dtype,
                )
                completed_successes = int(terms["navigation_success_fraction"].sum())
                np.savez_compressed(
                    output / f"replicate_{replicate}_rollout.npz",
                    **{
                        f"obs__{key}": np.asarray(value)
                        for key, value in rollout.observations.items()
                    },
                    **{
                        key: np.asarray(getattr(rollout, key))
                        for key in (
                            "pre_tanh",
                            "log_prob",
                            "rewards",
                            "values",
                            "next_values",
                            "terminated",
                            "done",
                        )
                    },
                    advantages=np.asarray(advantages),
                    entropy_noise=np.asarray(noise),
                    completed_successes=completed_successes,
                )
                result["new_sampling_interactions"] += 512 * 64
            result["sampling_interactions"] += 512 * 64
            atomic_json(output / "report.json", result)

            before_losses, gradients, actor_gradients = [], [], []
            for indices in groups:
                value, gradient = value_gradient(
                    flat_variables, jnp.asarray(indices), rollout, advantages, noise
                )
                before_losses.append(float(value))
                gradients.append(np.asarray(gradient))
                actor_gradients.append(np.asarray(ravel_pytree(unravel(gradient)["params"])[0]))
            gradients = np.stack(gradients)
            scene_gradients = gradients.reshape(8, 2, -1).mean(axis=1)
            actor_gradients = np.stack(actor_gradients).reshape(8, 2, -1).mean(axis=1)
            full_loss, full_gradient = value_gradient(
                flat_variables, jnp.arange(64), rollout, advantages, noise
            )
            _, repeated_gradient = value_gradient(
                flat_variables, jnp.arange(64), rollout, advantages, noise
            )
            np.testing.assert_array_equal(full_gradient, repeated_gradient)
            relative_error = np.linalg.norm(
                gradients.mean(axis=0) - full_gradient
            ) / np.linalg.norm(full_gradient)
            np.savez_compressed(
                output / f"replicate_{replicate}_gradient_identity.npz",
                half_gradients=gradients,
                full_gradient=np.asarray(full_gradient),
                before_losses=before_losses,
                full_loss=np.asarray(full_loss),
            )
            assert relative_error < 5e-5, relative_error
            np.testing.assert_allclose(np.mean(before_losses), full_loss, atol=1e-6, rtol=1e-5)
            candidate = apply_actor(state, unravel(full_gradient))
            candidate_flat, _ = ravel_pytree(ppo.policy_variables(candidate))
            delta = np.asarray(candidate_flat - flat_variables)
            after_losses = [
                float(value_only(candidate_flat, jnp.asarray(i), rollout, advantages, noise))
                for i in groups
            ]
            norms = np.linalg.norm(scene_gradients, axis=1)
            assert np.isfinite(gradients).all() and (norms > 1e-10).all()
            half_cosines = np.diag(cosine(gradients[::2], gradients[1::2]))
            directional = scene_gradients @ delta
            actual = np.asarray(after_losses).reshape(8, 2).mean(axis=1) - np.asarray(
                before_losses
            ).reshape(8, 2).mean(axis=1)
            report = {
                "replicate": replicate,
                "gradient_norms": norms.tolist(),
                "scene_cosines": cosine(scene_gradients, scene_gradients).tolist(),
                "actor_only_scene_cosines": cosine(actor_gradients, actor_gradients).tolist(),
                "within_scene_half_cosines": half_cosines.tolist(),
                "full_gradient_reconstruction_relative_error": float(relative_error),
                "adam_delta_norm": float(np.linalg.norm(delta)),
                "full_gradient_adam_cosine": float(
                    cosine(np.asarray(full_gradient)[None], delta[None])[0, 0]
                ),
                "gradient_descent_normalized_directional_loss_change": (
                    -cosine(scene_gradients, np.asarray(full_gradient)[None])[:, 0]
                ).tolist(),
                "adam_normalized_directional_loss_change": (
                    directional / (norms * np.linalg.norm(delta))
                ).tolist(),
                "adam_predicted_loss_change": directional.tolist(),
                "adam_actual_loss_change": actual.tolist(),
                "reward": float(rollout.rewards.mean()),
                "completed_episodes": int(rollout.done.sum()),
                "completed_successes": completed_successes,
                "reused_saved_rollout": reused,
                "full_gradient_exact_repeat": True,
                "seconds": time.monotonic() - before,
            }
            np.savez_compressed(
                output / f"replicate_{replicate}_gradients.npz",
                half_gradients=gradients,
                actor_scene_gradients=actor_gradients,
                full_gradient=np.asarray(full_gradient),
                adam_delta=delta,
                before_losses=before_losses,
                after_losses=after_losses,
            )
            all_scene_gradients.append(scene_gradients)
            all_actor_gradients.append(actor_gradients)
            result["replicates"].append(report)
            atomic_json(output / "report.json", result)
            print(json.dumps(report), flush=True)
            assert int(state.updates) == 180 and int(state.interactions) == 5898240
            np.testing.assert_array_equal(jax.random.key_data(state.rng), initial_rng)
        cross = np.diag(cosine(*all_scene_gradients))
        eligible = []
        for i in range(8):
            for j in range(i + 1, 8):
                repeated = all(r["scene_cosines"][i][j] < -0.1 for r in result["replicates"])
                coherent = all(
                    cross[k] > 0.5
                    and all(r["within_scene_half_cosines"][k] > 0 for r in result["replicates"])
                    for k in (i, j)
                )
                harmed = [
                    scenes[k]
                    for k in (i, j)
                    if scenes[k] in {"S01", "D03"}
                    and all(
                        r["adam_normalized_directional_loss_change"][k] > 0.1
                        and r["gradient_descent_normalized_directional_loss_change"][k] > 0.1
                        and r["adam_actual_loss_change"][k] > 0
                        for r in result["replicates"]
                    )
                ]
                if repeated and coherent and harmed:
                    eligible.append({"pair": [scenes[i], scenes[j]], "harmed_scenes": harmed})
        result.update(
            status="completed",
            total_wall_seconds=time.monotonic() - started,
            cross_replicate_scene_cosines=cross.tolist(),
            cross_replicate_actor_only_cosines=np.diag(cosine(*all_actor_gradients)).tolist(),
            eligible_pairs=eligible,
            pcgrad_selection_gate_passed=bool(eligible),
            total_sampling_interactions_including_failed_attempt=result["sampling_interactions"]
            + design.get("failed_attempt_interactions", 0),
        )
        atomic_json(output / "report.json", result)
        print(
            json.dumps(
                {
                    key: result[key]
                    for key in ("status", "cross_replicate_scene_cosines", "eligible_pairs")
                }
            ),
            flush=True,
        )
    except BaseException as error:
        result.update(status="interrupted", error=repr(error))
        atomic_json(output / "report.json", result)
        raise
    finally:
        trainer.env.sim.close()


if __name__ == "__main__":
    main()
