"""Inspect missing per-scene training outcomes at the paired update64 states."""

import json
import subprocess
import time
from functools import partial
from pathlib import Path

import jax
import numpy as np
from omegaconf import OmegaConf

from drone_playground.learning.checkpoint import load_model_state
from drone_playground.learning.ppo import collect_rollout, generalized_advantage_estimate
from drone_playground.learning.trainer import Trainer
from drone_playground.simulation.evaluation import create_environment, create_method
from drone_playground.simulation.records import atomic_json

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = Path(__file__).resolve().parent / "training-rollout"
SOURCES = {
    "random_starts": "results/fixed_scenes_parallel/depth_ppo_s0/checkpoints/model_000064",
    "nominal_starts": "results/fixed_scenes_parallel/depth_ppo_s0_nominal_starts/model_000064",
}
EVENTS = ("success", "collision", "out_of_bounds", "timeout", "numerical_failure")


def main():
    """Collect one native stochastic rollout per frozen learner, without optimization."""
    OUTPUT.mkdir(exist_ok=False)
    report = {
        "status": "running",
        "scope": "Two frozen training-state probes, one512x64 rollout each; no model update",
        "question": "Does the aggregate training success conceal different S03 outcomes?",
        "decision": "S03 failures despite successes elsewhere support a scene-specific training "
        "gap in this sample. S03 successes reject a total absence of training success here; "
        "zero completed S03 episodes leave that question unresolved.",
        "limits": "One rollout per source does not estimate converged success rates or a "
        "unique cause. Initial states differ because the reset intervention already acted. "
        "The critic error is against bootstrapped targets, not complete Monte Carlo returns. "
        "Separate compilation can differ from the original update65 rollout numerically.",
        "runtime_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.strip(),
        "models": {},
    }
    atomic_json(OUTPUT / "report.json", report)
    for name, relative in SOURCES.items():
        source = ROOT / relative
        metadata = json.loads((source / "metadata.json").read_text())
        config = OmegaConf.create(metadata["config"]["experiment"])
        assert config.learning.scene_sampling == "parallel"
        assert config.learning.options.horizon == 512 and config.simulation.num_envs == 64
        trainer = Trainer(
            create_environment(config, scene=list(config.learning.scenes)),
            kind=config.method.actor.kind,
            algorithm="ppo",
            seed=config.seed,
            config=OmegaConf.to_container(config.learning.options, resolve=True),
            actor_config=OmegaConf.to_container(config.method.actor, resolve=True),
            method=create_method(config),
            loss=config.learning.loss,
            backward_model=config.learning.backward_model,
            backward_options=OmegaConf.to_container(config.learning.backward_options),
        )
        try:
            assert trainer.resolved_config == metadata["config"]["learning"]
            state, _ = load_model_state(source, trainer.initialize())
            assert int(state.updates) == 64 and int(state.interactions) == 2097152
            before_position = np.asarray(state.env_state.physics.states.pos[:, 0])
            before_time = np.asarray(state.env_state.time)

            @partial(jax.jit, static_argnames=("trainer",))
            def probe(initial, trainer=trainer):
                following, rollout, terms = collect_rollout(trainer, initial)
                _, targets = generalized_advantage_estimate(
                    rollout.rewards,
                    rollout.values,
                    rollout.next_values,
                    rollout.terminated,
                    rollout.done,
                    gamma=trainer.config["gamma"],
                    gae_lambda=trainer.config["gae_lambda"],
                )
                return {
                    **{event: terms[f"navigation_{event}_fraction"] for event in EVENTS},
                    "done": rollout.done,
                    "progress_metres": terms["navigation_progress_metres"],
                    "remaining_metres": terms["navigation_remaining_metres"],
                    "rewards": rollout.rewards,
                    "values": rollout.values,
                    "targets": targets,
                    "after_position": following.env_state.physics.states.pos[:, 0],
                    "after_time": following.env_state.time,
                }

            started = time.monotonic()
            data = jax.device_get(probe(state))
            assert data["done"].shape == (512, 64)
            assert all(np.isfinite(value).all() for value in data.values())
            assert np.array_equal(sum(data[event] for event in EVENTS), data["done"])
            data.update(before_position=before_position, before_time=before_time)
            np.savez_compressed(OUTPUT / f"{name}.npz", **data)
            scenes = {}
            for scene in trainer.env.scene.scenes:
                indices = [
                    i for i in range(64) if trainer.env.scene.scene_for_world(i).name == scene.name
                ]
                assert len(indices) == 8
                scenes[scene.name] = {
                    "world_indices": indices,
                    "completed_episodes": int(data["done"][:, indices].sum()),
                    "events": {event: int(data[event][:, indices].sum()) for event in EVENTS},
                    "mean_remaining_metres": float(data["remaining_metres"][:, indices].mean()),
                    "summed_progress_metres": float(data["progress_metres"][:, indices].sum()),
                    "mean_value_prediction": float(data["values"][:, indices].mean()),
                    "mean_bootstrapped_target": float(data["targets"][:, indices].mean()),
                    "bootstrapped_target_rmse": float(
                        np.sqrt(
                            np.mean((data["values"][:, indices] - data["targets"][:, indices]) ** 2)
                        )
                    ),
                    "before_position_metres": before_position[indices].tolist(),
                    "before_time_seconds": before_time[indices].tolist(),
                }
            report["models"][name] = {
                "source_checkpoint": str(source),
                "source_provenance_reused": metadata["provenance"],
                "elapsed_probe_seconds": time.monotonic() - started,
                "additional_simulated_interactions": 32768,
                "scenes": scenes,
            }
            atomic_json(OUTPUT / "report.json", report)
            print(name, {k: v["events"] for k, v in scenes.items()}, flush=True)
        except BaseException as error:
            report.update(status="failed", error=repr(error))
            atomic_json(OUTPUT / "report.json", report)
            raise
        finally:
            trainer.env.sim.close()
    report["status"] = "completed"
    atomic_json(OUTPUT / "report.json", report)


if __name__ == "__main__":
    main()
