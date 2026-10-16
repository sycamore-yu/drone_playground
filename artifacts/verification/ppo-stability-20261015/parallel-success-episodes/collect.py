"""Collect complete, fresh training episodes for a frozen self-policy cloning probe."""

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

from drone_playground.learning import checkpoint, ppo
from drone_playground.learning.trainer import Trainer
from drone_playground.simulation.evaluation import create_environment, create_method
from drone_playground.simulation.networks import reset_actor_memory
from drone_playground.simulation.records import atomic_json, source_identity
from drone_playground.simulation.tasks import Event

PRIMARY = ("S01", "S02", "S03", "D01", "D02", "D03")


def collect(source, output):
    """Keep whole histories; no evaluation cases or historical hidden states enter data."""
    started = time.monotonic()
    output.mkdir(parents=True, exist_ok=False)
    metadata = json.loads((source / "metadata.json").read_text())
    runtime = source_identity()
    assert runtime["source_sha256"] == metadata["provenance"]["source_sha256"]
    config = OmegaConf.create(metadata["config"]["experiment"])
    scenes = list(config.learning.scenes)
    assert config.learning.algorithm == "ppo" and config.method.actor.kind == "depth"
    assert config.learning.scene_sampling == "parallel" and config.simulation.num_envs == 64
    assert scenes == [*PRIMARY, "S06", "D06"] and config.task.duration == 300
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
        backward_options=OmegaConf.to_container(config.learning.backward_options, resolve=True),
    )
    try:
        assert trainer.resolved_config == metadata["config"]["learning"]
        assert {s.name: s.geometry_hash for s in trainer.env.scene.scenes} == (
            metadata["config"]["training_sampling"]["geometry_sha256"]
        )
        state, _ = checkpoint.load_model_state(source, trainer.initialize())
        assert int(state.updates) == 176 and int(state.interactions) == 5767168
        reset_key = jax.random.fold_in(state.rng, 840101)
        progress = int(state.interactions) / config.learning.max_interactions
        # Complete nominal-start paths cover early obstacles omitted by wide-start suffixes.
        initial = trainer.env.reset(reset_key, randomize_position=False, sampling_progress=progress)
        zero_memory = trainer.actor.initialize_memory(trainer.env.num_envs)
        assert np.count_nonzero(np.asarray(zero_memory)) == 0
        carry = (initial, zero_memory)
        setup_seconds = time.monotonic() - started

        @jax.jit
        def chunk(carry):
            def step(carry, _):
                before, memory = carry
                observation = trainer.env.observe(before)
                raw, memory, _ = trainer.actor.apply(state.params, observation, memory)
                action = trainer.method.deterministic(raw)
                after = trainer.method.execute(trainer.env, before, action)
                records = {
                    **{f"obs__{key}": observation[key] for key in ("state", "depth", "depth_mask")},
                    "teacher_mean": raw,
                    "active": ~before.done,
                    "done": after.done,
                    "event": after.task.event,
                    "time": after.time,
                    "position": after.physics.states.pos[:, 0],
                    "action": action,
                }
                return (after, reset_actor_memory(memory, after.done)), records

            return jax.lax.scan(step, carry, None, length=25)

        chunks = []
        collection_started = time.monotonic()
        for index in range(int(np.ceil(trainer.env.task.duration / trainer.env.dt / 25)) + 1):
            carry, records = chunk(carry)
            chunks.append(jax.device_get(records))
            if index % 20 == 0:
                print(
                    "collected_slots",
                    (index + 1) * 25 * 64,
                    "completed",
                    int(carry[0].done.sum()),
                    flush=True,
                )
            if np.asarray(carry[0].done).all():
                break
        assert np.asarray(carry[0].done).all()
        collection_seconds = time.monotonic() - collection_started
        traces = {key: np.concatenate([row[key] for row in chunks]) for key in chunks[0]}
        assert all(np.isfinite(value).all() for value in traces.values())
        lengths = traces["active"].sum(axis=0).astype(int)
        assert (lengths <= int(np.ceil(trainer.env.task.duration / trainer.env.dt)) + 1).all()
        events = np.asarray(carry[0].task.event)
        episodes = []
        for world, length in enumerate(lengths):
            assert length > 0 and traces["active"][:length, world].all()
            assert not traces["active"][length:, world].any()
            assert not traces["done"][: length - 1, world].any()
            assert traces["done"][length - 1 :, world].all()
            assert traces["event"][length - 1, world] == events[world]
            episodes.append(
                {
                    "world": world,
                    "scene": scenes[world // 8],
                    "steps": int(length),
                    "event": Event(int(events[world])).name,
                }
            )
        observation = {
            key[5:]: jnp.asarray(value) for key, value in traces.items() if key.startswith("obs__")
        }
        _, (recomputed, _) = jax.jit(
            lambda params: ppo.recurrent_policy(
                trainer, params, observation, zero_memory, jnp.asarray(traces["done"])
            )
        )(state.params)
        errors = np.abs(np.asarray(recomputed) - traces["teacher_mean"])
        max_error = float(errors[traces["active"]].max())
        success_worlds = np.flatnonzero(events == Event.SUCCESS)
        counts = {
            scene: sum(e["event"] == "SUCCESS" and e["scene"] == scene for e in episodes)
            for scene in scenes
        }
        gate = max_error < 1e-4 and all(counts[scene] >= 4 for scene in PRIMARY)
        retained_steps = int(lengths[success_worlds].max(initial=0))
        data = {
            key: value[:retained_steps, success_worlds]
            for key, value in traces.items()
            if key.startswith("obs__") or key in ("teacher_mean", "active", "done")
        }
        data.update(
            teacher_log_std=np.asarray(state.log_std),
            source_world=success_worlds,
            source_scene_index=success_worlds // 8,
            lengths=lengths[success_worlds],
        )
        np.savez_compressed(output / "corpus.npz", **data)
        np.savez_compressed(
            output / "collection-traces.npz",
            initial_position=np.asarray(initial.physics.states.pos[:, 0]),
            **{
                key: value
                for key, value in traces.items()
                if key in ("active", "done", "event", "time", "position", "action")
            },
        )
        result = {
            "status": "completed",
            "collection_gate_passed": gate,
            "source_checkpoint": str(source.resolve()),
            "source_provenance_reused": metadata["provenance"],
            "runtime_provenance": runtime,
            "actor": trainer.actor.specification,
            "scenes": scenes,
            "worlds_per_scene": 8,
            "reset_key": np.asarray(jax.random.key_data(reset_key)).tolist(),
            "collection_randomize_position": False,
            "native_ppo_randomize_position": trainer.config["randomize_navigation_start"],
            "sampling_progress": progress,
            "episodes": episodes,
            "successful_episodes_per_scene": counts,
            "active_interactions": int(lengths.sum()),
            "allocated_simulation_slots": len(traces["active"]) * 64,
            "corpus_padded_steps": retained_steps,
            "source_policy_recompute_max_error": max_error,
            "setup_seconds": setup_seconds,
            "collection_seconds": collection_seconds,
            "total_wall_seconds": time.monotonic() - started,
            "scope": "Fresh nominal-start training demonstrations with independent RNG; "
            "this auxiliary data distribution differs from the native wide-start PPO sampler. "
            "Deterministic source-policy complete histories "
            "start at zero recurrent memory. Only successful episodes enter the corpus. "
            "No PPO update, off-policy importance ratio, critic target, or evaluation data.",
        }
        atomic_json(output / "corpus.json", result)
        print(
            json.dumps(
                {
                    key: result[key]
                    for key in (
                        "collection_gate_passed",
                        "successful_episodes_per_scene",
                        "active_interactions",
                        "source_policy_recompute_max_error",
                        "total_wall_seconds",
                    )
                }
            ),
            flush=True,
        )
    finally:
        trainer.env.sim.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    collect(args.checkpoint, args.output)
