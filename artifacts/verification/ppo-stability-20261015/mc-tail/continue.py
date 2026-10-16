"""Run the registered eight-update complete-return pilot from the full u176 state."""

import copy
import json
import time
from collections import Counter
from pathlib import Path

import jax
import numpy as np
import optax
from omegaconf import OmegaConf

from drone_playground.learning import checkpoint
from drone_playground.learning.trainer import Trainer
from drone_playground.simulation.evaluation import create_environment, create_method, evaluate
from drone_playground.simulation.records import atomic_json, source_identity
from tools.ppo_stability_window import SCENES, quality, read_cases, reuse_baseline


def main():
    """Keep prefix samples and optimizer passes matched; charge all extra simulation."""
    evidence = Path(__file__).resolve().parents[1]
    design = json.loads(
        (evidence / "candidate-configs/depth-parallel-mc-tail-design.json").read_text()
    )
    source, output = Path(design["source_checkpoint"]), Path(design["output"])
    metadata = json.loads((source / "metadata.json").read_text())
    control = json.loads(Path(design["control_report"]).read_text())
    baseline = reuse_baseline(control, source, metadata["provenance"])
    original = json.loads((source / "report.json").read_text())
    config_data = copy.deepcopy(metadata["config"])
    config_data["learning"]["ppo_monte_carlo_returns"] = True
    config_data["experiment"]["learning"]["options"]["ppo_monte_carlo_returns"] = True
    config = OmegaConf.create(config_data["experiment"])
    assert config.learning.scene_sampling == "parallel"
    assert config.learning.algorithm == "ppo" and config.method.actor.kind == "depth"
    assert original["passed"] and original["seed_base"] == control["evaluation_seed_base"]
    output.mkdir(exist_ok=False)
    setup_start = time.monotonic()
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
    assert trainer.resolved_config == config_data["learning"]
    assert {s.name: s.geometry_hash for s in trainer.env.scene.scenes} == (
        config_data["training_sampling"]["geometry_sha256"]
    )
    assert (
        trainer.env.scene.worlds_per_scene == config_data["training_sampling"]["worlds_per_scene"]
    )
    state, _ = checkpoint.load_model_state(source, trainer.initialize())
    origin_update, origin_interactions = int(state.updates), int(state.interactions)
    assert origin_update == control["origin_update"] == 176
    assert origin_interactions == control["origin_interactions"]
    batch = trainer.env.num_envs * trainer.config["horizon"]
    assert batch == 32768 and design["budget"]["ppo_updates"] == 8
    evaluation = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    evaluation.mode = "checkpoint_eval"
    evaluation.benchmark.protocol = "learning"
    evaluation.benchmark.scenes = list(SCENES)
    evaluation.benchmark.episodes = 25
    evaluation.benchmark.seed_base = original["seed_base"]
    evaluation.benchmark.record_replay = False
    evaluation.benchmark.record_trajectories = False
    result = {
        "status": "running",
        "diagnostic": True,
        "passed": False,
        "source_checkpoint": str(source),
        "source_config_sha256_reused": metadata["provenance"]["config_sha256"],
        "source_sha256_reused": metadata["provenance"]["source_sha256"],
        "runtime_provenance": source_identity(),
        "origin_update": origin_update,
        "origin_interactions": origin_interactions,
        "budget": design["budget"],
        "cost_comparison": design["cost_comparison"],
        "scope": design["scope"],
        "evaluation_seed_base": original["seed_base"],
        "prefix_interactions": 0,
        "tail_interactions": 0,
        "tail_allocated_slots": 0,
        "training_update_seconds": 0.0,
        "setup_seconds": time.monotonic() - setup_start,
        "checks": [],
    }
    atomic_json(output / "report.json", result)
    environments = {}
    started = time.monotonic()

    def save(directory):
        provenance = {
            **metadata["provenance"],
            **result["runtime_provenance"],
            "updates": int(state.updates),
            "interactions": int(state.interactions),
            "last_update": int(state.updates),
            "last_interactions": int(state.interactions),
            "status": result["status"],
            "diagnostic": True,
            "ppo_monte_carlo_returns": True,
            "pilot_source_checkpoint": str(source),
            "pilot_report": str(output / "report.json"),
            "consecutive_passes": 0,
            "last_evaluation": result["checks"][-1] if result["checks"] else original,
            "evaluation_index": original["evaluation_index"] + len(result["checks"]),
            "training_wall_seconds": metadata["provenance"]["training_wall_seconds"]
            + time.monotonic()
            - started,
        }
        provenance["source_config_sha256"] = provenance.pop("config_sha256")
        checkpoint.save_model(
            directory,
            state,
            config=config_data,
            provenance=provenance,
            kind=trainer.kind,
            actor_spec=trainer.actor.specification,
            export_policy=True,
        )

    try:
        for offset in range(1, 9):
            before = time.monotonic()
            candidate, metrics = trainer.update(state)
            measured = {key: float(value) for key, value in jax.device_get(metrics).items()}
            measured["actor_parameter_delta_norm"] = float(
                optax.tree.norm(jax.tree.map(lambda a, b: a - b, candidate.params, state.params))
            )
            assert np.isfinite(list(measured.values())).all()
            assert measured["actor_parameter_delta_norm"] > 0
            assert measured["monte_carlo_tail_unfinished"] == 0
            tail = int(measured["monte_carlo_tail_interactions"])
            assert int(candidate.interactions) == int(state.interactions) + batch + tail
            state = candidate
            assert int(state.updates) == origin_update + offset
            measured.update(update=int(state.updates), update_seconds=time.monotonic() - before)
            result["training_update_seconds"] += measured["update_seconds"]
            result["prefix_interactions"] += batch
            result["tail_interactions"] += tail
            result["tail_allocated_slots"] += int(measured["monte_carlo_tail_allocated_slots"])
            with (output / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(measured, allow_nan=False) + "\n")
            if offset == 1:
                reference = json.loads(
                    (evidence / "depth-parallel-postpass-entropy0-first-update.json").read_text()
                )
                excluded = {
                    "advantage_std",
                    "advantage_mean_pre_normalization",
                    "value_target_mean",
                    "value_target_std",
                    "critic_return_rmse_preupdate",
                }
                comparison = {
                    key: {"control": row["control"], "candidate": measured[key]}
                    for key, row in reference["pre_optimization_comparison"].items()
                    if key not in excluded
                }
                assert all(row["control"] == row["candidate"] for row in comparison.values())
                result["first_update_raw_rollout_comparison"] = comparison
            result.update(last_update=int(state.updates), last_interactions=int(state.interactions))
            print(json.dumps(measured), flush=True)
            if offset in design["budget"]["checks_after_updates"]:
                directory = output / f"model_{int(state.updates):06d}"
                save(directory)
                before = time.monotonic()
                evaluate(
                    evaluation,
                    directory,
                    actor=trainer.actor,
                    parameters=state.params,
                    evaluation_index=original["evaluation_index"] + len(result["checks"]) + 1,
                    environments=environments,
                )
                cases, _ = read_cases(directory)
                result["checks"].append(
                    {
                        "update": int(state.updates),
                        "model": str(directory),
                        "prefix_interactions": result["prefix_interactions"],
                        "tail_interactions": result["tail_interactions"],
                        "actual_additional_interactions": int(state.interactions)
                        - origin_interactions,
                        "evaluation_seconds": time.monotonic() - before,
                        "events": dict(Counter(cases.values())),
                        "control": next(
                            row for row in control["checks"] if row["update"] == int(state.updates)
                        ),
                        **quality(cases, baseline),
                    }
                )
                save(directory)
                print(json.dumps(result["checks"][-1]), flush=True)
            atomic_json(output / "report.json", result)
        result.update(
            status="completed",
            passed=all(row["passed"] for row in result["checks"]),
            actual_additional_interactions=int(state.interactions) - origin_interactions,
            final_checkpoint=result["checks"][-1]["model"],
            total_wall_seconds=time.monotonic() - started,
        )
        assert len(result["checks"]) == 2
        assert (
            result["prefix_interactions"] + result["tail_interactions"]
            == result["actual_additional_interactions"]
        )
        save(Path(result["final_checkpoint"]))
        atomic_json(output / "report.json", result)
    except BaseException as error:
        result.update(status="interrupted", error=repr(error), last_update=int(state.updates))
        save(output / f"model_{int(state.updates):06d}")
        atomic_json(output / "report.json", result)
        raise
    finally:
        for env in [trainer.env, *environments.values()]:
            if hasattr(env, "sim"):
                env.sim.close()


if __name__ == "__main__":
    main()
