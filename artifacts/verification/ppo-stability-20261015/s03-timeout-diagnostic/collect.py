"""Read two existing learner states and record the missing S03 timeout trajectories."""

import json
from pathlib import Path

from omegaconf import OmegaConf

from drone_playground.learning.checkpoint import load_model_state
from drone_playground.learning.trainer import Trainer
from drone_playground.simulation.evaluation import create_environment, create_method, evaluate
from drone_playground.simulation.records import atomic_json

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = Path(__file__).resolve().parent
SOURCES = {
    "altitude": "results/fixed_scenes_altitude_pd/depth_ppo_s0/checkpoints/model_000112",
    "parallel": "results/fixed_scenes_parallel/depth_ppo_s0/checkpoints/model_000048",
}


def main():
    """Preserve source state and use the original eight diagnostic cases."""
    for label, relative in SOURCES.items():
        source = ROOT / relative
        metadata = json.loads((source / "metadata.json").read_text())
        config = OmegaConf.create(metadata["config"]["experiment"])
        parallel = config.learning.get("scene_sampling", "sequential") == "parallel"
        scene = (
            list(config.learning.scenes)
            if parallel
            else config.learning.scenes[metadata["provenance"]["training_sampling"]["scene_index"]]
        )
        trainer = Trainer(
            create_environment(config, scene=scene),
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
        assert trainer.resolved_config == metadata["config"]["learning"]
        state, _ = load_model_state(source, trainer.initialize())
        assert int(state.updates) == metadata["provenance"]["updates"]
        directory = OUTPUT / f"{label}_from_state"
        directory.mkdir(exist_ok=False)
        atomic_json(
            directory / "source.json",
            {
                "checkpoint": str(source),
                "source_provenance_reused": metadata["provenance"],
                "scope": "Frozen parameters restored through the existing full-state loader",
            },
        )
        config.mode = "checkpoint_eval"
        config.benchmark.protocol = "diagnostic"
        config.benchmark.scenes = ["S03"]
        config.benchmark.episodes = 8
        config.benchmark.seed_base = 1_000_000
        config.benchmark.record_trajectories = True
        config.benchmark.record_replay = False
        OmegaConf.update(config, "screening.progress_aware", True, force_add=True)
        try:
            report = evaluate(
                config,
                directory,
                actor=trainer.actor,
                parameters=state.params,
                checkpoint_config=metadata["config"],
            )
            assert report["complete"] and report["reports"]["S03"]["episodes"] == 8
            print(
                json.dumps({"candidate": label, "report": str(directory / "report.json")}),
                flush=True,
            )
        finally:
            trainer.env.sim.close()


if __name__ == "__main__":
    main()
