"""A small real update reaches the public recorder, checkpoint and frozen evaluator."""

import json
from pathlib import Path

import pytest

from drone_playground.app import resolve_checkpoint_execution
from drone_playground.composition import compose_experiment, run_experiment


@pytest.mark.parametrize("algorithm", ["ppo", "bptt", "shac", "dva"])
def test_public_train_checkpoint_evaluate_without_replays(algorithm, tmp_path):
    recipe = "papers/dva" if algorithm == "dva" else "control/" + algorithm
    overrides = [
        "runtime.device=cpu",
        "runtime.action_delay_ms=null",
        "training.num_envs=2",
        "training.num_evals=2",
        "training.policy_updates=1",
        "training.num_timesteps=4",
        "training.checkpoint_eval_episodes=2",
        "env.task.duration=0.08",
        "network.hidden_sizes=[8,8]",
        "evaluation.record_replays=false",
        "training.publish_live=false",
    ]
    if algorithm == "ppo":
        overrides += [
            "algorithm.unroll_length=2",
            "algorithm.batch_size=2",
            "algorithm.num_minibatches=1",
            "algorithm.num_updates_per_batch=1",
        ]
    else:
        overrides += ["algorithm.horizon_length=2"]
    if algorithm in ("shac", "dva"):
        overrides += ["algorithm.critic_updates=1"]
    if algorithm == "dva":
        overrides += ["env.sensor.stride=15"]
    config = compose_experiment(recipe, overrides=overrides)
    result = run_experiment(config, tmp_path, "train")
    assert result["actual_steps"] == 4
    assert result["actor_parameter_delta_l2"] > 0
    snapshots = sorted(tmp_path.glob("results/runs/*/*/train/checkpoints/step-*.pkl"))
    assert len(snapshots) == 2
    checkpoint = snapshots[-1]
    sidecar = json.loads(checkpoint.with_suffix(".json").read_text())
    assert sidecar["observation_spec"]["shape"] == [sidecar["observation_size"]]
    requested = compose_experiment(overrides=["runtime.device=cpu", "mode=eval"])
    requested["checkpoint"] = str(checkpoint)
    requested["evaluation"]["episodes"] = 2
    resolved = resolve_checkpoint_execution(
        requested, ["evaluation.episodes=2", "runtime.device=cpu"]
    )
    assert resolved["algorithm"]["name"] == algorithm
    report = run_experiment(resolved, tmp_path, "eval")
    assert report["parameters_frozen"]
    assert not list(tmp_path.rglob("*.mj_unroll"))
    assert Path(checkpoint).read_bytes()
