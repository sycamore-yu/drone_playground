"""Public recurrent recipes reach their real update, selection and saved state."""

import pytest

from drone_playground.app import run_experiment
from drone_playground.configuration import compose_experiment


@pytest.mark.parametrize(
    "recipe",
    [
        "papers/differentiable_pointcloud",
        "navigation/differentiable_pointcloud",
        "papers/depth_diffphysics",
        "control/differentiable_pointcloud_hovering",
    ],
)
def test_recurrent_training_entry_completes_one_real_update(recipe, tmp_path):
    overrides = [
        "runtime.device=cpu",
        "training.num_envs=1",
        "training.policy_updates=1",
        "algorithm.horizon_length=2",
        "training.num_evals=2",
        "network.hidden_size=8",
        "training.checkpoint_eval_episodes=1",
        "++training.checkpoint_eval_envs=1",
        "training.checkpoint_eval_seed_start=20000",
        "env.task.duration=0.2",
        "evaluation.record_replays=false",
    ]
    if recipe != "papers/depth_diffphysics":
        overrides += [
            "env.sensor.azimuth_count=8",
            "env.sensor.elevation_count=3",
            "network.encoder.point_channels=[8,8]",
        ]
    config = compose_experiment(recipe, overrides=overrides)
    config["evaluation"]["protocol"] = None
    config["training"]["checkpoint_eval_metric"] = None
    config["training"]["checkpoint_eval_initial_conditions"] = None
    result = run_experiment(config, tmp_path, "recurrent")
    assert result["full_budget_completed"]
    assert result["actual_updates"] == 1
    assert result.get("parameter_delta_l2", result.get("actor_parameter_delta_l2")) > 0
    assert list(tmp_path.glob("results/runs/*/*/recurrent/training-state/*.pkl"))
    assert not list(tmp_path.rglob("*.mj_unroll"))
