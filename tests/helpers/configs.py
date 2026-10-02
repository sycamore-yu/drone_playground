"""Reusable configuration builders for tests."""

from drone_playground.composition import compose_experiment


def bodyrates_config(environment="hovering", forward="lotf_simplified"):
    return compose_experiment(
        'control/bptt',
        environment,
        [
            "dynamics@env.dynamics=" + forward,
            "action/controller@env.action.controller=bodyrates",
            "env.action.command=thrust_bodyrates",
            "runtime.action_delay_ms=null",
            "runtime.device=cpu",
            "env.task.duration=0.04",
        ],
    )


def differentiable_pointcloud_config():
    return compose_experiment("navigation/differentiable_pointcloud")
