"""Read retained v3 configuration identities after source/layout-only migrations.

Weight files and their sidecars are immutable inputs. The explicit names below
translate recorded implementations; no current experiment defaults are merged
into a saved policy, and all numeric settings remain those in its checkpoint.
"""

import copy

_TARGETS = {
    "environments.scenes.EmptyScene": "environments.scenes.empty.EmptyScene",
    "environments.scenes.LSYScene": "environments.scenes.racing.RacingScene",
    "environments.scenes.catalog.NavigationScene": "environments.scenes.catalog.CatalogNavigationScene",
    "environments.scenes.pointcloud.PaperPrimitiveScene": "environments.scenes.primitive_sampling.PrimitiveScene",
    "environments.sensors.pointcloud.UniformMid360Lidar": "environments.sensors.lidar.UniformRayLidar",
    "environments.sensors.depth_flight.DepthFlightCamera": "environments.sensors.depth.PinholeDepthCamera",
    "environments.tasks.pointcloud.PaperObservation": "environments.observations.flight_state.FlightStateObservation",
    "environments.observations.TrackingObservation": "environments.observations.state.TrackingObservation",
    "environments.observations.NavigationObservation": "environments.observations.state.NavigationObservation",
    "environments.observations.NavigationSensorObservation": "environments.observations.state.NavigationSensorObservation",
    "methods.planners.reference.TrajectoryPlan": "environments.tasks.references.ReferenceGenerator",
    "evaluation.tracking.PolicyEvaluator": "evaluation.tracking.policy.PolicyEvaluator",
    "learning.objectives.TrackingObjective": "environments.tasks.rewards.TrackingObjective",
    "learning.objectives.NavigationObjective": "environments.tasks.rewards.NavigationObjective",
    "learning.objectives.pointcloud.PaperObjective": "learning.objectives.acceleration_flight.VelocityTrackingAvoidanceObjective",
    "learning.objectives.acceleration_flight.PaperObjective": "learning.objectives.acceleration_flight.VelocityTrackingAvoidanceObjective",
    "learning.objectives.pointcloud_navigation.PointCloudNavigationObjective": "learning.objectives.navigation_acceleration.PointCloudNavigationObjective",
}


def _moved_targets(value):
    if isinstance(value, list):
        return [_moved_targets(item) for item in value]
    if not isinstance(value, dict):
        return value
    result = {key: _moved_targets(item) for key, item in value.items()}
    for key in ("_target_", "trainer", "policy_evaluator", "evaluation_entrypoint"):
        target = result.get(key)
        if not isinstance(target, str) or not target.startswith("drone_playground."):
            continue
        suffix = target.removeprefix("drone_playground.")
        suffix = _TARGETS.get(suffix, suffix)
        if suffix.startswith("models."):
            suffix = "dynamics." + suffix.removeprefix("models.")
        elif suffix.startswith("execution.controllers."):
            suffix = "actions.controllers." + suffix.removeprefix("execution.controllers.")
        result[key] = "drone_playground." + suffix
    return result


def _move(mapping, old, new):
    if old not in mapping:
        return
    value = mapping.pop(old)
    if new in mapping and mapping[new] != value:
        raise ValueError(f"Checkpoint contains conflicting {old} and {new} values")
    mapping[new] = value


def require_current(config: dict) -> dict:
    value = config.get("components", config)
    if value.get("config_version") != 3:
        raise ValueError(
            "Artifact config_version must be 3; historical schemas are unsupported"
        )
    value = _moved_targets(copy.deepcopy(value))
    env = value.get("env", {})
    if "execution" in env:
        action = env.pop("execution")
        dynamics = action.pop("dynamics")
        for field, item in (("action", action), ("dynamics", dynamics)):
            if field in env and env[field] != item:
                raise ValueError(f"Checkpoint has conflicting env.{field} layouts")
            env[field] = item
    training = value.get("training", {})
    for old, new in (
        ("initial_condition_randomization", "reset_randomization"),
        ("development_envs", "checkpoint_eval_envs"),
        ("development_episodes", "checkpoint_eval_episodes"),
        ("development_seed_start", "checkpoint_eval_seed_start"),
        ("development_initial_conditions", "checkpoint_eval_initial_conditions"),
        ("development_metric", "checkpoint_eval_metric"),
    ):
        _move(training, old, new)
    if "evaluation" in value:
        value["evaluation"].setdefault("role", "eval")
    algorithm = value.get("algorithm", {})
    algorithm["name"] = {
        "pointcloud_bptt": "recurrent_bptt",
        "pointcloud_navigation_bptt": "recurrent_navigation_bptt",
        "depth_navigation_bptt": "recurrent_navigation_bptt",
    }.get(algorithm.get("name"), algorithm.get("name"))
    # This was a public method rename, not a transfer to another trained policy.
    method = value.get("method", {})
    if method.get("name") == "pointcloud_flight":
        method["name"] = "differentiable_pointcloud"
    return value
