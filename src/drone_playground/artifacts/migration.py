"""Explicit, copy-only migration of historical policy metadata.

The caller supplies a reviewed, resolved v4 configuration. Runtime loading never
imports this module or replaces missing fields with current experiment defaults.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pickle
from pathlib import Path

from drone_playground.artifacts.schema import require_current


def _legacy_target(target):
    """Translate former source locations only at this explicit migration boundary."""
    replacements = {
        "environments.observations.TrackingObservation": (
            "environments.observations.state.TrackingObservation"
        ),
        "environments.observations.NavigationObservation": (
            "environments.observations.state.NavigationObservation"
        ),
        "environments.observations.NavigationSensorObservation": (
            "environments.observations.state.NavigationSensorObservation"
        ),
        "environments.tasks.pointcloud.PaperObservation": (
            "environments.observations.flight_state.FlightStateObservation"
        ),
        "environments.sensors.pointcloud.UniformMid360Lidar": (
            "environments.sensors.lidar.UniformRayLidar"
        ),
        "environments.sensors.depth_flight.DepthFlightCamera": (
            "environments.sensors.depth.PinholeDepthCamera"
        ),
        "learning.objectives.TrackingObjective": "environments.tasks.rewards.TrackingObjective",
        "learning.objectives.NavigationObjective": "environments.tasks.rewards.NavigationObjective",
        "learning.objectives.navigation.MotionNavigationObjective": (
            "environments.tasks.rewards.MotionNavigationReward"
        ),
        "planning.minimum_jerk.MinimumJerkPlanning": "planning.minimum_jerk.MinimumJerkPlanner",
        "evaluation.tracking.policy.PolicyEvaluator": (
            "evaluation.tracking.policy.TrackingEvaluator"
        ),
        "integrations.ros1.RosPlanner": "integrations.ros1.planner.RosPlanner",
    }
    prefix = "drone_playground."
    if not target.startswith(prefix):
        return target
    suffix = target.removeprefix(prefix)
    suffix = replacements.get(suffix, suffix)
    for old, new in (
        ("control.controllers.crazyflow.", "control.controllers.attitude."),
        ("control.controllers.bodyrates.", "control.controllers.body_rate."),
        ("control.external_tracking.", "runtime.tracking."),
        ("networks.policies.", "learning.inference."),
        ("environments.environment.", "environments.factory."),
    ):
        if suffix.startswith(old):
            suffix = new + suffix[len(old) :]
    return prefix + suffix


def _normalized(value):
    if isinstance(value, str):
        return _legacy_target(value)
    if isinstance(value, list):
        return [_normalized(item) for item in value]
    if not isinstance(value, dict):
        return value
    result = {name: _normalized(item) for name, item in value.items()}
    if "_target_" in result:
        result["_target_"] = _legacy_target(result["_target_"])
    return result


def _check_contract(source, target):
    """Require the frozen network, sensor, observation and physical decoder to match."""
    old = source.get("components", source)
    if old.get("config_version") not in (3, 4):
        raise ValueError("Only explicitly selected v3/v4 policy artifacts can be migrated")
    env = old["env"]
    execution = env.get("execution", env.get("action", {}))
    dynamics = env.get("dynamics", execution.get("dynamics", {}))
    current = target["env"]
    observation = env.get("observation", env["task"].get("observation"))
    for name, previous, new in (
        ("sensor", env.get("sensor"), current["sensor"]),
        ("observation", observation, current["task"]["observation"]),
        ("network", old["network"], target["network"]),
        (
            "physical decoder",
            old["method"].get("physical_decoder"),
            target["method"].get("physical_decoder"),
        ),
    ):
        if _normalized(previous) != _normalized(new):
            raise ValueError(f"Explicit migration changes the frozen {name} contract")
    for name in ("forward", "drone"):
        if dynamics[name] != current["dynamics"][name]:
            raise ValueError(f"Explicit migration changes dynamics.{name}")
    if env["task"]["name"] != current["task"]["name"]:
        raise ValueError("Explicit migration changes task.name")
    for field in ("freq", "physics_freq"):
        previous = env[field] if field in env else env["task"].get(field)
        if previous != current.get(field):
            raise ValueError(f"Explicit migration changes environment {field}")
    if old["config_version"] == 4:
        previous = _normalized(copy.deepcopy(env))
        for field in ("freq", "physics_freq"):
            if field in previous["task"]:
                previous[field] = previous["task"].pop(field)
        for field in ("dynamics", "controller", "scene", "reference", "task"):
            if previous[field] != _normalized(current[field]):
                raise ValueError(f"Explicit migration changes the frozen {field} contract")
    kinds = {
        "attitude_thrust": "attitude",
        "thrust_bodyrates": "rates",
        "world_acceleration": "state",
        "velocity_yaw": "state",
    }
    if kinds.get(old["method"]["output"], old["method"]["output"]) != target["method"]["output"]:
        raise ValueError("Explicit migration changes physical output type")


class _HistoricalStateReader(pickle.Unpickler):
    """Resolve moved recurrent-state types in a user-selected historical artifact."""

    def find_class(self, module, name):
        if (
            module == "drone_playground.learning.algorithms.pointcloud_bptt"
            and name == "TrainingState"
        ):
            from drone_playground.learning.algorithms.recurrent_bptt import TrainingState

            return TrainingState
        return super().find_class(module, name)


def migrate_checkpoint(source, destination, config):
    """Write a new checkpoint and sidecar, keeping the source artifact unchanged."""
    import jax
    from brax.io import model

    from drone_playground.artifacts.reporting import tree_digest
    from drone_playground.environments.factory import build_observer, build_sensor

    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or destination.exists() or destination.with_suffix(".json").exists():
        raise FileExistsError(
            "Migration requires a new output path; original artifacts stay immutable"
        )
    metadata = json.loads(source.with_suffix(".json").read_text())
    content = source.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    if digest != metadata["sha256"]:
        raise ValueError("Source checkpoint digest does not match its sidecar")
    current = require_current(config)
    _check_contract(metadata["config"], current)
    result = copy.deepcopy(metadata)
    if metadata.get("family") in ("pointcloud_gru", "paper_pointcloud_gru", "recurrent_policy"):
        with source.open("rb") as stream:
            state = _HistoricalStateReader(stream).load()
        if tree_digest(state.params) != metadata["parameter_sha256"]:
            raise ValueError("Recurrent parameter digest differs before migration")
        content = pickle.dumps(state)
        restored = pickle.loads(content)
        for before, after in zip(jax.tree.leaves(state), jax.tree.leaves(restored), strict=True):
            import numpy as np

            np.testing.assert_array_equal(before, after)
        result["family"] = "recurrent_policy"
    elif metadata.get("policy_family") == "brax":
        params = model.load_params(str(source))
        if tree_digest(params) != metadata["parameter_sha256"]:
            raise ValueError("Policy parameter digest differs before migration")
        observer = build_observer(current, build_sensor(current))
        if observer.size != metadata["observation_size"]:
            raise ValueError("Migrated observation dimensions differ")
        result["observation_spec"] = observer.specification()
    else:
        raise ValueError(
            "Migrate an inference checkpoint or recurrent learner state; a persisted "
            "old environment is not transferable"
        )
    result.update(config=current, config_version=4, sha256=hashlib.sha256(content).hexdigest())
    result["migration"] = dict(
        source=str(source),
        source_sha256=digest,
        source_config=metadata["config"],
        target_config_sha256=hashlib.sha256(
            json.dumps(current, sort_keys=True).encode()
        ).hexdigest(),
        source_preserved=True,
        parameter_sha256=metadata["parameter_sha256"],
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        stream.write(content)
    try:
        with destination.with_suffix(".json").open("x") as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")
    except BaseException:
        destination.unlink()
        raise
    return result["migration"]


def main(argv=None):
    """Migrate with an explicitly reviewed resolved configuration, never a recipe guess."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    result = migrate_checkpoint(args.source, args.destination, json.loads(args.config.read_text()))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
