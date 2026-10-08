"""Public methods and execution share one typed decision without transport wrappers."""

from types import SimpleNamespace

import numpy as np
from hydra.utils import instantiate

from drone_playground.configuration import compose_experiment
from drone_playground.references import Waypoint
from drone_playground.runtime.decision import Decision, output_reply
from drone_playground.runtime.pipeline import Pipeline


def test_physical_output_is_a_decision_not_a_second_dictionary_contract():
    result = output_reply(Waypoint([[1.0, 0.0, 1.0]], 0.1), 0.0, "goal", 1.0)
    assert isinstance(result, Decision)
    assert result.output.valid_until == result.valid_until


def test_pipeline_keeps_typed_stages_and_does_not_sample_controller_reference(tmp_path):
    pipeline = Pipeline(
        dict(
            output="trajectory",
            stages=[
                dict(_target_="drone_playground.planning.goal.GoalWaypoints"),
                dict(_target_="drone_playground.planning.minimum_jerk.MinimumJerkPlanner"),
            ],
        ),
        tmp_path,
        SimpleNamespace(freq=50),
    )
    try:
        pipeline.start({}, [3.0, 0.0, 1.0])
        result = pipeline.step(
            dict(
                time=0.0,
                position=[0.0, 0.0, 1.0],
                velocity=[0.0, 0.0, 0.0],
                quaternion=[0.0, 0.0, 0.0, 1.0],
            )
        )
        assert isinstance(result, Decision)
        assert all(isinstance(stage, Decision) for stage in result.stages)
        assert result.sampled_reference is None
        np.testing.assert_allclose(
            result.output.sample(result.output.end_time)["position"], [3.0, 0.0, 1.0]
        )
    finally:
        pipeline.close()


def test_hydra_constructs_ros_adapter_without_factory_or_starting_ros(tmp_path):
    config = compose_experiment("papers/super")
    method = config["method"]
    assert method["_target_"] == "drone_playground.integrations.ros1.planner.RosPlanner"
    adapter = instantiate(
        {
            "_target_": method["_target_"],
            "settings": {key: value for key, value in method.items() if not key.startswith("_")},
        },
        directory=tmp_path,
        _recursive_=False,
        _convert_="all",
    )
    try:
        assert adapter.client is None
        assert not hasattr(adapter, "request")
    finally:
        adapter.close()
