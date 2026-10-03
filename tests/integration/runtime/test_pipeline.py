from types import SimpleNamespace

import numpy as np
import pytest

from drone_playground.control.setpoints import StateSetpoint
from drone_playground.planning.minimum_jerk import minimum_jerk_path
from drone_playground.references import Trajectory, Waypoint
from drone_playground.runtime.pipeline import Pipeline
from tests.helpers.paths import REPO_ROOT


def test_minimum_jerk_interpolation_preserves_physical_endpoint_constraints():
    path = Waypoint([[2, 1, 1], [3, 0, 2]], 0.1)
    curve = minimum_jerk_path(path, [0, 0, 1], [0.2, 0, 0], 3.0)
    samples = curve.sample_many([3.0, 3.0 + curve.durations[0], curve.end_time])
    np.testing.assert_allclose(samples["position"], [[0, 0, 1], *path.positions], atol=1e-12)
    np.testing.assert_allclose(samples["velocity"], [[0.2, 0, 0], [0, 0, 0], [0, 0, 0]], atol=1e-12)
    np.testing.assert_allclose(samples["acceleration"], 0, atol=1e-12)
    assert not curve.yaw_defined


def test_configured_waypoint_planning_chain_preserves_curve_and_resets(tmp_path):
    planner = Pipeline(
        dict(
            output="trajectory",
            stages=[
                dict(_target_="drone_playground.planning.goal.GoalWaypoints"),
                dict(_target_="drone_playground.planning.minimum_jerk.MinimumJerkPlanning"),
            ],
        ),
        tmp_path,
        SimpleNamespace(freq=50),
    )
    packet = dict(time=0.0, position=[0, 0, 1], velocity=[0, 0, 0], quaternion=[0, 0, 0, 1])
    try:
        planner.start({}, [2, 0, 1])
        first = planner.step(packet)
        second = planner.step(dict(packet, time=0.02, position=[0.001, 0, 1]))
        assert first["output"] is second["output"]
        assert first["generated_at"] == second["generated_at"] == 0.0
        assert second["reference"]["position"][0] > 0
        assert second["plans"] == 1 and second["commands"] == 2
        planner.start({}, [4, 0, 1])
        restarted = planner.step(packet)
        np.testing.assert_allclose(
            restarted["output"].sample(restarted["output"].end_time)["position"], [4, 0, 1]
        )
        assert restarted["plans"] == 1
    finally:
        planner.close()


def test_pipeline_rejects_missing_physical_adapter_before_running(tmp_path):
    with pytest.raises(ValueError, match=r"contract mismatch"):
        Pipeline(
            dict(
                output="trajectory",
                stages=[
                    dict(_target_="drone_playground.planning.minimum_jerk.MinimumJerkPlanning")
                ],
            ),
            tmp_path,
            SimpleNamespace(freq=50),
        )


@pytest.mark.parametrize(
    "kind,value",
    [
        ("waypoint", Waypoint([[3, 0, 1]], 0.1)),
        ("state", StateSetpoint(acceleration=np.array([1, 0, 0]))),
        (
            "trajectory",
            Trajectory(0.0, [1.0], np.array([[[0.0, 1.0], [0.0, 0.0], [1.0, 0.0], [0.0, 0.0]]])),
        ),
    ],
)
def test_cpp_downstream_receives_exact_upstream_physical_value(kind, value):
    from drone_playground.rpc.client import NativeClient

    binary = REPO_ROOT / "tmp/direct-composition-refactor/native-sdk-v2/interop_server"
    if not binary.is_file():
        pytest.skip("Build C++ SDK fixture")
    with NativeClient("echo", command=[str(binary), "{address}"]) as client:
        client.reset()
        reply = client.step(
            time=0.0,
            state=dict(position=[0, 0, 1], velocity=[0, 0, 0], quaternion=[0, 0, 0, 1]),
            upstream=value,
        )
        assert type(reply.output) is type(value)
        if kind == "trajectory":
            np.testing.assert_array_equal(reply.output.coefficients, value.coefficients)
        else:
            attribute = "positions" if kind == "waypoint" else "acceleration"
            np.testing.assert_array_equal(
                getattr(reply.output, attribute), getattr(value, attribute)
            )


def test_real_cpp_stages_chain_without_algorithm_specific_host_changes(tmp_path):
    binary = REPO_ROOT / "tmp/direct-composition-refactor/native-sdk-v2/interop_server"
    if not binary.is_file():
        pytest.skip("Build C++ SDK fixture")
    settings = dict(output="waypoint", deployment=dict(command=[str(binary), "{address}"]))
    stages = [
        dict(
            _target_="drone_playground.integrations.grpc_service.NativeServicePlanner",
            settings=dict(settings, algorithm=algorithm, **extra),
        )
        for algorithm, extra in [("waypoint", {}), ("echo", {"input": "waypoint"})]
    ]
    stages.append(dict(_target_="drone_playground.planning.minimum_jerk.MinimumJerkPlanning"))
    planner = Pipeline(dict(output="trajectory", stages=stages), tmp_path, SimpleNamespace(freq=50))
    try:
        planner.start({}, [4, 0, 1])
        reply = planner.step(
            dict(time=0.0, position=[0, 0, 1], velocity=[0, 0, 0], quaternion=[0, 0, 0, 1])
        )
        curve = reply["output"]
        np.testing.assert_allclose(curve.sample(curve.end_time)["position"], [3, 0, 1])
        assert reply["commands"] == 1
    finally:
        planner.close()


def test_no_plan_stops_downstream_and_close_visits_all_modules(monkeypatch, tmp_path):
    import drone_playground.runtime.pipeline as pipeline

    class Source:
        input_kind, output_kind, derivatives = None, "waypoint", "none"

        def start(self, *args):
            pass

        def step(self, packet, upstream):
            return dict(output=None, decision_status="no_plan")

        def close(self):
            closed.append("source")

    class Downstream:
        input_kind, output_kind, derivatives = "waypoint", "trajectory", "none"

        def start(self, *args):
            pass

        def step(self, *args):
            raise AssertionError("A downstream module ran without input")

        def close(self):
            closed.append("downstream")
            raise RuntimeError("deliberate cleanup failure")

    closed, modules = [], iter([Source(), Downstream()])
    monkeypatch.setattr(pipeline, "instantiate", lambda _, **kwargs: next(modules))
    planner = Pipeline(
        dict(
            output="trajectory",
            stages=[dict(_target_="tests.fixture"), dict(_target_="tests.fixture")],
        ),
        tmp_path,
        SimpleNamespace(freq=50),
    )
    planner.start({}, [1, 0, 1])
    reply = planner.step(dict(time=0.0))
    assert reply["output"] is None and reply["commands"] == 0 and reply["plans"] == 0
    with pytest.raises(RuntimeError, match=r"cleanup"):
        planner.close()
    assert closed == ["downstream", "source"]


def test_saved_neural_module_preserves_decoder_and_rejects_same_width_wrong_fields(tmp_path):
    from dataclasses import replace

    import jax
    import jax.numpy as jnp
    from brax.training.acme import running_statistics, specs

    from drone_playground.artifacts.checkpoints import save_policy
    from drone_playground.composition import compose_experiment
    from drone_playground.environments.environment import build_environment
    from drone_playground.learning.brax_configuration import native_training_config
    from drone_playground.learning.inference import FrozenNeuralCommand
    from drone_playground.networks.factory import network_factory

    config = compose_experiment("control/bptt", "tracking")
    config["network"]["hidden_sizes"] = [16, 16]
    native = native_training_config(config)
    net = network_factory(native)(43, 4)
    params = (
        running_statistics.init_state(specs.Array((43,), jnp.float32)),
        net.policy_network.init(jax.random.key(1)),
    )
    checkpoint = save_policy(tmp_path / "weights", params, native, 0)
    env = build_environment(config, "cpu")
    try:
        module = FrozenNeuralCommand(checkpoint).bind(env, tmp_path)
        module.start({}, [0, 0, 1], None, {})
        state = env.reset(jax.random.key(2))
        reply = module.step(dict(time=0.0, policy_observation=np.asarray(state.obs).tolist()), None)
        expected = env.physical_action(module.policy.act(state.obs))
        np.testing.assert_allclose(reply["output"].as_array(), expected, atol=1e-6)
        assert module.provenance["sha256"] == module.policy.metadata["sha256"]
        env.task.observation = replace(env.task.observation, interval=0.2)
        with pytest.raises(ValueError, match=r"semantics"):
            FrozenNeuralCommand(checkpoint).bind(env, tmp_path)
    finally:
        env.close()


def test_module_rates_hold_valid_outputs_but_never_reuse_expired_decisions(monkeypatch, tmp_path):
    import drone_playground.runtime.pipeline as pipeline

    class Source:
        input_kind, output_kind, derivatives = None, "waypoint", "none"

        def start(self, *args):
            self.calls = []

        def step(self, packet, upstream):
            self.calls.append(packet["time"])
            generated_at = packet["time"]
            valid_until = generated_at + 0.05
            return dict(
                output=Waypoint(
                    [[3, 0, 1]],
                    0.1,
                    generated_at=generated_at,
                    valid_until=valid_until,
                ),
                plan_id=str(len(self.calls)),
                generated_at=generated_at,
                valid_until=valid_until,
                decision_status="valid",
            )

        def close(self):
            pass

    source = Source()
    monkeypatch.setattr(pipeline, "instantiate", lambda _, **kwargs: source)
    planner = Pipeline(
        dict(output="waypoint", stages=[dict(_target_="tests.fixture", frequency_hz=10)]),
        tmp_path,
        SimpleNamespace(freq=50),
    )
    planner.start({}, [3, 0, 1])
    replies = [planner.step(dict(time=t)) for t in [0.0, 0.02, 0.04, 0.06, 0.08, 0.10]]
    assert source.calls == [0.0, 0.10]
    assert [reply["output"] is not None for reply in replies] == [
        True,
        True,
        True,
        False,
        False,
        True,
    ]
    assert [reply["generated_at"] for reply in replies[:5]] == [0.0, 0.0, 0.0, 0.0, 0.0]
    assert replies[-1]["generated_at"] == 0.10
    assert replies[3]["decision_status"] == "no_plan"
    assert planner.contracts[0]["frequency_hz"] == 10
    with pytest.raises(ValueError, match=r"clock"):
        planner.step(dict(time=0.09))
    planner.start({}, [3, 0, 1])
    planner.step(dict(time=0.0))
    assert source.calls == [0.0]
    planner.close()


def test_pipeline_rejects_future_generated_output(monkeypatch, tmp_path):
    import drone_playground.runtime.pipeline as pipeline

    class Source:
        input_kind, output_kind, derivatives = None, "waypoint", "none"

        def start(self, *args):
            pass

        def step(self, packet, upstream):
            generated_at = packet["time"] + 0.1
            valid_until = generated_at + 1.0
            return dict(
                output=Waypoint(
                    [[3, 0, 1]],
                    0.1,
                    generated_at=generated_at,
                    valid_until=valid_until,
                ),
                plan_id="future",
                generated_at=generated_at,
                valid_until=valid_until,
                decision_status="valid",
            )

        def close(self):
            pass

    monkeypatch.setattr(pipeline, "instantiate", lambda _, **kwargs: Source())
    planner = Pipeline(
        dict(output="waypoint", stages=[dict(_target_="tests.fixture")]),
        tmp_path,
        SimpleNamespace(freq=50),
    )
    planner.start({}, [3, 0, 1])
    with pytest.raises(ValueError, match=r"time envelope"):
        planner.step(dict(time=0.0))
    planner.close()


@pytest.mark.parametrize("rate", [0.0, -1.0, float("nan"), 60.0, 7.0])
def test_pipeline_rejects_unsupported_module_clocks(tmp_path, rate):
    with pytest.raises(ValueError, match=r"frequency"):
        Pipeline(
            dict(
                output="waypoint",
                stages=[
                    dict(_target_="drone_playground.planning.goal.GoalWaypoints", frequency_hz=rate)
                ],
            ),
            tmp_path,
            SimpleNamespace(freq=50),
        )
