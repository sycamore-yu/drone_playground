"""A missing reference horizon is a valid no-plan decision."""

import numpy as np

from drone_playground.learning.inference import FrozenNeuralCommand
from drone_playground.references import Trajectory
from drone_playground.runtime.decision import Decision, validate_decision


def test_short_reference_returns_a_typed_no_plan():
    tracker = FrozenNeuralCommand("unused.pkl", input_kind="trajectory")
    tracker.reference_offsets = np.array([0.0, 0.1, 0.2])
    tracker.count = 0
    curve = Trajectory(0.0, [0.1], np.zeros((1, 4, 2)))
    decision = tracker.step({"time": 0.0}, curve)
    assert isinstance(decision, Decision)
    assert decision.output is None
    assert decision.status == "no_plan"
    assert decision.diagnostics["reason"] == "insufficient_reference_horizon"
    assert decision.diagnostics["required_reference_until"] == 0.2
    validate_decision(decision, 0.0)
