"""Environment initialization retains ownership when native setup fails."""

from types import SimpleNamespace

import pytest

from drone_playground.environments import initialization


@pytest.mark.parametrize("failure", ["bind", "reset"])
def test_failed_native_initialization_closes_its_allocated_simulation(monkeypatch, failure):
    closed = []

    def reject(*args, **kwargs):
        raise ValueError("invalid native setup")

    simulation = SimpleNamespace(
        sim=SimpleNamespace(freq=500, reset=reject if failure == "reset" else lambda: None),
        single_action_space=object(),
        close=lambda: closed.append(True),
    )
    monkeypatch.setattr(initialization, "FigureEightEnv", lambda **kwargs: simulation)
    dynamics = SimpleNamespace(
        forward="so_rpy",
        drone="cf2x_L250",
        bind=reject if failure == "bind" else lambda *args, **kwargs: None,
    )
    env = SimpleNamespace(
        dynamics=dynamics,
        duration=1.0,
        freq=50,
        device="cpu",
        controller=SimpleNamespace(native_mode="attitude"),
    )
    with pytest.raises(ValueError, match="invalid native setup"):
        initialization.initialize_rigid_body(env, start=[0, 0, 1], figure_eight=True, reset=True)
    assert closed == [True]
