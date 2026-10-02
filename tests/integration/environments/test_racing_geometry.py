"""Racing geometry after full environment composition."""

import numpy as np


def test_control_course_keeps_gate_geometry_and_excludes_robot():
    from drone_playground.composition import build_environment, compose_experiment
    from drone_playground.environments.scenes.mujoco_geometry import bank_from_environment

    env = build_environment(compose_experiment('control/apg', "racing"), "cpu", "eval", 1)
    try:
        bank, identity = bank_from_environment(env)
        assert bank.rotations.shape[-2:] == (3, 3)
        assert bank.active_count(0) >= 20
        assert not any("drone" in name.lower() for name in identity["geometry_names"])
        assert len(identity["gate_order"]) == 5
        assert np.any(np.abs(np.asarray(bank.rotations)[..., 0, 1]) > 0.5)
    finally:
        env.close()
