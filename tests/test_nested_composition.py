"""The CLI already owns Hydra when a task needs a canonical reference recipe."""

from hydra import initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from drone_playground.composition import CONFIG_ROOT, compose_method


def test_reference_composition_reuses_cli_hydra_without_clearing_it():
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_ROOT)):
        outer = GlobalHydra.instance().hydra
        config = compose_method("learning/apg", "hovering")
        assert config["env"]["task"]["name"] == "hovering"
        assert GlobalHydra.instance().hydra is outer
        assert compose_method("learning/ppo", "tracking")["algorithm"]["name"] == "ppo"
    assert not GlobalHydra.instance().is_initialized()
