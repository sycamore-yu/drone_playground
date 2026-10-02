"""Navigation catalog contracts after Hydra composition."""

from hydra.utils import instantiate

from drone_playground.composition import compose_experiment


def test_navigation_benchmark_entries_expand_the_accepted_scenes():
    expected = {
        "navigation/static": (
            False,
            ["S01"] * 4 + ["S02"] * 4 + ["S03", "S06", "S03", "S06"],
        ),
        "navigation/dynamic": (
            True,
            ["D01"] * 4 + ["D02"] * 4 + ["D03", "D06", "D03", "D06"],
        ),
    }
    for environment, (dynamic, scene_ids) in expected.items():
        config = compose_experiment('control/ppo', environment)
        assert config["env"]["scene"]["name"] == "navigation"
        assert config["env"]["scene"]["dynamic"] is dynamic
        assert config["env"]["scene"]["_target_"].endswith("catalog.CatalogNavigationScene")
        scene = instantiate(config["env"]["scene"], _convert_="all")
        bank, manifest = scene.build(seed=12345, per_difficulty=4)
        assert bank.num_instances == 12
        assert list(bank.subtype_names) == scene_ids
        assert manifest["catalog"] == "navigation"
        assert manifest["accepted_scene_ids"] == list(
            config["env"]["scene"]["scene_ids"]
        )


def test_navigation_experiments_share_catalog_task_identity():
    experiments = [
        (method, environment)
        for method in ("control/ppo", "control/apg", "control/shac", "papers/dva", "papers/ego_planner", "papers/super")
        for environment in ("navigation/static", "navigation/dynamic")
    ]
    checked = []
    for method, environment in experiments:
        config = compose_experiment(method, environment)
        if config["env"]["task"]["name"] != "navigation":
            continue
        checked.append((method, environment))
        assert config["env"]["scene"]["name"] == "navigation"
        assert config["env"]["scene"]["_target_"].endswith("catalog.CatalogNavigationScene")
        assert config["env"]["scene"]["dynamic"] is config["env"]["task"]["dynamic"]
    assert checked
