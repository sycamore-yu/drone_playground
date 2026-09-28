"""Fixed numbered P5 scene catalog: feasibility and provenance."""

from pathlib import Path

import numpy as np
from hydra.utils import instantiate

from drone_playground.environments.scenes.catalog import (
    build_fixed_bank,
    catalog_obstacle,
    load_fixed_catalog,
    validate_fixed_catalog,
)
from drone_playground.environments.scenes.navigation import MOTION_STATIC
from tests.reference_configs import compose_reference as compose_config


def test_fixed_catalog_has_numbered_static_and_dynamic_candidates():
    catalog = load_fixed_catalog()
    assert catalog["name"] == "navigation8"
    assert catalog["version"] == "navigation8-v1"
    assert catalog["status"] == "accepted"
    ids = [scene["id"] for scene in catalog["scenes"]]
    assert ids == [
        "S01",
        "S02",
        "S03",
        "D01",
        "D02",
        "D03",
        "S06",
        "D06",
    ]
    primary = [scene for scene in catalog["scenes"] if scene["benchmark_role"] == "primary"]
    assert len(primary) == 6
    assert all(scene["source"] == "SANDO" for scene in primary)
    assert catalog["references"]["NavRL"]["license"] == "MIT"
    assert catalog["references"]["P2M"]["license"] == "MIT"


def test_every_candidate_blocks_direct_flight_and_has_route_free_connectivity():
    catalog = load_fixed_catalog()
    reports = validate_fixed_catalog(catalog, maximum_clear_straight_run_m=35.0)
    assert len(reports) == 8
    assert all(report["straight_line_blocked"] for report in reports)
    for report in reports:
        topology = report["topology"]
        assert topology["all_snapshots_reachable"]
        scene = next(scene for scene in catalog["scenes"] if scene["id"] == report["scene_id"])
        acceptance = scene["topology_acceptance"]
        max_lanes = acceptance["max_full_length_straight_lanes"]
        max_run = acceptance["max_clear_straight_run_m"]
        if max_lanes is not None:
            assert topology["max_full_length_straight_lanes"] <= max_lanes
        if max_run is not None:
            assert topology["max_clear_straight_run_m"] <= max_run


def test_fixed_bank_is_exact_and_does_not_generate_extra_obstacles():
    catalog = load_fixed_catalog()
    ids = ["S01", "S03", "D01", "D03"]
    bank, manifest = build_fixed_bank(catalog, ids)
    assert bank.num_instances == len(ids)
    assert tuple(bank.subtype_names) == tuple(ids)
    declared = [
        len(next(scene for scene in catalog["scenes"] if scene["id"] == item)["obstacles"])
        + len(catalog.get("boundary_obstacles", []))
        for item in ids
    ]
    assert [bank.active_count(index) for index in range(len(ids))] == declared
    assert manifest["source"] == "explicit numbered fixed-scene catalog; no random generation"
    assert manifest["bank_digest"]


def test_static_and_dynamic_motion_identity_is_explicit():
    catalog = load_fixed_catalog()
    static_ids = [scene["id"] for scene in catalog["scenes"] if not scene["dynamic"]]
    dynamic_ids = [scene["id"] for scene in catalog["scenes"] if scene["dynamic"]]
    static, _ = build_fixed_bank(catalog, static_ids)
    dynamic, _ = build_fixed_bank(catalog, dynamic_ids)
    assert np.all(np.asarray(static.motion)[np.asarray(static.active)] == MOTION_STATIC)
    active_motion = np.asarray(dynamic.motion)[np.asarray(dynamic.active)]
    assert np.any(active_motion != MOTION_STATIC)


def test_default_catalog_is_a_committed_route_free_human_readable_file():
    path = Path("assets/scenes/navigation/catalog.json")
    assert path.is_file()
    text = path.read_text()
    assert '"inspection_path"' not in text
    assert '"P2M"' in text and '"NavRL"' in text


def test_navigation8_primary_scenes_match_sando_difficulty_protocol():
    catalog = load_fixed_catalog("assets/scenes/navigation/catalog.json")
    by_id = {scene["id"]: scene for scene in catalog["scenes"]}
    assert [len(by_id[scene_id]["obstacles"]) for scene_id in ("S01", "S02", "S03")] == [
        41,
        81,
        162,
    ]
    assert [len(by_id[scene_id]["obstacles"]) for scene_id in ("D01", "D02", "D03")] == [
        50,
        100,
        200,
    ]
    assert [
        sum(
            obstacle.get("motion", "static") != "static"
            for obstacle in by_id[scene_id]["obstacles"]
        )
        for scene_id in ("D01", "D02", "D03")
    ] == [32, 65, 130]
    expected_static_geometry = {
        "D01": (9, 3, 6),
        "D02": (17, 6, 12),
        "D03": (35, 12, 23),
    }
    for scene_id, (cylinders, vertical_boxes, horizontal_boxes) in expected_static_geometry.items():
        obstacles = by_id[scene_id]["obstacles"]
        assert sum(o.get("role") == "sando_dynamic_static_cylinder" for o in obstacles) == cylinders
        assert (
            sum(o.get("role") == "sando_static_vertical_box" for o in obstacles) == vertical_boxes
        )
        assert (
            sum(o.get("role") == "sando_static_horizontal_box" for o in obstacles)
            == horizontal_boxes
        )
        rectangular = [
            o
            for o in obstacles
            if o.get("role") in {"sando_static_vertical_box", "sando_static_horizontal_box"}
        ]
        assert all(o["motion"] == "static" for o in rectangular)
        assert all(
            o["size"] == [0.2, 0.2, 2.0]
            for o in rectangular
            if o["role"] == "sando_static_vertical_box"
        )
        assert all(
            o["size"] == [0.2, 2.0, 0.2]
            for o in rectangular
            if o["role"] == "sando_static_horizontal_box"
        )


def test_navigation8_retains_3d_extensions_and_moves_d06_crossbars():
    catalog = load_fixed_catalog("assets/scenes/navigation/catalog.json")
    by_id = {scene["id"]: scene for scene in catalog["scenes"]}
    assert by_id["S06"]["benchmark_role"] == "3d-extension"
    assert by_id["D06"]["benchmark_role"] == "3d-extension"
    bars = [
        obstacle
        for obstacle in by_id["D06"]["obstacles"]
        if obstacle.get("role") == "moving_crossbar"
    ]
    assert len(bars) == 4
    assert all(obstacle["motion"] == "linear_bounce" for obstacle in bars)
    for payload in bars:
        obstacle = catalog_obstacle(payload)
        positions = np.asarray([obstacle.position(time) for time in (0.0, 2.0, 4.0)])
        assert np.ptp(positions[:, 2]) > 0.1


def test_navigation8_hydra_entries_expand_the_accepted_eight_scenes():
    expected = {
        "p5_navigation_static": (
            False,
            ["S01"] * 4 + ["S02"] * 4 + ["S03", "S06", "S03", "S06"],
        ),
        "p5_navigation_dynamic": (
            True,
            ["D01"] * 4 + ["D02"] * 4 + ["D03", "D06", "D03", "D06"],
        ),
    }
    for experiment, (dynamic, scene_ids) in expected.items():
        config = compose_config(experiment)
        assert config["env"]["scene"]["name"] == "navigation"
        assert config["env"]["scene"]["dynamic"] is dynamic
        assert config["env"]["scene"]["_target_"].endswith("catalog.NavigationScene")
        scene = instantiate(config["env"]["scene"], _convert_="all")
        bank, manifest = scene.build(seed=12345, per_difficulty=4)
        assert bank.num_instances == 12
        assert list(bank.subtype_names) == scene_ids
        assert manifest["catalog"] == "navigation8"
        assert manifest["accepted_scene_ids"] == list(config["env"]["scene"]["scene_ids"])


def test_all_p5_navigation_experiments_use_navigation8():
    from drone_playground.composition import compose_method

    experiments = [
        (method, environment)
        for method in (
            "learning/ppo",
            "learning/apg",
            "learning/shac",
            "learning/dva",
            "paper/ego_planner",
            "paper/super",
        )
        for environment in ("navigation/static", "navigation/dynamic")
    ]
    checked = []
    for method, environment in experiments:
        config = compose_method(method, environment)
        if config["env"]["task"]["name"] != "navigation":
            continue
        checked.append((method, environment))
        assert config["env"]["scene"]["name"] == "navigation"
        assert config["env"]["scene"]["_target_"].endswith("catalog.NavigationScene")
        assert config["env"]["scene"]["dynamic"] is config["env"]["task"]["dynamic"]
    assert checked


def test_v2_catalog_is_100_by_40_and_uses_half_to_six_meter_flight_bounds():
    catalog = load_fixed_catalog("assets/scenes/archive/p5_fixed_catalog_v2.json")
    world = catalog["world"]
    assert world["length_m"] == 100.0
    assert world["width_m"] == 40.0
    assert world["z_min_m"] == 0.5
    assert world["z_max_m"] == 6.0
    assert world["start"] == [2.0, 0.0, 3.0]
    assert world["goal"] == [98.0, 0.0, 3.0]
    reports = validate_fixed_catalog(catalog, minimum_route_clearance_m=0.5)
    assert len(reports) == 12
    assert all(report["inspection_path_length_m"] > 95.0 for report in reports)
    static, _ = build_fixed_bank(catalog, ["S01", "S06"])
    np.testing.assert_allclose(np.asarray(static.world_low), [0.0, -20.0, 0.5])
    np.testing.assert_allclose(np.asarray(static.world_high), [100.0, 20.0, 6.0])


def test_v3_catalog_scales_density_and_has_visible_physical_boundaries():
    catalog = load_fixed_catalog("assets/scenes/archive/p5_fixed_catalog_v3.json")
    reports = validate_fixed_catalog(catalog, minimum_route_clearance_m=0.5)
    expected = {"easy": 50, "medium": 100, "hard": 150}
    assert len(catalog["boundary_obstacles"]) == 4
    for row in reports:
        assert row["field_obstacles"] == expected[row["difficulty"]]
        assert row["boundary_obstacles"] == 4
        assert row["obstacles"] == expected[row["difficulty"]] + 4
        assert row["inspection_route_min_clearance_m"] >= 0.5

    bank, manifest = build_fixed_bank(catalog, ["S01", "S04", "D05"])
    assert [bank.active_count(i) for i in range(3)] == [54, 154, 154]
    assert "physical boundary walls" in " ".join(manifest["design_rules"])

    walls = catalog["boundary_obstacles"]
    by_name = {wall["name"]: wall for wall in walls}
    assert by_name["wall_y_min"]["origin"] == [50.0, -19.75, 3.25]
    assert by_name["wall_y_max"]["origin"] == [50.0, 19.75, 3.25]
    assert by_name["wall_x_min"]["size"] == [0.25, 19.5, 2.75]
    assert by_name["wall_x_max"]["size"] == [0.25, 19.5, 2.75]


def test_v4_catalog_has_no_reference_route_and_requires_topological_connectivity():
    catalog = load_fixed_catalog("assets/scenes/archive/p5_fixed_catalog_v4.json")
    assert all("inspection_path" not in scene for scene in catalog["scenes"])
    assert all("inspection_duration_s" not in scene for scene in catalog["scenes"])
    reports = validate_fixed_catalog(
        catalog,
        minimum_route_clearance_m=0.5,
        maximum_clear_straight_run_m=35.0,
    )
    assert len(reports) == 12
    for report in reports:
        topology = report["topology"]
        assert topology["all_snapshots_reachable"]
        assert topology["max_full_length_straight_lanes"] == 0
        assert topology["max_clear_straight_run_m"] <= 35.0
        assert topology["validation"].endswith("path coordinates are discarded")
