"""Fixed numbered P5 scene catalog: feasibility and provenance."""

from pathlib import Path

import numpy as np

from drone_playground.tasks.scenes.fixed_navigation import (
    build_fixed_bank,
    load_fixed_catalog,
    validate_fixed_catalog,
)
from drone_playground.tasks.scenes.navigation import MOTION_STATIC


def test_fixed_catalog_has_numbered_static_and_dynamic_candidates():
    catalog = load_fixed_catalog()
    assert catalog["version"] == "p5-fixed-candidate-v4-route-free"
    ids = [scene["id"] for scene in catalog["scenes"]]
    assert ids == [
        "S01",
        "S02",
        "S03",
        "S04",
        "S05",
        "S06",
        "D01",
        "D02",
        "D03",
        "D04",
        "D05",
        "D06",
    ]
    assert {scene["source"] for scene in catalog["scenes"]} >= {
        "NavRL",
        "P2M",
        "MIGHTY+SANDO",
        "manual",
    }
    assert catalog["references"]["NavRL"]["license"] == "MIT"
    assert catalog["references"]["P2M"]["license"] == "MIT"


def test_every_candidate_blocks_direct_flight_and_has_route_free_connectivity():
    reports = validate_fixed_catalog(load_fixed_catalog(), maximum_clear_straight_run_m=35.0)
    assert len(reports) == 12
    assert all(report["straight_line_blocked"] for report in reports)
    for report in reports:
        topology = report["topology"]
        assert topology["all_snapshots_reachable"]
        assert topology["max_full_length_straight_lanes"] == 0
        assert topology["max_clear_straight_run_m"] <= 35.0


def test_fixed_bank_is_exact_and_does_not_generate_extra_obstacles():
    catalog = load_fixed_catalog()
    ids = ["S01", "S04", "D01", "D04"]
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
    path = Path("configs/scene/p5_fixed_catalog_v4.json")
    assert path.is_file()
    text = path.read_text()
    assert '"inspection_path"' not in text
    assert '"P2M"' in text and '"NavRL"' in text


def test_v2_catalog_is_100_by_40_and_uses_half_to_six_meter_flight_bounds():
    catalog = load_fixed_catalog("configs/scene/p5_fixed_catalog_v2.json")
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
    catalog = load_fixed_catalog("configs/scene/p5_fixed_catalog_v3.json")
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
    catalog = load_fixed_catalog("configs/scene/p5_fixed_catalog_v4.json")
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
