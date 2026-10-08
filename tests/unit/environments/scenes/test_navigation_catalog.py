"""Fixed numbered P5 scene catalog: feasibility and provenance."""

import numpy as np

from drone_playground.environments.scenes.catalog import (
    build_fixed_bank,
    catalog_obstacle,
    load_fixed_catalog,
)
from drone_playground.environments.scenes.geometry import MOTION_STATIC


def test_fixed_catalog_has_numbered_static_and_dynamic_candidates():
    catalog = load_fixed_catalog()
    assert catalog["name"] == "navigation"
    assert catalog["version"] == "navigation-v1"
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


def validation_stub(catalog):
    return {
        scene["id"]: {"scene_id": scene["id"], "validated": True} for scene in catalog["scenes"]
    }


def test_fixed_bank_is_exact_and_does_not_generate_extra_obstacles():
    catalog = load_fixed_catalog()
    ids = ["S01", "S03", "D01", "D03"]
    bank, manifest = build_fixed_bank(catalog, ids, validated_reports=validation_stub(catalog))
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
    reports = validation_stub(catalog)
    static, _ = build_fixed_bank(catalog, static_ids, validated_reports=reports)
    dynamic, _ = build_fixed_bank(catalog, dynamic_ids, validated_reports=reports)
    assert np.all(np.asarray(static.motion)[np.asarray(static.active)] == MOTION_STATIC)
    active_motion = np.asarray(dynamic.motion)[np.asarray(dynamic.active)]
    assert np.any(active_motion != MOTION_STATIC)


def test_default_catalog_is_a_committed_route_free_human_readable_file():
    from drone_playground.environments.scenes.catalog import DEFAULT_CATALOG, load_fixed_catalog

    path = DEFAULT_CATALOG
    assert path.is_file()
    text = path.read_text()
    assert '"inspection_path"' not in text
    catalog = load_fixed_catalog(path)
    references = catalog["references"]
    assert "P2M" in references and "NavRL" in references


def test_navigation_benchmark_primary_scenes_match_sando_difficulty_protocol():
    catalog = load_fixed_catalog()
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


def test_navigation_benchmark_retains_3d_extensions_and_moves_d06_crossbars():
    catalog = load_fixed_catalog()
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
