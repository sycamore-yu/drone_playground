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
        "SANDO",
        "manual",
    }
    assert catalog["references"]["NavRL"]["license"] == "MIT"
    assert catalog["references"]["P2M"]["license"] == "MIT"


def test_every_candidate_blocks_direct_flight_and_has_a_known_clear_route():
    reports = validate_fixed_catalog(load_fixed_catalog(), minimum_route_clearance_m=0.35)
    assert len(reports) == 12
    assert all(report["straight_line_blocked"] for report in reports)
    assert min(report["inspection_route_min_clearance_m"] for report in reports) >= 0.35


def test_fixed_bank_is_exact_and_does_not_generate_extra_obstacles():
    catalog = load_fixed_catalog()
    ids = ["S01", "S04", "D01", "D04"]
    bank, manifest = build_fixed_bank(catalog, ids)
    assert bank.num_instances == len(ids)
    assert tuple(bank.subtype_names) == tuple(ids)
    declared = [
        len(next(scene for scene in catalog["scenes"] if scene["id"] == item)["obstacles"])
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


def test_catalog_is_a_committed_human_readable_file():
    path = Path("configs/scene/p5_fixed_catalog.json")
    assert path.is_file()
    text = path.read_text()
    assert '"inspection_path"' in text
    assert '"P2M"' in text and '"NavRL"' in text
