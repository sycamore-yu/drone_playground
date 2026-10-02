"""Committed benchmark files remain reachable through the public loaders."""

from drone_playground.benchmarks import load_protocol, load_specification
from drone_playground.environments.scenes.catalog import (
    DEFAULT_CATALOG,
    verified_geometry,
)


def test_committed_benchmark_specifications_are_loadable():
    assert load_protocol()["name"] == "navigation"
    assert load_specification("tracking")["name"] == "tracking"
    assert load_specification("racing")["name"] == "racing"


def test_navigation_geometry_verification_matches_catalog_bytes():
    reports, digest = verified_geometry(DEFAULT_CATALOG)
    assert digest
    assert len(reports) == 8
    assert all(report["straight_line_blocked"] for report in reports.values())
    assert all(
        report["topology"]["all_snapshots_reachable"] for report in reports.values()
    )
