"""Committed benchmark files remain reachable through the public loaders."""

from drone_playground.benchmarks import load_protocol, load_specification
from drone_playground.environments.scenes.catalog import NavigationCatalogScene


def test_committed_benchmark_specifications_are_loadable():
    assert load_protocol()["name"] == "navigation"
    assert load_specification("tracking")["name"] == "tracking"
    assert load_specification("racing")["name"] == "racing"


def test_navigation_scene_assets_are_loaded_without_historical_audit_results():
    bank, manifest = NavigationCatalogScene().build()
    assert bank.num_instances == 8
    assert manifest["scene_ids"] == ["S01", "S02", "S03", "S06", "D01", "D02", "D03", "D06"]
    assert all("review" not in scene for scene in manifest["scenes"])
