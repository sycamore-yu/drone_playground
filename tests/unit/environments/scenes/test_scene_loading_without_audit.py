"""Scene construction needs the MJCF assets, not historical audit evidence."""

import shutil
from pathlib import Path

import numpy as np

from drone_playground.environments.scenes import catalog, catalog_validation


def test_scene_load_uses_no_offline_report_or_topology_pass(monkeypatch):
    original = Path.read_text

    def read(path, *args, **kwargs):
        if path.name == "navigation-mjcf-verification.json":
            raise AssertionError("Scene loading read a historical verification report")
        return original(path, *args, **kwargs)

    def audit(*args, **kwargs):
        raise AssertionError("Scene loading ran offline topology validation")

    monkeypatch.setattr(Path, "read_text", read)
    monkeypatch.setattr(catalog_validation, "validate_fixed_catalog", audit)
    bank, manifest = catalog.NavigationCatalogScene(scene_ids=("S01", "D01")).build()
    assert bank.num_instances == 2
    assert manifest["scene_ids"] == ["S01", "D01"]
    assert "verification" not in manifest


def test_valid_custom_mjcf_does_not_need_a_new_sha_manifest(tmp_path):
    destination = tmp_path / "scenes"
    shutil.copytree(catalog.DEFAULT_CATALOG.parent, destination)
    scene = destination / "S01.xml"
    scene.write_text(scene.read_text() + "\n<!-- local scene edit -->\n")
    bank, _ = catalog.NavigationCatalogScene(
        scene_ids=("S01",), catalog_path=str(destination / "catalog.xml")
    ).build()
    assert bank.num_instances == 1
    assert np.isfinite(np.asarray(bank.origin)).all()
