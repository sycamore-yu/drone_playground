"""Explicit copy-only conversion retains parameter content and rejects overwrites."""

import hashlib
import json

import jax.numpy as jnp
import numpy as np
import pytest
from brax.io import model

from drone_playground.composition import compose_method
from drone_playground.runs.migration import migrate_checkpoint, require_current


def test_current_checkpoint_copy_verifies_bytes_and_preserves_numeric_payload(tmp_path):
    source = tmp_path / "source.pkl"
    model.save_params(str(source), {"weight": np.arange(9, dtype=np.float32)})
    config = compose_method("learning/ppo", "tracking")
    meta = dict(config=config, sha256=hashlib.sha256(source.read_bytes()).hexdigest(), step=42)
    source.with_suffix(".json").write_text(json.dumps(meta))
    before = source.read_bytes()
    destination = migrate_checkpoint(source, tmp_path / "converted")
    np.testing.assert_array_equal(model.load_params(str(destination))["weight"], np.arange(9))
    converted = json.loads(destination.with_suffix(".json").read_text())
    assert converted["step"] == 42
    assert converted["migration"]["source_sha256"] == meta["sha256"]
    assert source.read_bytes() == before
    with pytest.raises(FileExistsError):
        migrate_checkpoint(source, tmp_path / "converted")


def test_migration_rejects_tampered_checkpoint_before_creating_destination(tmp_path):
    source = tmp_path / "source.pkl"
    model.save_params(str(source), jnp.ones(1))
    source.with_suffix(".json").write_text(
        json.dumps(dict(config=compose_method(), sha256="wrong"))
    )
    with pytest.raises(ValueError, match="digest|摘要"):
        migrate_checkpoint(source, tmp_path / "converted")
    assert not (tmp_path / "converted").exists()


def test_historical_config_requires_explicit_migration():
    with pytest.raises(ValueError, match="显式迁移"):
        require_current({"config_version": 2})
