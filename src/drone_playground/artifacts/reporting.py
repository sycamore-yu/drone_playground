"""Report serialization and parameter-content identities for saved runs."""

import hashlib
import json
from pathlib import Path

import jax
import numpy as np


def tree_digest(tree) -> str:
    """Hash array content without depending on pickle or device placement."""
    digest = hashlib.sha256()
    for leaf in jax.tree.leaves(tree):
        array = np.asarray(leaf)
        digest.update(str((array.shape, array.dtype)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def save_report(path: Path, report: dict) -> None:
    """Persist a machine-readable experiment report as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # NaN results remain visible rather than being silently converted to success.
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
