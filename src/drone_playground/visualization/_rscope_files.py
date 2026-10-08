"""Shared RScope file locking and atomic writes for export and publication."""

from __future__ import annotations

import os
import tempfile
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

_RSCOPE_LOCK = threading.RLock()


@contextmanager
def _exclusive_rscope() -> Iterator[None]:
    """Serialize rscope's mutable module-level configuration across threads and processes."""
    lock_path = Path("/tmp/rscope/.drone_playground.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with _RSCOPE_LOCK, lock_path.open("a+b") as lock_file:
        try:
            import fcntl

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        except ImportError:
            fcntl = None
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        tmp_path = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)
