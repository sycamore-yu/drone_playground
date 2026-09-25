"""Capture Python training output while leaving the terminal usable."""

import contextlib
import sys
import threading
from pathlib import Path


@contextlib.contextmanager
def capture_console(path: Path):
    """Tee stdout/stderr; low-level CUDA diagnostics remain in the launcher's log."""
    lock = threading.Lock()
    with Path(path).open("a", buffering=1) as handle:

        class Tee:
            def __init__(self, original):
                self.original = original

            def write(self, text):
                with lock:
                    handle.write(text)
                    self.original.write(text)
                return len(text)

            def flush(self):
                with lock:
                    handle.flush()
                    self.original.flush()

            def __getattr__(self, name):
                return getattr(self.original, name)

        with (
            contextlib.redirect_stdout(Tee(sys.stdout)),
            contextlib.redirect_stderr(Tee(sys.stderr)),
        ):
            yield
