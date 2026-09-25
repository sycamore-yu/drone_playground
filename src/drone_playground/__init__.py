"""Crazyflow-based drone learning and control experiments."""

import os

# Crazyflow requires this setting before the first SciPy import.
os.environ.setdefault("SCIPY_ARRAY_API", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

__version__ = "0.1.0"
