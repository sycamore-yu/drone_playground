"""Global pytest setup for the repository test suite."""

import os

# Crazyflow requires SciPy's Array API mode before anything imports SciPy.
# Keep this at collection time so direct pytest and the Pixi task behave alike.
os.environ.setdefault("SCIPY_ARRAY_API", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
