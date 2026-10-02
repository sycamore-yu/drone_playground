import sys
import tempfile
from pathlib import Path


def test_console_captures_python_stdout_and_stderr_and_restores_streams():
    from drone_playground.artifacts.console import capture_console

    original = (sys.stdout, sys.stderr)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "console.log"
        with capture_console(path):
            print("training step 100")
            print("evaluation detail", file=sys.stderr)
        assert "training step 100" in path.read_text()
        assert "evaluation detail" in path.read_text()
    assert (sys.stdout, sys.stderr) == original
