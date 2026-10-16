"""The browser and release checks use the same finite set of supported commands."""

import json
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir


def test_every_builder_command_composes_with_independent_oa_training():
    """Resolve every displayed combination and reject fixed-map training defaults."""
    if shutil.which("node") is None:
        pytest.skip("Node is needed to execute the standalone browser command generator")
    root = Path(__file__).parents[1]
    html = (root / "docs/command-builder.html").read_text()
    source = html.split("<script>")[1].split("</script>")[0]
    result = subprocess.run(
        ["node"],
        input=source + "\nprocess.stdout.write(JSON.stringify(CommandBuilder.combinations()));",
        text=True,
        capture_output=True,
        check=True,
    )
    cases = json.loads(result.stdout)
    assert len(cases) == len({case["command"] for case in cases})
    assert len(cases) == 486
    assert {
        mode: sum(case["mode"] == mode for case in cases) for mode in ("train", "play", "benchmark")
    } == {"train": 42, "play": 334, "benchmark": 110}
    training = set()
    with initialize_config_dir(
        version_base=None, config_dir=str(root / "src/drone_playground/configs")
    ):
        for case in cases:
            config = compose(config_name="config", overrides=shlex.split(case["command"])[3:])
            if case["mode"] == "train":
                if config.learning.loss is not None:
                    assert config.method.action.level == "acceleration"
                else:
                    assert not config.learning.options.perception_loss
                training.add((case["task"], case["sensor"], case["algorithm"], case["backward"]))
                if case["task"].endswith("_oa"):
                    assert config.scene.name == "train/procedural"
                    assert (
                        config.simulation.procedural.obstacle_motion == case["task"].split("_")[0]
                    )
            assert "checkpoint_eval" not in case["command"]
    assert len(training) == 42
