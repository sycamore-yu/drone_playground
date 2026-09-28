"""Present already recorded physical trajectories through the existing viewer."""

from pathlib import Path

from .rscope_io import publish_run


def replay(directory: Path, *, publish=True):
    directory = Path(directory).resolve()
    candidates = sorted({p.parent for p in directory.rglob("*.mj_unroll")})
    if not candidates:
        raise FileNotFoundError(f"Replay directory contains no physical trajectories: {directory}")
    result = dict(
        source_directory=str(directory),
        available_cases=[str(p) for p in candidates],
        execution="recorded-trajectory-replay",
    )
    if publish:
        result["active_directory"] = str(publish_run(candidates[0]))
    return result
