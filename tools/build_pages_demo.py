"""Export a small, truthful GitHub Pages site from recorded native flights.

Only the listed browser replays and their MJCF assets leave the ignored results
tree. No training checkpoints, native ROS workers, or privileged state is run.
"""

from __future__ import annotations

import argparse
import collections
import html
import json
import re
import shutil
from pathlib import Path, PurePosixPath
from urllib.parse import urlencode


ROOT = Path(__file__).resolve().parents[1]
RESULTS = "command-builder-native-full-20261017"
PUBLIC_CASES = (
    0, 1, 2, 3, 4, 5, 6, 7,
    8, 9, 10, 11, 21, 24, 37, 73,
    86, 89, 93, 94, 100, 121,
    136, 160,
)
PRIVATE_EVENTS = ("SUCCESS", "OUT_OF_BOUNDS", "METHOD_FAILURE", "COLLISION")


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def copy_project_file(destination: Path, source: Path) -> int:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    return source.stat().st_size


def safe_result_path(path: str) -> Path:
    prefix = f"/results/{RESULTS}/"
    if not isinstance(path, str) or not path.startswith(prefix):
        raise ValueError(f"Unexpected recording path: {path!r}")
    rel = PurePosixPath(path.lstrip("/"))
    if any(part in ("", ".", "..") for part in rel.parts):
        raise ValueError(f"Unsafe recording path: {path!r}")
    absolute = (ROOT / Path(*rel.parts)).resolve()
    if not absolute.is_relative_to((ROOT / "results" / RESULTS).resolve()):
        raise ValueError(f"Recording escaped results directory: {path!r}")
    return absolute


def copy_replay_files(record: dict, output: Path) -> int:
    url = record["browser_record"]
    source = safe_result_path(url)
    total = copy_project_file(output / url.lstrip("/"), source)
    replay = json.loads(source.read_text())
    if replay.get("format") != "drone-playground-mujoco-replay-v1":
        raise ValueError(f"Unsupported replay format: {source}")
    if not isinstance(replay.get("assets"), list):
        raise ValueError(f"Missing MJCF asset catalog: {source}")
    for name in [replay["model"], *replay["assets"]]:
        rel = PurePosixPath(name)
        if rel.is_absolute() or not rel.parts or any(x in (".", "..") for x in rel.parts):
            raise ValueError(f"Unsafe MJCF asset name: {name!r}")
        src = (source.parent / Path(*rel.parts)).resolve()
        if not src.is_relative_to(source.parent.resolve()):
            raise ValueError(f"Asset escapes replay folder: {name!r}")
        total += copy_project_file(output / url.lstrip("/").rsplit("/", 1)[0] / Path(*rel.parts), src)
    return total


def public_url(path: str, base_path: str) -> str:
    safe_result_path(path)
    return base_path.rstrip("/") + path


def case_link(item: dict) -> str:
    fields = {
        "mode": item["mode"],
        "task": item["task"],
        "method": item["method"],
        "controller": item["controller"],
    }
    if item.get("planner"):
        fields["planner"] = item["planner"]
    if item.get("scene"):
        fields["scene"] = item["scene"]
    if item.get("suite"):
        fields["suite"] = item["suite"]
    return "../../docs/command-builder.html?" + urlencode(fields)


def archive_html(cases: list[dict], full: dict) -> str:
    rows = []
    for item in cases:
        events = ", ".join(
            f"{r['scene']}: {r['event'].replace('_', ' ')}"
            for r in item["recordings"]
        )
        method = (
            f"{item['planner'].upper()} + {item['controller']}"
            if item.get("planner") else item["controller"]
        )
        rows.append(
            "<tr><td><a href='" + html.escape(case_link(item), quote=True) + "'>"
            + str(item["id"]).zfill(3) + "</a></td><td>"
            + html.escape(item["task"].replace("_", " ").title()) + "</td><td>"
            + html.escape(method) + "</td><td>" + html.escape(item["mode"])
            + "</td><td>" + html.escape(events) + "</td></tr>"
        )
    return """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Drone Playground — Public Flight Recordings</title>
<style>body{margin:0 auto;max-width:1100px;padding:32px 20px 70px;font:15px/1.7 system-ui,sans-serif;color:#202124}
h1{font-size:clamp(28px,4vw,42px);line-height:1.2}a{color:#a93414}
table{width:100%;border-collapse:collapse;font-size:13px}th,td{text-align:left;padding:13px 9px;border-bottom:1px solid #e9e9e9}
th{background:#f7f8f9}tr:nth-child(even){background:#fafafa}td:last-child{min-width:175px}
.scroll{overflow-x:auto}p{max-width:78ch;color:#52585c}</style></head><body>
<p><a href="../../docs/command-builder.html">← Interactive flight explorer</a></p>
<h1>Public native flight recordings</h1>
<p>These are selected real recordings from the historical native-method acceptance.
The full original suite executed """ + str(full["attempted_cases"]) + """ method cases and recorded """ + str(sum(full["event_counts"].values())) + """ Episodes.
This public selection is for inspection, not a statistical success-rate estimate.
Training checkpoints and native worker executables are not included.</p>
<div class="scroll"><table><thead><tr><th>Case</th><th>Task</th><th>Method</th><th>Mode</th><th>Recorded outcomes</th></tr></thead>
<tbody>""" + "".join(rows) + """</tbody></table></div>
<p>Source: historical results from 2026-10-17. Case IDs and terminal events are preserved.
<a href="public-provenance.json">Machine-readable provenance</a>.</p>
</body></html>"""


def build(output: Path, base_path: str) -> dict:
    if base_path != "/" and not re.fullmatch(r"/(?:[A-Za-z0-9_-]+/)+", base_path):
        raise ValueError("--base-path must be / or a path such as /drone_playground/")
    source_dir = ROOT / "results" / RESULTS
    source_catalog = json.loads((source_dir / "preview-catalog.json").read_text())
    source_audit = json.loads((source_dir / "replay-audit.json").read_text())
    by_id = {item["id"]: item for item in source_catalog["cases"]}
    if len(by_id) != source_audit["catalog_cases"]:
        raise ValueError("Native manifest and catalog case counts differ")
    if not all(case in by_id for case in PUBLIC_CASES):
        raise ValueError("A chosen public case is not in the native catalog")
    if output.exists():
        if any(output.iterdir()):
            raise FileExistsError(f"Choose an empty new output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    for name in ("environments.css", "task-preview.js"):
        copy_project_file(output / "docs" / name, ROOT / "docs" / name)
    builder = (ROOT / "docs/command-builder.html").read_text()
    builder = builder.replace('href="../README.md"', 'href="../index.html"')
    builder = builder.replace(
        'href="README.md"',
        'href="https://github.com/sycamore-yu/drone_playground"',
    )
    for name in ("installation", "benchmark"):
        builder = builder.replace(
            f'href="{name}.md"',
            f'href="https://github.com/sycamore-yu/drone_playground/blob/main/docs/{name}.md"',
        )
    builder = builder.replace(
        "Native combinations</span>", "Public cases</span>"
    ).replace(
        "Based on completed local evaluation runs",
        "Curated recordings from 172 historical native-method cases",
    ).replace(
        "Recorded method library", "Public flight library"
    ).replace(
        "Search verified command combinations and select one to load its recorded flight.",
        "Browse the selected published recordings and load their actual flights.",
    )
    (output / "docs/command-builder.html").write_text(builder)
    if not (ROOT / "docs/viewer/dist/index.html").is_file():
        raise FileNotFoundError("Build the official MuJoCo viewer first: npm run build --prefix docs/viewer")
    shutil.copytree(ROOT / "docs/viewer/dist", output / "docs/viewer/dist")

    cases = []
    bytes_copied = 0
    event_count = collections.Counter()
    for case_id in PUBLIC_CASES:
        item = dict(by_id[case_id])
        recordings = []
        for original in item["recordings"]:
            record = dict(original)
            bytes_copied += copy_replay_files(record, output)
            record["browser_record"] = public_url(record["browser_record"], base_path)
            # The original RScope is often tens of MB and the raw report may
            # contain local runtime configuration; expose only browser playback.
            record["rscope_record"] = None
            record["report"] = None
            recordings.append(record)
            event_count[record["event"]] += 1
        if not recordings:
            raise ValueError(f"Public case {case_id} has no native recordings")
        item["recordings"] = recordings
        cases.append(item)

    release = {
        "format": source_catalog["format"],
        "source": source_catalog["source"],
        "summary": {
            "native_methods": len(cases),
            "recorded_methods": len(cases),
            "recorded_episodes": sum(event_count.values()),
            "terminal_events": dict(event_count),
        },
        "cases": cases,
    }
    out_results = output / "results" / RESULTS
    write_json(out_results / "preview-catalog.json", release)
    provenance = {
        "type": "curated-browser-preview",
        "source_revision": source_catalog["source"],
        "native_acceptance_cases": source_audit["attempted_cases"],
        "native_acceptance_terminal_events": source_audit["event_counts"],
        "published_case_ids": list(PUBLIC_CASES),
        "published_episodes": sum(event_count.values()),
        "published_terminal_events": dict(event_count),
        "base_path": base_path,
    }
    write_json(out_results / "public-provenance.json", provenance)
    (out_results / "index.html").write_text(archive_html(cases, source_audit))
    (output / ".nojekyll").write_text("")
    (output / "index.html").write_text(
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>Drone Playground — Interactive Flight Explorer</title>'
        '</head>'
        '<body><p><a href="docs/command-builder.html">Open the interactive flight explorer</a></p>'
        '<script>location.replace(new URL("docs/command-builder.html"+location.search+location.hash,location.href))</script>'
        '<noscript><meta http-equiv="refresh" content="0;url=docs/command-builder.html"></noscript>'
        '</body></html>'
    )
    return {
        "output": str(output),
        "base_path": base_path,
        "published_cases": len(cases),
        "published_episodes": sum(event_count.values()),
        "recording_assets_bytes": bytes_copied,
        "historical_acceptance_cases": source_audit["attempted_cases"],
        "outcomes": dict(event_count),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-path", default="/drone_playground/")
    args = parser.parse_args()
    print(json.dumps(build(args.output.resolve(), args.base_path), indent=2))


if __name__ == "__main__":
    main()
