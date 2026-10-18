"""Validate the static Pages export contract without executing flight physics."""

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from build_pages_demo import PUBLIC_CASES, RESULTS, public_url, safe_result_path  # noqa: E402


class PagesExportContract(unittest.TestCase):
    def test_selection_is_unique_and_covers_all_tasks(self):
        catalog_file = ROOT / "results" / RESULTS / "preview-catalog.json"
        if not catalog_file.is_file():
            self.skipTest("Historical native episodes are available only on the research server")
        cases = json.loads(catalog_file.read_text())["cases"]
        selected = [case for case in cases if case["id"] in PUBLIC_CASES]
        self.assertEqual(len(set(PUBLIC_CASES)), len(PUBLIC_CASES))
        self.assertEqual(len(selected), len(PUBLIC_CASES))
        self.assertEqual(
            {case["task"] for case in selected},
            {"tracking", "racing", "static_oa", "dynamic_oa"},
        )
        self.assertTrue(all(case["recordings"] for case in selected))
        outcomes = {row["event"] for case in selected for row in case["recordings"]}
        self.assertEqual(outcomes, {"SUCCESS", "COLLISION", "OUT_OF_BOUNDS", "METHOD_FAILURE"})

    def test_replays_cannot_escape_historical_results(self):
        prefix = f"/results/{RESULTS}/runs/example/replays/scene-0000/"
        self.assertEqual(
            public_url(prefix + "browser_replay.json", "/drone_playground/"),
            "/drone_playground" + prefix + "browser_replay.json",
        )
        for path in (
            "/tmp/private.json",
            f"/results/{RESULTS}/../../.env",
            f"/results/{RESULTS}/runs/../../../secrets",
        ):
            with self.subTest(path=path), self.assertRaises(ValueError):
                safe_result_path(path)


if __name__ == "__main__":
    unittest.main()
