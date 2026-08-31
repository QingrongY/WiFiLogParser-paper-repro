from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_template_baselines", ROOT / "scripts" / "run_template_baselines.py"
)
assert SPEC and SPEC.loader
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class MetricDefinitionTests(unittest.TestCase):
    def test_perfect_three_class_and_strict_fea(self) -> None:
        lines = ["a", "b", "c"]
        gt = {
            0: {"event": "connect", "ap_candidates": {"ap-a"}, "client_candidates": {"client-a"}},
            1: {"event": "disconnect", "ap_candidates": {"ap-b"}, "client_candidates": {"client-b"}},
        }
        predictions = [
            {"line_index": 0, "event": "connect", "ap_id": "ap-a", "client_id": "client-a"},
            {"line_index": 1, "event": "disconnect", "ap_id": "ap-b", "client_id": "client-b"},
            {"line_index": 2, "event": "other", "ap_id": "", "client_id": ""},
        ]
        metrics = RUNNER._evaluate(predictions, gt, lines)
        self.assertEqual(metrics["event_macro_f1_3class"], 1.0)
        self.assertEqual(metrics["strict_fea_paper_definition"], 1.0)

    def test_strict_fea_does_not_include_event_accuracy(self) -> None:
        lines = ["a"]
        gt = {
            0: {"event": "connect", "ap_candidates": {"ap-a"}, "client_candidates": {"client-a"}},
        }
        predictions = [
            {"line_index": 0, "event": "other", "ap_id": "ap-a", "client_id": "client-a"},
        ]
        metrics = RUNNER._evaluate(predictions, gt, lines)
        self.assertEqual(metrics["strict_fea_paper_definition"], 1.0)
        self.assertAlmostEqual(metrics["line_index_style_fea_with_event_and_partial"], 2 / 3)

    def test_partial_containment_is_reported_but_not_strict(self) -> None:
        lines = ["a"]
        gt = {
            0: {"event": "connect", "ap_candidates": {"ap-123"}, "client_candidates": {"client-a"}},
        }
        predictions = [
            {
                "line_index": 0,
                "event": "connect",
                "ap_id": "prefix-ap-123-suffix",
                "client_id": "client-a",
            },
        ]
        metrics = RUNNER._evaluate(predictions, gt, lines)
        self.assertEqual(metrics["strict_fea_paper_definition"], 0.5)
        self.assertEqual(metrics["line_index_style_fea_with_event_and_partial"], 1.0)
        self.assertEqual(metrics["partial_only_matches"]["ap"], 1)


if __name__ == "__main__":
    unittest.main()
