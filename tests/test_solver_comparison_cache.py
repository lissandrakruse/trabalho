from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app


class SolverComparisonCacheTest(unittest.TestCase):
    def setUp(self) -> None:
        app._cached_query_result.cache_clear()
        app._cached_standalone_solver_result.cache_clear()
        app._cached_solver_comparison.cache_clear()

    def tearDown(self) -> None:
        app._cached_query_result.cache_clear()
        app._cached_standalone_solver_result.cache_clear()
        app._cached_solver_comparison.cache_clear()

    def test_same_query_reuses_main_solver_result(self) -> None:
        payload_a = {
            "conditions": [{"value": "acido", "attribute": "ph"}],
            "target": {"value": "rice", "attribute": "label"},
        }
        payload_b = {
            "target": {"attribute": "label", "value": "rice"},
            "conditions": [{"attribute": "ph", "value": "acido"}],
        }
        expected = {"ok": True, "linear": {"solverMethod": "highs-ipm"}}

        with patch("app._compute_query_uncached", return_value=expected) as compute:
            first = app.compute_query(payload_a)
            second = app.compute_query(payload_b)

        self.assertIs(first, second)
        self.assertEqual(compute.call_count, 1)

    def test_solver_method_must_be_one_of_the_executed_engines(self) -> None:
        self.assertEqual(app.solver_engine_for_method("highs-ds")["name"], "HiGHS Dual Simplex")
        with self.assertRaisesRegex(ValueError, "Metodo de solver invalido"):
            app.solver_engine_for_method("solver-inexistente")

    def test_same_payload_reuses_solver_results_for_pdf(self) -> None:
        payload_a = {
            "conditions": [{"value": "acido", "attribute": "ph"}],
            "target": {"value": "rice", "attribute": "label"},
        }
        payload_b = {
            "target": {"attribute": "label", "value": "rice"},
            "conditions": [{"attribute": "ph", "value": "acido"}],
        }
        expected = {"ok": True, "solverEngineResults": ["highs", "highs-ds", "highs-ipm"]}

        with patch("app._build_solver_comparison_uncached", return_value=expected) as build:
            first = app.build_solver_comparison(payload_a)
            second = app.build_solver_comparison(payload_b)

        self.assertIs(first, second)
        self.assertEqual(build.call_count, 1)


if __name__ == "__main__":
    unittest.main()
