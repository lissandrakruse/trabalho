from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app


class VoiIntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.payload = {
            "target": {"attribute": "label", "value": "rice"},
            "budget": 2,
            "observables": [
                {"attribute": attribute, "cost": 1}
                for attribute in ("N", "P", "K", "temperature", "humidity", "ph", "rainfall")
            ],
        }

    def test_agricultural_plan_has_the_reproducible_reference_result(self) -> None:
        result = app.compute_voi_plan(self.payload)

        self.assertTrue(result["ok"])
        self.assertEqual(result["tree"]["choice"]["observable"], "rainfall")
        self.assertAlmostEqual(result["initialEntropy"], 0.26676498780302604)
        self.assertAlmostEqual(result["expectedFinalEntropy"], 0.10858243722262287)
        self.assertAlmostEqual(result["planVoi"], 0.15818255058040318)
        self.assertFalse(result["computation"]["linearSolverUsed"])

    def test_metadata_exposes_article_and_seven_observables(self) -> None:
        response = app.app.test_client().get("/api/metadata")
        result = response.get_json()["voi"]

        self.assertEqual(response.status_code, 200)
        self.assertEqual(result["article"]["doi"], "10.4204/EPTCS.306.14")
        self.assertEqual(len(result["observables"]), 7)

    def test_api_rejects_non_agricultural_observable(self) -> None:
        response = app.app.test_client().post(
            "/api/voi/plan",
            json={
                "target": {"attribute": "label", "value": "rice"},
                "budget": 2,
                "observables": [{"attribute": "label", "cost": 1}],
            },
        )

        self.assertEqual(response.status_code, 400)
        self.assertFalse(response.get_json()["ok"])

    def test_active_selection_reduces_interval_and_prunes_lp_solves(self) -> None:
        result = app.compute_active_selection(
            {
                "target": {"attribute": "label", "value": "rice"},
                "conditions": [
                    {"attribute": "ph", "value": "acido"},
                    {"attribute": "rainfall", "value": "alto"},
                ],
                "budget": 25,
                "minimumLiteralOverlap": 2,
                "maxCandidates": 80,
            }
        )

        self.assertEqual(result["selectedCount"], 25)
        self.assertEqual(result["candidatePool"]["evaluated"], 52)
        self.assertAlmostEqual(result["baseModel"]["width"], 0.023265306122449037)
        self.assertAlmostEqual(result["activeSelection"]["width"], 0.019084049685785698)
        self.assertGreater(
            result["baselines"]["random"]["meanWidth"],
            result["activeSelection"]["width"],
        )
        self.assertGreater(
            result["baselines"]["supportConfidence"]["width"],
            result["activeSelection"]["width"],
        )
        self.assertGreater(result["solverEffort"]["exactPruningRate"], 0.70)
        self.assertGreater(result["solverEffort"]["totalAvoidanceRate"], 0.97)


if __name__ == "__main__":
    unittest.main()
