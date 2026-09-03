from __future__ import annotations

import math
import sys
import unittest
from itertools import product
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from voi import (
    binary_entropy,
    build_conditional_plan,
    scenario_summary,
    subset_value_of_information,
    value_of_information,
)


class ValueOfInformationTest(unittest.TestCase):
    def test_perfect_observable_reduces_one_bit_of_uncertainty(self) -> None:
        worlds = [
            {"values": {"query": "sim", "sensor": "positivo"}, "count": 50},
            {"values": {"query": "nao", "sensor": "negativo"}, "count": 50},
        ]
        result = value_of_information(
            worlds,
            {"attribute": "query", "value": "sim"},
            "sensor",
        )

        self.assertAlmostEqual(binary_entropy(0.5), 1.0)
        self.assertAlmostEqual(result["currentEntropy"], 1.0)
        self.assertAlmostEqual(result["expectedEntropy"], 0.0)
        self.assertAlmostEqual(result["voi"], 1.0)

    def test_evidence_creates_the_conditioned_scenario(self) -> None:
        worlds = [
            {"values": {"query": "sim", "region": "norte"}, "count": 30},
            {"values": {"query": "nao", "region": "norte"}, "count": 10},
            {"values": {"query": "sim", "region": "sul"}, "count": 5},
            {"values": {"query": "nao", "region": "sul"}, "count": 55},
        ]
        summary = scenario_summary(
            worlds,
            {"attribute": "query", "value": "sim"},
            [{"attribute": "region", "value": "norte"}],
        )

        self.assertAlmostEqual(summary["probability"], 0.75)
        self.assertEqual(summary["scenarioMass"], 40)

    def test_reproduces_temperature_example_from_the_article(self) -> None:
        worlds = []
        transition = {
            ("lo", "lo"): 0.7,
            ("lo", "hi"): 0.3,
            ("hi", "lo"): 0.3,
            ("hi", "hi"): 0.7,
        }
        for t1, t2, t3 in product(("lo", "hi"), repeat=3):
            probability = 0.5 * transition[(t1, t2)] * transition[(t2, t3)]
            worlds.append(
                {
                    "values": {
                        "T1": t1,
                        "T2": t2,
                        "T3": t3,
                        "heat_on": "sim" if "lo" in (t1, t2, t3) else "nao",
                    },
                    "count": probability,
                }
            )

        target = {"attribute": "heat_on", "value": "sim"}
        baseline = scenario_summary(worlds, target)
        subset = subset_value_of_information(worlds, target, ["T1", "T3"])
        plan = build_conditional_plan(
            worlds,
            target,
            ["T1", "T2", "T3"],
            {"T1": 1, "T2": 1, "T3": 1},
            budget=2,
        )

        self.assertAlmostEqual(baseline["probability"], 0.755)
        self.assertAlmostEqual(baseline["entropy"], 0.8032566998)
        self.assertAlmostEqual(subset["expectedEntropy"], 0.1805639517)
        self.assertAlmostEqual(subset["voi"], 0.6226927481)
        self.assertAlmostEqual(plan["planVoi"], subset["voi"])
        self.assertEqual(plan["tree"]["choice"]["observable"], "T1")

    def test_cost_is_a_budget_constraint_not_the_ranking_objective(self) -> None:
        worlds = [
            {"values": {"q": "sim", "best": "a", "cheap": "x"}, "count": 45},
            {"values": {"q": "nao", "best": "b", "cheap": "x"}, "count": 45},
            {"values": {"q": "sim", "best": "a", "cheap": "y"}, "count": 5},
            {"values": {"q": "nao", "best": "b", "cheap": "y"}, "count": 5},
        ]
        plan = build_conditional_plan(
            worlds,
            {"attribute": "q", "value": "sim"},
            ["best", "cheap"],
            {"best": 2, "cheap": 1},
            budget=1,
        )

        self.assertIsNone(plan["tree"]["choice"])
        self.assertEqual(plan["tree"]["stopReason"], "no_utility_gain")
        self.assertTrue(math.isclose(plan["planVoi"], 0.0, abs_tol=1e-12))


if __name__ == "__main__":
    unittest.main()
