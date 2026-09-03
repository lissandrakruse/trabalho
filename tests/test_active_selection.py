import unittest

from active_selection import select_active_constraints


class ActiveConstraintSelectionTest(unittest.TestCase):
    @staticmethod
    def model() -> dict:
        return {
            "solverVariables": 2,
            "objectiveLower": {0: 1.0},
            "objectiveUpperAsMin": {0: -1.0},
            "aUb": [
                {0: 1.0, 1: -1.0},
                {0: 1.0, 1: -0.4},
                {0: -1.0, 1: 0.2},
            ],
            "bUb": [0.0, 0.0, 0.0],
            "aEq": [{1: 1.0}],
            "bEq": [1.0],
            "bounds": [(0.0, None), (0.0, None)],
            "records": [
                {
                    "kind": "marginal",
                    "conditions": [{"attribute": "label", "value": "rice"}],
                    "rowIndexes": [0],
                },
                {
                    "kind": "apriori_rule_support",
                    "conditions": [
                        {"attribute": "label", "value": "rice"},
                        {"attribute": "ph", "value": "acido"},
                    ],
                    "value": 0.4,
                    "rowIndexes": [1],
                },
                {
                    "kind": "apriori_rule_support",
                    "conditions": [
                        {"attribute": "label", "value": "rice"},
                        {"attribute": "rainfall", "value": "alto"},
                    ],
                    "value": 0.2,
                    "rowIndexes": [2],
                },
            ],
        }

    def test_reoptimizes_only_the_violated_extreme(self):
        result = select_active_constraints(
            self.model(),
            {"attribute": "label", "value": "rice"},
            [
                {"attribute": "ph", "value": "acido"},
                {"attribute": "rainfall", "value": "alto"},
            ],
            budget=2,
            minimum_literal_overlap=2,
        )

        self.assertAlmostEqual(result["baseModel"]["width"], 1.0)
        self.assertAlmostEqual(result["activeSelection"]["lower"], 0.2)
        self.assertAlmostEqual(result["activeSelection"]["upper"], 0.4)
        self.assertAlmostEqual(result["activeSelection"]["width"], 0.2)
        trace = result["activeSelection"]["selectionTrace"]
        self.assertEqual([item["requiredLpSolves"] for item in trace], [1, 1])
        self.assertEqual(result["solverEffort"]["selectedEndpointLpSolves"], 2)
        self.assertGreater(result["solverEffort"]["exactPruningSavedLpSolves"], 0)

    def test_stops_when_all_extremes_already_satisfy_candidates(self):
        model = self.model()
        model["aUb"][1] = {0: 1.0, 1: -1.1}
        model["records"] = [model["records"][0], model["records"][1]]
        result = select_active_constraints(
            model,
            {"attribute": "label", "value": "rice"},
            [{"attribute": "ph", "value": "acido"}],
            budget=1,
            minimum_literal_overlap=2,
        )

        self.assertEqual(result["selectedCount"], 0)
        self.assertEqual(
            result["activeSelection"]["stopReason"],
            "all_remaining_constraints_satisfied_by_extremes",
        )


if __name__ == "__main__":
    unittest.main()
