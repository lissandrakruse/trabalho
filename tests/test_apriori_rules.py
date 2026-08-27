from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apriori_rules import mine_apriori_rules


class AprioriRulesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.worlds = [
            {"values": {"A": "sim", "B": "sim", "C": "sim"}, "count": 30},
            {"values": {"A": "sim", "B": "nao", "C": "sim"}, "count": 30},
            {"values": {"A": "sim", "B": "nao", "C": "nao"}, "count": 20},
            {"values": {"A": "nao", "B": "sim", "C": "sim"}, "count": 20},
        ]

    def test_uses_world_counts_as_transaction_weights(self) -> None:
        result = mine_apriori_rules(
            self.worlds,
            total=100,
            min_support=0.1,
            min_confidence=0.0,
            max_itemset_size=3,
        )
        rule = next(
            rule
            for rule in result["rules"]
            if rule["antecedent"] == [{"attribute": "B", "value": "sim"}]
            and rule["consequent"] == [{"attribute": "A", "value": "sim"}]
        )
        self.assertAlmostEqual(rule["support"], 0.30)
        self.assertAlmostEqual(rule["confidence"], 0.60)
        self.assertAlmostEqual(rule["lift"], 0.75)

    def test_lift_below_one_is_not_filtered_as_quality(self) -> None:
        result = mine_apriori_rules(
            self.worlds,
            total=100,
            min_support=0.1,
            min_confidence=0.0,
            max_itemset_size=3,
        )
        self.assertTrue(any(rule["lift"] < 1 for rule in result["rules"]))

    def test_generates_multi_condition_antecedents(self) -> None:
        result = mine_apriori_rules(
            self.worlds,
            total=100,
            min_support=0.1,
            min_confidence=0.0,
            max_itemset_size=3,
        )
        self.assertTrue(any(len(rule["antecedent"]) == 2 for rule in result["rules"]))
        self.assertEqual(result["algorithm"], "Apriori")
        self.assertEqual(result["omegaWorlds"], 4)


if __name__ == "__main__":
    unittest.main()
