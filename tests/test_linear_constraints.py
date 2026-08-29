from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app


class LinearConstraintsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = [
            {"A": "sim", "B": "sim", "C": "sim"},
            {"A": "sim", "B": "nao", "C": "sim"},
            {"A": "nao", "B": "sim", "C": "nao"},
            {"A": "nao", "B": "nao", "C": "nao"},
        ]
        self.worlds = [
            {"values": row, "count": 1}
            for row in self.rows
        ]
        self.rule = {
            "antecedent": [{"attribute": "B", "value": "sim"}],
            "consequent": [{"attribute": "A", "value": "sim"}],
            "support": 0.25,
            "confidence": 0.5,
            "lift": 1.0,
        }

    def test_apriori_support_and_confidence_become_constraints(self) -> None:
        with patch("app.learned_association_rules", return_value=(self.rule,)):
            a_ub, b_ub, records = app.build_linear_constraints(
                self.worlds,
                self.rows,
                target={"attribute": "A", "value": "sim"},
                base=[{"attribute": "B", "value": "sim"}],
            )

        kinds = [record["kind"] for record in records]
        self.assertIn("apriori_rule_confidence", kinds)
        # The pairwise constraint already anchors this two-item support, so it
        # is deduplicated instead of being added a second time.
        confidence_record = next(
            record for record in records if record["kind"] == "apriori_rule_confidence"
        )
        self.assertFalse(confidence_record["supportConstraintAdded"])
        self.assertEqual(len(a_ub), len(b_ub))
        self.assertTrue(all(isinstance(row, dict) for row in a_ub))
        self.assertFalse(any(kind.startswith("selected") for kind in kinds))

    def test_query_rule_is_never_fabricated(self) -> None:
        missing = app.query_association_rule(
            [self.rule],
            antecedent=[{"attribute": "C", "value": "sim"}],
            consequent=[{"attribute": "A", "value": "sim"}],
        )
        present = app.query_association_rule(
            [self.rule],
            antecedent=self.rule["antecedent"],
            consequent=self.rule["consequent"],
        )
        self.assertIsNone(missing)
        self.assertIs(present, self.rule)

    def test_zero_empirical_joint_adds_possible_world_for_the_lp(self) -> None:
        rows = [
            {"A": "sim", "B": "nao"},
            {"A": "nao", "B": "sim"},
        ]
        worlds = [{"values": row, "count": 1} for row in rows]
        completed, added = app.complete_unobserved_query_worlds(
            worlds,
            rows,
            target={"attribute": "A", "value": "sim"},
            base=[{"attribute": "B", "value": "sim"}],
        )

        self.assertEqual(added, 1)
        self.assertEqual(len(completed), 3)
        self.assertEqual(completed[-1]["count"], 0)
        self.assertTrue(completed[-1]["queryCompletion"])
        self.assertEqual(completed[-1]["values"], {"A": "sim", "B": "sim"})

    def test_auditable_txt_uses_the_exact_model_solved_by_linprog(self) -> None:
        target = {"attribute": "A", "value": "sim"}
        base = [{"attribute": "B", "value": "sim"}]
        with patch("app.learned_association_rules", return_value=(self.rule,)):
            constraints = app.build_linear_constraints(
                self.worlds,
                self.rows,
                target=target,
                base=base,
            )

        with (
            patch("app.cached_linear_constraint_model", return_value=constraints),
            patch("app.learned_association_rules", return_value=(self.rule,)),
        ):
            result = app.solve_linear_interval(self.worlds, self.rows, target, base)
            exported = app.full_linear_program_text(
                self.worlds,
                self.rows,
                target,
                base,
                result,
            )

        self.assertTrue(result["ok"])
        self.assertEqual(result["worldVariables"], 4)
        self.assertEqual(result["solverVariables"], 5)
        self.assertIn(result["modelDigest"], exported)
        self.assertIn("c_lower=", exported)
        self.assertIn("c_upper_as_min=", exported)
        self.assertIn("A_ub[0]=", exported)
        self.assertIn("b_ub[0]=", exported)
        self.assertIn("A_eq[0]=", exported)
        self.assertIn("b_eq[0]=", exported)
        self.assertIn("bounds[0:y_0001]", exported)
        self.assertNotIn("soma(x_w)", exported)
        self.assertNotIn("soma(x_w onde", exported)


if __name__ == "__main__":
    unittest.main()
