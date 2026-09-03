from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app
from article_conformity import check_no_rounding_policy, run_article_conformity


class ArticleConformityTest(unittest.TestCase):
    def test_fast_conformity_matrix_passes(self) -> None:
        result = run_article_conformity(include_active_selection=False)

        self.assertTrue(result["ok"])
        self.assertEqual(result["classification"], "conforme_com_adaptacao_explicita")
        self.assertEqual(result["intervalPolicy"]["computationalRounding"], False)
        self.assertEqual(result["summary"]["failed"], 0)

    def test_interval_uses_full_probability_as_center(self) -> None:
        probability = 33 / 218
        lower, upper = app.probability_interval(probability)

        self.assertAlmostEqual(lower, probability - 0.001)
        self.assertAlmostEqual(upper, probability + 0.001)
        self.assertNotAlmostEqual(lower, round(probability, 3) - 0.001, places=12)

    def test_robot_rejects_a_rounded_implementation(self) -> None:
        def rounded_again(value: float, width: float = 0.001) -> tuple[float, float]:
            center = round(value, 3)
            return max(0.0, center - width), min(1.0, center + width)

        with patch("app.probability_interval", side_effect=rounded_again):
            with self.assertRaisesRegex(Exception, "Limite inferior incorreto"):
                check_no_rounding_policy()


if __name__ == "__main__":
    unittest.main()
