from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.site_robot import close_to, require


class SiteRobotHelpersTest(unittest.TestCase):
    def test_close_to_accepts_solver_tolerance(self) -> None:
        self.assertTrue(close_to(0.15199999999999975, 0.152, 1e-8))
        self.assertFalse(close_to(0.15, 0.152, 1e-8))

    def test_require_reports_failed_verification(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "evidencia ausente"):
            require(False, "evidencia ausente")


if __name__ == "__main__":
    unittest.main()
