#!/usr/bin/env python3
"""Gera o relatorio cientifico versionado da selecao ativa."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app


OUTPUT = ROOT / "output" / "pdf" / "relatorio_selecao_ativa_agricultura.pdf"
PAYLOAD = {
    "target": {"attribute": "label", "value": "apple"},
    "conditions": [
        {"attribute": "ph", "value": "acido"},
        {"attribute": "rainfall", "value": "alto"},
    ],
    "budget": 25,
    "minimumLiteralOverlap": 2,
    "maxCandidates": 80,
}


def main() -> int:
    result = app.compute_active_selection(PAYLOAD)
    app.write_active_selection_report(result, OUTPUT)
    print(f"Relatorio gerado: {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
