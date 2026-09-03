#!/usr/bin/env python3
"""Executa a selecao ativa de restricoes fora da interface Flask."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app


def parse_condition(value: str) -> dict[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Use atributo=valor.")
    attribute, item_value = value.split("=", 1)
    return {"attribute": attribute.strip(), "value": item_value.strip()}


def parser() -> argparse.ArgumentParser:
    created = argparse.ArgumentParser(
        description="Seleciona restricoes Apriori que estreitam P(A|B)."
    )
    created.add_argument("--target", type=parse_condition, default=parse_condition("label=rice"))
    created.add_argument("--condition", type=parse_condition, action="append", default=[])
    created.add_argument("--budget", type=int, default=25)
    created.add_argument("--minimum-overlap", type=int, default=2)
    created.add_argument("--max-candidates", type=int, default=80)
    created.add_argument("--output", type=Path)
    return created


def main() -> int:
    args = parser().parse_args()
    conditions = args.condition or [
        parse_condition("ph=acido"),
        parse_condition("rainfall=alto"),
    ]
    result = app.compute_active_selection(
        {
            "target": args.target,
            "conditions": conditions,
            "budget": args.budget,
            "minimumLiteralOverlap": args.minimum_overlap,
            "maxCandidates": args.max_candidates,
        }
    )
    summary = {
        "query": {"target": result["target"], "conditions": result["conditions"]},
        "method": result["method"],
        "candidatePool": result["candidatePool"],
        "baseModel": result["baseModel"],
        "activeSelection": result["activeSelection"],
        "baselines": result["baselines"],
        "solverEffort": result["solverEffort"],
        "fullModel": result["fullModel"],
        "recoveredFullModelReduction": result["recoveredFullModelReduction"],
        "limitations": result["limitations"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"\nResultado completo salvo em: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
