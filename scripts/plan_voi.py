#!/usr/bin/env python3
"""Planejador de Valor da Informacao independente da interface Flask."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.solve_query import DEFAULT_DATASET, load_dataset, parse_condition, validate_conditions
from voi import DEFAULT_MAX_NODES, build_conditional_plan


def parse_observable(text: str) -> tuple[str, float]:
    attribute, separator, cost_text = text.partition(":")
    attribute = attribute.strip()
    if not attribute:
        raise argparse.ArgumentTypeError("Use atributo:custo, por exemplo ph:1.")
    try:
        cost = float(cost_text) if separator else 1.0
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"Custo invalido em {text}.") from error
    if cost <= 0:
        raise argparse.ArgumentTypeError("O custo precisa ser positivo.")
    return attribute, cost


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Constroi o plano condicional guloso da Figura 3 de Ghosh e "
            "Ramakrishnan (2019) para uma cultura."
        )
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--crop", default="rice", help="Cultura da consulta ground.")
    parser.add_argument("--budget", type=float, default=2.0)
    parser.add_argument(
        "--observable",
        action="append",
        type=parse_observable,
        help="Atributo e custo. Pode repetir. Exemplo: --observable rainfall:1",
    )
    parser.add_argument(
        "--evidence",
        action="append",
        type=parse_condition,
        default=[],
        help="Evidencia ja conhecida. Exemplo: --evidence ph=acido",
    )
    parser.add_argument("--max-nodes", type=int, default=DEFAULT_MAX_NODES)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    data = load_dataset(args.dataset)
    target = {"attribute": "label", "value": args.crop}
    try:
        target = validate_conditions([target], data["domains"])[0]
        evidence = validate_conditions(args.evidence, data["domains"])
        configured = args.observable or [
            (attribute, 1.0)
            for attribute in data["numericAttributes"]
        ]
        observables = []
        costs = {}
        for attribute, cost in configured:
            if attribute not in data["numericAttributes"]:
                raise ValueError(f"Observavel agricola invalido: {attribute}")
            if attribute in costs:
                raise ValueError(f"Observavel repetido: {attribute}")
            observables.append(attribute)
            costs[attribute] = cost
        result = build_conditional_plan(
            data["worlds"],
            target,
            observables,
            costs,
            args.budget,
            evidence,
            max_nodes=args.max_nodes,
        )
    except ValueError as error:
        print(f"Erro: {error}", file=sys.stderr)
        return 2

    result["computation"] = {
        "probabilityInference": "enumeracao exata dos mundos observados de Omega",
        "observationOptimizer": "plano condicional guloso da Figura 3",
        "linearSolverUsed": False,
    }
    output = json.dumps(result, ensure_ascii=False, indent=2)
    print(output)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
        print(f"\nPlano salvo em: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
