from __future__ import annotations

import json
import sys
from pathlib import Path

from scripts.solve_query import (
    DEFAULT_DATASET,
    compute_query,
    load_dataset,
    main as solve_query_main,
    validate_conditions,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "reports" / "generated" / "solver_resultado_padrao.json"
DEFAULT_TARGET = {"attribute": "label", "value": "rice"}
DEFAULT_CONDITIONS = [
    {"attribute": "ph", "value": "acido"},
    {"attribute": "rainfall", "value": "alto"},
]


def run_default_query() -> int:
    data = load_dataset(DEFAULT_DATASET)
    target = validate_conditions([DEFAULT_TARGET], data["domains"])[0]
    conditions = validate_conditions(DEFAULT_CONDITIONS, data["domains"])
    result = compute_query(data, target, conditions)

    output_text = json.dumps(result, ensure_ascii=False, indent=2)
    DEFAULT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_OUTPUT.write_text(output_text + "\n", encoding="utf-8")

    print("Solver executado com sucesso.")
    print("Consulta padrao: P(label=rice | ph=acido, rainfall=alto)")
    print(f"Suporte P(A e B): {result['support']:.3f}")
    confidence = result["confidence"]
    lift = result["lift"]
    print(f"Confianca P(A | B): {confidence:.3f}" if confidence is not None else "Confianca P(A | B): -")
    print(f"Lift: {lift:.3f}" if lift is not None else "Lift: -")

    linear = result["linear"]
    if linear.get("ok"):
        print(f"Intervalo linear: {linear['lower']:.3f} <= P(A | B) <= {linear['upper']:.3f}")
        print(f"Variaveis: {linear['variables']}")
        print(f"Restricoes: {linear['constraints']}")
    else:
        print(f"Intervalo linear: {linear.get('error', 'nao calculado')}")

    print(f"JSON completo salvo em: {DEFAULT_OUTPUT}")
    return 0


def main() -> int:
    if len(sys.argv) > 1:
        return solve_query_main()
    return run_default_query()


if __name__ == "__main__":
    raise SystemExit(main())
