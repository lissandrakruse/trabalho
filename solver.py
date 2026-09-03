from __future__ import annotations

import json
import sys
from pathlib import Path

import app as project_app
from scripts.solve_query import (
    DEFAULT_DATASET,
    compute_query,
    load_dataset,
    main as solve_query_main,
    validate_conditions,
)
from voi import build_conditional_plan


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "reports" / "generated" / "solver_resultado_padrao.json"
DEFAULT_VOI_OUTPUT = ROOT / "reports" / "generated" / "plano_voi_padrao.json"
DEFAULT_ACTIVE_OUTPUT = ROOT / "reports" / "generated" / "selecao_ativa_padrao.json"
DEFAULT_TARGET = {"attribute": "label", "value": "rice"}
DEFAULT_ACTIVE_TARGET = {"attribute": "label", "value": "apple"}
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
    support = result["support"]
    print(f"Suporte da regra: {support:.3f}" if support is not None else "Suporte da regra: -")
    confidence = result["confidence"]
    lift = result["lift"]
    print(f"Confianca da regra: {confidence:.3f}" if confidence is not None else "Confianca da regra: -")
    print(f"Lift: {lift:.3f}" if lift is not None else "Lift: -")
    print(f"Probabilidade empirica completa P(A e B), somente para auditoria: {result['pAB']!r}")
    print("A resposta empirica da consulta nao e fixada como restricao do PL.")
    print("O PL usa p completo +/- 0.001; nenhuma probabilidade e arredondada.")

    linear = result["linear"]
    if linear.get("ok"):
        print(f"Intervalo linear completo: {linear['lower']!r} <= P(A | B) <= {linear['upper']!r}")
        print(f"Variaveis: {linear['variables']}")
        print(f"Restricoes: {linear['constraints']}")
    else:
        print(f"Intervalo linear: {linear.get('error', 'nao calculado')}")

    observables = list(data["numericAttributes"])
    voi_plan = build_conditional_plan(
        data["worlds"],
        DEFAULT_TARGET,
        observables,
        {attribute: 1.0 for attribute in observables},
        budget=2.0,
    )
    DEFAULT_VOI_OUTPUT.write_text(
        json.dumps(voi_plan, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    first_choice = (voi_plan["tree"].get("choice") or {}).get("observable", "nenhuma")
    print("\nPlanejador de Valor da Informacao executado.")
    print(f"Primeira medicao: {first_choice}")
    print(f"Entropia inicial: {voi_plan['initialEntropy']:.6f} bits")
    print(f"Entropia final esperada: {voi_plan['expectedFinalEntropy']:.6f} bits")
    print(f"VoI do plano: {voi_plan['planVoi']:.6f} bits")
    print("O planejador de VoI usa inferencia e busca gulosa; nao chama o linprog.")

    active = project_app.compute_active_selection(
        {
            "target": DEFAULT_ACTIVE_TARGET,
            "conditions": DEFAULT_CONDITIONS,
            "budget": 25,
            "minimumLiteralOverlap": 2,
            "maxCandidates": 80,
        }
    )
    DEFAULT_ACTIVE_OUTPUT.write_text(
        json.dumps(active, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("\nSelecao ativa de restricoes executada.")
    print(
        f"Restricoes selecionadas: {active['selectedCount']} de "
        f"{active['candidatePool']['evaluated']} candidatas relevantes"
    )
    print(
        f"Largura: {active['baseModel']['width']:.6f} -> "
        f"{active['activeSelection']['width']:.6f} "
        f"({100 * active['activeSelection']['relativeWidthReduction']:.2f}% de reducao)"
    )
    print(
        f"Poda exata por factibilidade: "
        f"{100 * active['solverEffort']['exactPruningRate']:.2f}%"
    )
    print(
        f"Chamadas candidatas evitadas no total: "
        f"{100 * active['solverEffort']['totalAvoidanceRate']:.2f}%"
    )
    print("Somente os extremos p_L ou p_U violados foram reotimizados com HiGHS.")

    print(f"JSON completo salvo em: {DEFAULT_OUTPUT}")
    print(f"Plano de VoI salvo em: {DEFAULT_VOI_OUTPUT}")
    print(f"Selecao ativa salva em: {DEFAULT_ACTIVE_OUTPUT}")
    return 0


def main() -> int:
    if len(sys.argv) > 1:
        return solve_query_main()
    return run_default_query()


if __name__ == "__main__":
    raise SystemExit(main())
