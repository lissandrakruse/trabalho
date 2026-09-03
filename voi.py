"""Valor da Informacao para consultas agricolas sobre mundos possiveis.

Este modulo implementa, para uma distribuicao finita de mundos, as Definicoes
5--10 e o algoritmo guloso da Figura 3 de Ghosh e Ramakrishnan (2019). A
distribuicao e a mesma representacao Omega usada pelo restante do projeto: cada
mundo contem uma valoracao completa e uma massa ``count``.

A consulta e ground e binaria (por exemplo, ``label=rice``). Um atributo
mensuravel e um observavel; cada valor do atributo e uma realizacao. Adicionar
uma realizacao ao cenario equivale a condicionar a distribuicao nessa evidencia.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any, Iterable


ARTICLE = {
    "title": "Value of Information in Probabilistic Logic Programs",
    "authors": ["Sarthak Ghosh", "C. R. Ramakrishnan"],
    "year": 2019,
    "venue": "EPTCS 306 / ICLP 2019",
    "pages": "71-84",
    "doi": "10.4204/EPTCS.306.14",
    "url": "https://arxiv.org/abs/1909.08234",
}

DEFAULT_OBSERVABLE_COST = 1.0
DEFAULT_MAX_NODES = 400
VOI_TOLERANCE = 1e-12


def _weight(world: dict[str, Any]) -> float:
    value = float(world.get("count", 0.0))
    if not math.isfinite(value) or value < 0:
        raise ValueError("Cada mundo deve possuir massa finita e nao negativa.")
    return value


def _evidence_map(evidence: Iterable[dict[str, str]] | None) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in evidence or []:
        attribute = str(item.get("attribute", "")).strip()
        value = str(item.get("value", "")).strip()
        if not attribute or not value:
            raise ValueError("Cada evidencia precisa de atributo e valor.")
        previous = result.get(attribute)
        if previous is not None and previous != value:
            raise ValueError(f"O cenario contem valores conflitantes para {attribute}.")
        result[attribute] = value
    return result


def _conditions(evidence: dict[str, str]) -> list[dict[str, str]]:
    return [
        {"attribute": attribute, "value": value}
        for attribute, value in sorted(evidence.items())
    ]


def _matches(values: dict[str, str], evidence: dict[str, str]) -> bool:
    return all(values.get(attribute) == value for attribute, value in evidence.items())


def _scenario_worlds(
    worlds: Iterable[dict[str, Any]],
    evidence: dict[str, str],
) -> tuple[list[tuple[dict[str, str], float]], float]:
    selected: list[tuple[dict[str, str], float]] = []
    total = 0.0
    for world in worlds:
        values = world.get("values")
        if not isinstance(values, dict):
            raise ValueError("Cada mundo deve possuir um dicionario values.")
        weight = _weight(world)
        if weight > 0 and _matches(values, evidence):
            selected.append((values, weight))
            total += weight
    if total <= 0:
        raise ValueError("O cenario informado nao possui massa de probabilidade.")
    return selected, total


def binary_entropy(probability: float) -> float:
    """Entropia de Shannon, em bits, da verdade de uma consulta ground."""

    probability = min(1.0, max(0.0, float(probability)))
    if probability <= 0.0 or probability >= 1.0:
        return 0.0
    return -(
        probability * math.log2(probability)
        + (1.0 - probability) * math.log2(1.0 - probability)
    )


def uncertainty_utility(probability: float) -> float:
    """Utilidade da Secao 3(a) do artigo: o negativo da entropia."""

    return -binary_entropy(probability)


def scenario_summary(
    worlds: Iterable[dict[str, Any]],
    target: dict[str, str],
    evidence: Iterable[dict[str, str]] | None = None,
) -> dict[str, float | int]:
    """Probabilidade e utilidade da consulta no cenario ``S_evidence``."""

    target_attribute = str(target.get("attribute", "")).strip()
    target_value = str(target.get("value", "")).strip()
    if not target_attribute or not target_value:
        raise ValueError("A consulta ground precisa de atributo e valor.")

    evidence_map = _evidence_map(evidence)
    selected, total = _scenario_worlds(worlds, evidence_map)
    positive = sum(
        weight
        for values, weight in selected
        if values.get(target_attribute) == target_value
    )
    probability = positive / total
    entropy = binary_entropy(probability)
    return {
        "probability": probability,
        "entropy": entropy,
        "utility": -entropy,
        "scenarioMass": total,
        "worldCount": len(selected),
    }


def value_of_information(
    worlds: Iterable[dict[str, Any]],
    target: dict[str, str],
    observable: str,
    evidence: Iterable[dict[str, str]] | None = None,
    *,
    cost: float = DEFAULT_OBSERVABLE_COST,
) -> dict[str, Any]:
    """Calcula a Definicao 7 para um observavel no cenario fornecido."""

    observable = str(observable).strip()
    if not observable:
        raise ValueError("O observavel precisa de um nome de atributo.")
    cost = float(cost)
    if not math.isfinite(cost) or cost <= 0:
        raise ValueError("O custo de observacao deve ser positivo e finito.")

    evidence_map = _evidence_map(evidence)
    if observable in evidence_map:
        raise ValueError(f"{observable} ja foi observado neste cenario.")

    selected, scenario_mass = _scenario_worlds(worlds, evidence_map)
    outcomes: dict[str, float] = {}
    for values, weight in selected:
        if observable not in values:
            raise ValueError(f"O observavel {observable} nao existe nos mundos.")
        outcome = str(values[observable])
        outcomes[outcome] = outcomes.get(outcome, 0.0) + weight

    current = scenario_summary(worlds, target, _conditions(evidence_map))
    expected_utility = 0.0
    outcome_payload: list[dict[str, Any]] = []
    for outcome, mass in sorted(outcomes.items()):
        probability_outcome = mass / scenario_mass
        realized_evidence = dict(evidence_map)
        realized_evidence[observable] = outcome
        realized = scenario_summary(worlds, target, _conditions(realized_evidence))
        expected_utility += probability_outcome * float(realized["utility"])
        outcome_payload.append(
            {
                "value": outcome,
                "probability": probability_outcome,
                "queryProbability": realized["probability"],
                "entropy": realized["entropy"],
                "utility": realized["utility"],
                "scenarioMass": realized["scenarioMass"],
            }
        )

    voi = expected_utility - float(current["utility"])
    if abs(voi) <= VOI_TOLERANCE:
        voi = 0.0
    expected_entropy = -expected_utility
    return {
        "observable": observable,
        "cost": cost,
        "currentQueryProbability": current["probability"],
        "currentEntropy": current["entropy"],
        "currentUtility": current["utility"],
        "expectedEntropy": expected_entropy,
        "expectedUtility": expected_utility,
        "voi": max(0.0, voi),
        "voiPerCost": max(0.0, voi) / cost,
        "outcomes": outcome_payload,
    }


def subset_value_of_information(
    worlds: Iterable[dict[str, Any]],
    target: dict[str, str],
    observables: Iterable[str],
    evidence: Iterable[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """VoI de um subconjunto escolhido de uma vez (subset selection)."""

    observable_list = list(dict.fromkeys(str(item).strip() for item in observables))
    if not observable_list or any(not item for item in observable_list):
        raise ValueError("Informe ao menos um observavel valido.")
    evidence_map = _evidence_map(evidence)
    if set(observable_list) & set(evidence_map):
        raise ValueError("Um atributo ja observado nao pode voltar ao subconjunto.")

    selected, scenario_mass = _scenario_worlds(worlds, evidence_map)
    grouped: dict[tuple[str, ...], float] = {}
    for values, weight in selected:
        try:
            realization = tuple(str(values[attribute]) for attribute in observable_list)
        except KeyError as error:
            raise ValueError(f"Observavel inexistente: {error.args[0]}") from error
        grouped[realization] = grouped.get(realization, 0.0) + weight

    current = scenario_summary(worlds, target, _conditions(evidence_map))
    expected_entropy = 0.0
    realization_payload: list[dict[str, Any]] = []
    for realization, mass in sorted(grouped.items()):
        realized_evidence = dict(evidence_map)
        realized_evidence.update(dict(zip(observable_list, realization)))
        realized = scenario_summary(worlds, target, _conditions(realized_evidence))
        probability_realization = mass / scenario_mass
        expected_entropy += probability_realization * float(realized["entropy"])
        realization_payload.append(
            {
                "values": dict(zip(observable_list, realization)),
                "probability": probability_realization,
                "queryProbability": realized["probability"],
                "entropy": realized["entropy"],
            }
        )

    voi = float(current["entropy"]) - expected_entropy
    if abs(voi) <= VOI_TOLERANCE:
        voi = 0.0
    return {
        "observables": observable_list,
        "initialEntropy": current["entropy"],
        "expectedEntropy": expected_entropy,
        "voi": max(0.0, voi),
        "realizations": realization_payload,
    }


def rank_observables(
    worlds: Iterable[dict[str, Any]],
    target: dict[str, str],
    observables: Iterable[str],
    costs: dict[str, float] | None = None,
    evidence: Iterable[dict[str, str]] | None = None,
) -> list[dict[str, Any]]:
    """Avalia todos os candidatos, mantendo a escolha por VoI do artigo."""

    costs = costs or {}
    evidence_map = _evidence_map(evidence)
    candidates = [
        attribute
        for attribute in dict.fromkeys(str(item).strip() for item in observables)
        if attribute and attribute not in evidence_map
    ]
    ranking = [
        value_of_information(
            worlds,
            target,
            attribute,
            _conditions(evidence_map),
            cost=float(costs.get(attribute, DEFAULT_OBSERVABLE_COST)),
        )
        for attribute in candidates
    ]
    # A Figura 3 maximiza VoI entre observacoes que cabem no orcamento. Custo
    # serve como restricao, nao como divisor do criterio de escolha.
    ranking.sort(key=lambda item: (-item["voi"], item["cost"], item["observable"]))
    return ranking


def build_conditional_plan(
    worlds: Iterable[dict[str, Any]],
    target: dict[str, str],
    observables: Iterable[str],
    costs: dict[str, float] | None,
    budget: float,
    evidence: Iterable[dict[str, str]] | None = None,
    *,
    max_nodes: int = DEFAULT_MAX_NODES,
) -> dict[str, Any]:
    """Constroi o plano condicional guloso da Figura 3 em largura (BFS)."""

    worlds = list(worlds)
    initial_evidence = _evidence_map(evidence)
    observable_list = [
        attribute
        for attribute in dict.fromkeys(str(item).strip() for item in observables)
        if attribute and attribute not in initial_evidence
    ]
    if not observable_list:
        raise ValueError("Informe ao menos um observavel ainda nao medido.")

    costs = {attribute: float((costs or {}).get(attribute, DEFAULT_OBSERVABLE_COST)) for attribute in observable_list}
    for attribute, cost in costs.items():
        if not math.isfinite(cost) or cost <= 0:
            raise ValueError(f"Custo invalido para {attribute}.")
    budget = float(budget)
    if not math.isfinite(budget) or budget < 0:
        raise ValueError("O orcamento deve ser finito e nao negativo.")
    max_nodes = int(max_nodes)
    if max_nodes < 1:
        raise ValueError("O limite de nos deve ser ao menos 1.")

    root_summary = scenario_summary(worlds, target, _conditions(initial_evidence))
    next_node_id = 1
    root: dict[str, Any] = {
        "id": 0,
        "depth": 0,
        "scenario": _conditions(initial_evidence),
        "remainingBudget": budget,
        "reachProbability": 1.0,
        "queryProbability": root_summary["probability"],
        "entropy": root_summary["entropy"],
        "utility": root_summary["utility"],
        "choice": None,
        "children": [],
        "stopReason": None,
    }
    worklist: deque[tuple[dict[str, Any], dict[str, str], list[str]]] = deque(
        [(root, initial_evidence, observable_list)]
    )
    expanded_nodes = 0

    while worklist:
        node, scenario, remaining = worklist.popleft()
        if not remaining:
            node["stopReason"] = "no_observables"
            continue

        affordable = [attribute for attribute in remaining if costs[attribute] <= node["remainingBudget"] + VOI_TOLERANCE]
        if not affordable:
            node["stopReason"] = "insufficient_budget"
            continue

        ranking = rank_observables(
            worlds,
            target,
            affordable,
            costs,
            _conditions(scenario),
        )
        node["ranking"] = ranking
        choice = ranking[0] if ranking else None
        if choice is None or choice["voi"] <= VOI_TOLERANCE:
            node["stopReason"] = "no_utility_gain"
            continue

        outcomes = choice["outcomes"]
        if next_node_id + len(outcomes) > max_nodes:
            node["stopReason"] = "node_limit"
            continue

        node["choice"] = {
            "observable": choice["observable"],
            "cost": choice["cost"],
            "voi": choice["voi"],
            "expectedEntropy": choice["expectedEntropy"],
        }
        next_remaining = [attribute for attribute in remaining if attribute != choice["observable"]]
        for outcome in outcomes:
            child_scenario = dict(scenario)
            child_scenario[choice["observable"]] = outcome["value"]
            child = {
                "id": next_node_id,
                "depth": node["depth"] + 1,
                "realization": {
                    "attribute": choice["observable"],
                    "value": outcome["value"],
                    "conditionalProbability": outcome["probability"],
                },
                "scenario": _conditions(child_scenario),
                "remainingBudget": max(0.0, node["remainingBudget"] - choice["cost"]),
                "reachProbability": node["reachProbability"] * outcome["probability"],
                "queryProbability": outcome["queryProbability"],
                "entropy": outcome["entropy"],
                "utility": outcome["utility"],
                "choice": None,
                "children": [],
                "stopReason": None,
            }
            next_node_id += 1
            node["children"].append(child)
            worklist.append((child, child_scenario, next_remaining))
        expanded_nodes += 1

    stack = [root]
    leaves: list[dict[str, Any]] = []
    max_depth = 0
    while stack:
        node = stack.pop()
        max_depth = max(max_depth, int(node["depth"]))
        if node["children"]:
            stack.extend(node["children"])
        else:
            leaves.append(node)

    expected_leaf_entropy = sum(
        float(leaf["reachProbability"]) * float(leaf["entropy"])
        for leaf in leaves
    )
    plan_voi = float(root_summary["entropy"]) - expected_leaf_entropy
    if abs(plan_voi) <= VOI_TOLERANCE:
        plan_voi = 0.0

    return {
        "article": ARTICLE,
        "method": "greedy_conditional_plan",
        "algorithm": "Figura 3 de Ghosh e Ramakrishnan (2019)",
        "utility": "negative_binary_entropy",
        "target": target,
        "initialScenario": _conditions(initial_evidence),
        "observables": [
            {"attribute": attribute, "cost": costs[attribute]}
            for attribute in observable_list
        ],
        "budget": budget,
        "maxNodes": max_nodes,
        "initialQueryProbability": root_summary["probability"],
        "initialEntropy": root_summary["entropy"],
        "expectedFinalEntropy": expected_leaf_entropy,
        "planVoi": max(0.0, plan_voi),
        "tree": root,
        "summary": {
            "nodes": next_node_id,
            "expandedNodes": expanded_nodes,
            "leaves": len(leaves),
            "maxDepth": max_depth,
            "completeWithinNodeLimit": all(leaf.get("stopReason") != "node_limit" for leaf in leaves),
        },
    }
