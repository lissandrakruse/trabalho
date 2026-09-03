"""Selecao ativa de restricoes para consultas probabilisticas intervalares.

A adaptacao usa os dois extremos ja obtidos pelo programa linear. Uma
restricao candidata que e satisfeita por um extremo nao pode alterar aquele
limite quando e adicionada: o extremo continua factivel no conjunto menor.
Assim, somente os extremos violados precisam ser reotimizados.
"""

from __future__ import annotations

import math
import random
import time
from typing import Any


BASE_CONSTRAINT_KINDS = {"marginal", "pairwise_joint"}
CANDIDATE_CONSTRAINT_KINDS = {
    "apriori_rule_support",
    "apriori_rule_confidence",
}


def _record_conditions(record: dict[str, Any]) -> list[dict[str, str]]:
    if "conditions" in record:
        return list(record["conditions"])
    return [*record.get("antecedent", []), *record.get("consequent", [])]


def _literal_set(record: dict[str, Any]) -> set[tuple[str, str]]:
    return {
        (condition["attribute"], condition["value"])
        for condition in _record_conditions(record)
    }


def _event_text(conditions: list[dict[str, str]]) -> str:
    return ", ".join(
        f"{condition['attribute']}={condition['value']}"
        for condition in conditions
    )


def _record_description(record: dict[str, Any]) -> str:
    if record["kind"] == "apriori_rule_confidence":
        return (
            f"{_event_text(record.get('antecedent', []))} -> "
            f"{_event_text(record.get('consequent', []))}"
        )
    return f"P({_event_text(record.get('conditions', []))})"


def _quality_score(record: dict[str, Any]) -> float:
    """Pontuacao da heuristica descritiva de suporte/confianca."""

    if record["kind"] == "apriori_rule_confidence":
        return float(record.get("support", 0.0)) * float(record.get("value", 0.0))
    return float(record.get("value", 0.0))


def select_active_constraints(
    model: dict[str, Any],
    target: dict[str, str],
    base: list[dict[str, str]],
    *,
    budget: int = 5,
    minimum_literal_overlap: int = 2,
    max_candidates: int = 80,
    solver_method: str = "highs-ipm",
    feasibility_tolerance: float = 1e-8,
    random_seed: int = 20260903,
) -> dict[str, Any]:
    """Seleciona restricoes por reducao exata da largura do intervalo.

    Em cada passo, e escolhida a restricao mais violada pelos extremos atuais.
    Somente um extremo violado e reotimizado. Isso evita a busca combinatoria
    de subconjuntos; a escolha gulosa e uma aproximacao e nao uma garantia de
    otimo global.
    """

    started = time.perf_counter()
    if budget < 0:
        raise ValueError("O orcamento de restricoes deve ser nao negativo.")
    if minimum_literal_overlap < 1:
        raise ValueError("A sobreposicao minima deve ser pelo menos 1.")
    if max_candidates < 1:
        raise ValueError("O limite de candidatos deve ser pelo menos 1.")

    try:
        import numpy as np
        from scipy.optimize import linprog
        from scipy.sparse import coo_matrix, vstack
    except Exception as error:
        raise RuntimeError(f"scipy indisponivel para selecao ativa: {error}") from error

    variable_count = int(model["solverVariables"])

    def sparse_matrix(rows: list[dict[int, float]]) -> Any:
        values: list[float] = []
        row_indexes: list[int] = []
        column_indexes: list[int] = []
        for row_index, row in enumerate(rows):
            for column_index, value in row.items():
                row_indexes.append(row_index)
                column_indexes.append(column_index)
                values.append(float(value))
        return coo_matrix(
            (values, (row_indexes, column_indexes)),
            shape=(len(rows), variable_count),
        ).tocsr()

    def dense_objective(sparse_objective: dict[int, float]) -> list[float]:
        return [float(sparse_objective.get(index, 0.0)) for index in range(variable_count)]

    equality_matrix = sparse_matrix(model["aEq"])
    objective_lower = dense_objective(model["objectiveLower"])
    objective_upper = dense_objective(model["objectiveUpperAsMin"])

    def optimize(inequality_matrix: Any, *, upper: bool) -> tuple[float, Any]:
        result = linprog(
            c=objective_upper if upper else objective_lower,
            A_ub=inequality_matrix,
            b_ub=np.zeros(inequality_matrix.shape[0]),
            A_eq=equality_matrix,
            b_eq=model["bEq"],
            bounds=model["bounds"],
            method=solver_method,
        )
        if not result.success:
            raise RuntimeError(f"HiGHS falhou na selecao ativa: {result.message}")
        value = -float(result.fun) if upper else float(result.fun)
        value = min(1.0, max(0.0, value))
        return value, result.x

    records = list(model["records"])
    base_row_indexes: list[int] = []
    raw_candidates: list[dict[str, Any]] = []
    query_literals = {
        (condition["attribute"], condition["value"])
        for condition in [target, *base]
    }
    for record_index, record in enumerate(records):
        if record["kind"] in BASE_CONSTRAINT_KINDS:
            base_row_indexes.extend(record.get("rowIndexes", []))
            continue
        if record["kind"] not in CANDIDATE_CONSTRAINT_KINDS:
            continue
        overlap = len(_literal_set(record) & query_literals)
        raw_candidates.append(
            {
                "id": f"R{record_index + 1:04d}",
                "record": record,
                "literalOverlap": overlap,
                "qualityScore": _quality_score(record),
            }
        )

    requested_overlap = minimum_literal_overlap
    relevant = [
        candidate
        for candidate in raw_candidates
        if candidate["literalOverlap"] >= minimum_literal_overlap
    ]
    fallback_used = False
    if not relevant and minimum_literal_overlap > 1:
        minimum_literal_overlap = 1
        fallback_used = True
        relevant = [
            candidate
            for candidate in raw_candidates
            if candidate["literalOverlap"] >= minimum_literal_overlap
        ]

    relevant.sort(
        key=lambda candidate: (
            -candidate["literalOverlap"],
            -candidate["qualityScore"],
            candidate["id"],
        )
    )
    relevant_before_cap = len(relevant)
    candidates = relevant[:max_candidates]

    base_rows = [model["aUb"][index] for index in base_row_indexes]
    base_matrix = sparse_matrix(base_rows)
    lower, lower_solution = optimize(base_matrix, upper=False)
    upper, upper_solution = optimize(base_matrix, upper=True)
    base_width = max(0.0, upper - lower)

    candidate_by_id = {candidate["id"]: candidate for candidate in candidates}
    candidate_matrices = {
        candidate["id"]: sparse_matrix(
            [
                model["aUb"][index]
                for index in candidate["record"].get("rowIndexes", [])
            ]
        )
        for candidate in candidates
    }

    def candidate_state(
        candidate: dict[str, Any],
        current_lower_solution: Any,
        current_upper_solution: Any,
    ) -> dict[str, Any]:
        record = candidate["record"]
        matrix = candidate_matrices[candidate["id"]]
        lower_violations = matrix @ current_lower_solution
        upper_violations = matrix @ current_upper_solution
        max_lower_violation = (
            float(lower_violations.max()) if len(lower_violations) else 0.0
        )
        max_upper_violation = (
            float(upper_violations.max()) if len(upper_violations) else 0.0
        )
        return {
            "id": candidate["id"],
            "kind": record["kind"],
            "description": _record_description(record),
            "literalOverlap": candidate["literalOverlap"],
            "qualityScore": candidate["qualityScore"],
            "support": record.get("support", record.get("value")),
            "confidence": (
                record.get("value")
                if record["kind"] == "apriori_rule_confidence"
                else None
            ),
            "lift": record.get("lift"),
            "rowIndexes": list(record.get("rowIndexes", [])),
            "violatesLowerExtreme": max_lower_violation > feasibility_tolerance,
            "violatesUpperExtreme": max_upper_violation > feasibility_tolerance,
            "maxLowerViolation": max(0.0, max_lower_violation),
            "maxUpperViolation": max(0.0, max_upper_violation),
            "violationScore": max(0.0, max_lower_violation, max_upper_violation),
        }

    initial_ranking = [
        candidate_state(candidate, lower_solution, upper_solution)
        for candidate in candidates
    ]
    initial_ranking.sort(
        key=lambda candidate: (
            -candidate["violationScore"],
            -candidate["literalOverlap"],
            -candidate["qualityScore"],
            candidate["id"],
        )
    )

    current_matrix = base_matrix
    current_lower = lower
    current_upper = upper
    current_lower_solution = lower_solution
    current_upper_solution = upper_solution
    remaining = list(candidates)
    selected: list[dict[str, Any]] = []
    selected_endpoint_solves = 0
    algebraic_endpoint_checks = 0
    lower_extremes_pruned = 0
    upper_extremes_pruned = 0
    stop_reason = "budget_reached"

    for step in range(1, min(budget, len(candidates)) + 1):
        states = [
            candidate_state(
                candidate,
                current_lower_solution,
                current_upper_solution,
            )
            for candidate in remaining
        ]
        algebraic_endpoint_checks += 2 * len(states)
        lower_extremes_pruned += sum(
            not state["violatesLowerExtreme"] for state in states
        )
        upper_extremes_pruned += sum(
            not state["violatesUpperExtreme"] for state in states
        )
        states.sort(
            key=lambda candidate: (
                -candidate["violationScore"],
                -(
                    candidate["maxLowerViolation"]
                    + candidate["maxUpperViolation"]
                ),
                -candidate["literalOverlap"],
                -candidate["qualityScore"],
                candidate["id"],
            )
        )
        chosen = states[0]
        if chosen["violationScore"] <= feasibility_tolerance:
            stop_reason = "all_remaining_constraints_satisfied_by_extremes"
            break

        before_width = max(0.0, current_upper - current_lower)
        current_matrix = vstack(
            [current_matrix, candidate_matrices[chosen["id"]]],
            format="csr",
        )
        required_solves = 0
        if chosen["violatesLowerExtreme"]:
            current_lower, current_lower_solution = optimize(
                current_matrix,
                upper=False,
            )
            required_solves += 1
        if chosen["violatesUpperExtreme"]:
            current_upper, current_upper_solution = optimize(
                current_matrix,
                upper=True,
            )
            required_solves += 1
        selected_endpoint_solves += required_solves
        current_width = max(0.0, current_upper - current_lower)
        selected.append(
            {
                **chosen,
                "step": step,
                "requiredLpSolves": required_solves,
                "lowerAfterSelection": current_lower,
                "upperAfterSelection": current_upper,
                "widthAfterSelection": current_width,
                "stepWidthReduction": max(0.0, before_width - current_width),
                "cumulativeWidthReduction": max(0.0, base_width - current_width),
            }
        )
        remaining = [
            candidate for candidate in remaining if candidate["id"] != chosen["id"]
        ]

    selected_count = len(selected)
    active_interval = {
        "lower": current_lower,
        "upper": current_upper,
        "width": max(0.0, current_upper - current_lower),
    }

    def interval_for(candidate_ids: list[str]) -> dict[str, float]:
        selected_rows = list(base_rows)
        for candidate_id in candidate_ids:
            record = candidate_by_id[candidate_id]["record"]
            selected_rows.extend(
                model["aUb"][index]
                for index in record.get("rowIndexes", [])
            )
        matrix = sparse_matrix(selected_rows)
        candidate_lower, _ = optimize(matrix, upper=False)
        candidate_upper, _ = optimize(matrix, upper=True)
        return {
            "lower": candidate_lower,
            "upper": candidate_upper,
            "width": max(0.0, candidate_upper - candidate_lower),
        }

    pool_interval = interval_for([candidate["id"] for candidate in candidates])

    heuristic_order = sorted(
        initial_ranking,
        key=lambda candidate: (
            -candidate["qualityScore"],
            -candidate["literalOverlap"],
            candidate["id"],
        ),
    )
    heuristic_selected = heuristic_order[:selected_count]
    heuristic_interval = interval_for(
        [candidate["id"] for candidate in heuristic_selected]
    )

    def public_selection(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "id": item["id"],
                "kind": item["kind"],
                "description": item["description"],
                "qualityScore": item["qualityScore"],
            }
            for item in items
        ]

    random_trials = []
    for trial in range(5):
        trial_seed = random_seed + trial
        random_generator = random.Random(trial_seed)
        random_selected = (
            random_generator.sample(initial_ranking, selected_count)
            if selected_count
            else []
        )
        random_interval = interval_for(
            [candidate["id"] for candidate in random_selected]
        )
        random_trials.append(
            {
                **random_interval,
                "seed": trial_seed,
                "selected": public_selection(random_selected),
            }
        )

    random_mean_width = sum(trial["width"] for trial in random_trials) / len(random_trials)
    exact_pruning_saved = lower_extremes_pruned + upper_extremes_pruned
    unselected_violated_solves_avoided = max(
        0,
        algebraic_endpoint_checks
        - exact_pruning_saved
        - selected_endpoint_solves,
    )
    total_candidate_solves_avoided = algebraic_endpoint_checks - selected_endpoint_solves
    subset_count = (
        math.comb(len(candidates), selected_count)
        if selected_count <= len(candidates)
        else 0
    )

    return {
        "method": "greedy_query_directed_endpoint_pruning",
        "target": target,
        "conditions": base,
        "budget": budget,
        "selectedCount": selected_count,
        "candidatePool": {
            "availableAprioriConstraints": len(raw_candidates),
            "queryRelevantBeforeCap": relevant_before_cap,
            "evaluated": len(candidates),
            "requestedMinimumLiteralOverlap": requested_overlap,
            "effectiveMinimumLiteralOverlap": minimum_literal_overlap,
            "fallbackUsed": fallback_used,
            "maxCandidates": max_candidates,
            "truncated": relevant_before_cap > len(candidates),
        },
        "baseModel": {
            "constraintRecords": sum(
                record["kind"] in BASE_CONSTRAINT_KINDS for record in records
            ),
            "inequalityRows": len(base_row_indexes),
            "lower": lower,
            "upper": upper,
            "width": base_width,
        },
        "activeSelection": {
            **active_interval,
            "widthReduction": max(0.0, base_width - active_interval["width"]),
            "relativeWidthReduction": (
                max(0.0, base_width - active_interval["width"]) / base_width
                if base_width > 0
                else 0.0
            ),
            "selected": public_selection(selected),
            "selectionTrace": selected,
            "stopReason": stop_reason,
        },
        "baselines": {
            "allCandidatePool": {
                **pool_interval,
                "selectedCount": len(candidates),
            },
            "supportConfidence": {
                **heuristic_interval,
                "selectedCount": selected_count,
                "selected": public_selection(heuristic_selected),
            },
            "random": {
                "meanWidth": random_mean_width,
                "meanWidthReduction": max(0.0, base_width - random_mean_width),
                "selectedCount": selected_count,
                "trials": random_trials,
            },
        },
        "ranking": initial_ranking,
        "solverEffort": {
            "baselineLpSolves": 2,
            "selectedEndpointLpSolves": selected_endpoint_solves,
            "algebraicEndpointChecks": algebraic_endpoint_checks,
            "naiveGreedyCandidateLpSolves": algebraic_endpoint_checks,
            "exactPruningSavedLpSolves": exact_pruning_saved,
            "exactPruningRate": (
                exact_pruning_saved / algebraic_endpoint_checks
                if algebraic_endpoint_checks
                else 0.0
            ),
            "unselectedViolatedEndpointSolvesAvoided": unselected_violated_solves_avoided,
            "totalCandidateLpSolvesAvoided": total_candidate_solves_avoided,
            "totalAvoidanceRate": (
                total_candidate_solves_avoided / algebraic_endpoint_checks
                if algebraic_endpoint_checks
                else 0.0
            ),
            "lowerExtremesPruned": lower_extremes_pruned,
            "upperExtremesPruned": upper_extremes_pruned,
            "candidateEvaluations": len(candidates),
            "fullSubsetSearchCount": str(subset_count),
            "selectionStrategy": (
                "em cada passo, escolher a restricao mais violada pelos extremos; "
                "reotimizar somente os extremos que ela viola"
            ),
        },
        "solver": f"scipy.optimize.linprog {solver_method}",
        "durationSeconds": time.perf_counter() - started,
        "limitations": (
            "A poda de cada extremo e exata. A escolha gulosa pela maior violacao e uma "
            "heuristica e nao garante o subconjunto globalmente otimo. O filtro de "
            "relevancia tambem reduz o universo de busca."
        ),
    }
