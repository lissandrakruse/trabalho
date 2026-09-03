from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any


# SOLVER SEPARADO DO EXERCICIO
#
# Este script replica a formulacao do backend principal sem depender da
# interface Flask. Ele serve para demonstrar que a consulta P(A | B) e o mesmo
# programa linear podem ser resolvidos por um modulo independente, comparando
# valores e tempo entre metodos do HiGHS.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apriori_rules import mine_apriori_rules

DEFAULT_DATASET = ROOT / "data" / "Crop_recommendation.csv"
APRIORI_MIN_SUPPORT = 0.01
APRIORI_MIN_CONFIDENCE = 0.0
APRIORI_MAX_ITEMSET_SIZE = 3
APRIORI_LP_MIN_CONFIDENCE = 0.70
SOLVER_ENGINES = [
    {"id": "highs", "name": "SciPy HiGHS", "method": "highs"},
    {"id": "highs-ds", "name": "HiGHS Dual Simplex", "method": "highs-ds"},
    {"id": "highs-ipm", "name": "HiGHS Interior Point", "method": "highs-ipm"},
]


def parse_condition(text: str) -> dict[str, str]:
    if "=" not in text:
        raise argparse.ArgumentTypeError("Use o formato atributo=valor. Exemplo: ph=acido")
    attribute, value = text.split("=", 1)
    attribute = attribute.strip()
    value = value.strip()
    if not attribute or not value:
        raise argparse.ArgumentTypeError("Atributo e valor nao podem ficar vazios.")
    return {"attribute": attribute, "value": value}


def is_number(value: str) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    base = math.floor(position)
    rest = position - base
    if base + 1 >= len(ordered):
        return ordered[base]
    return ordered[base] + rest * (ordered[base + 1] - ordered[base])


def category_for(attribute: str, value: str, thresholds: dict[str, dict[str, float]]) -> str:
    number = float(value)
    if attribute.lower() == "ph":
        if number < 6:
            return "acido"
        if number <= 7.5:
            return "neutro"
        return "alcalino"
    if number <= thresholds[attribute]["low"]:
        return "baixo"
    if number <= thresholds[attribute]["high"]:
        return "medio"
    return "alto"


def load_dataset(path: Path) -> dict[str, Any]:
    # Mesma preparacao do app.py: carrega a base, categoriza valores numericos
    # e cria mundos possiveis observados para receber variaveis x_w no PL.
    with path.open("r", encoding="utf-8", newline="") as file:
        raw_rows = list(csv.DictReader(file))
    if not raw_rows:
        raise RuntimeError("Dataset vazio.")

    attributes = list(raw_rows[0].keys())
    numeric_attributes = [
        attribute
        for attribute in attributes
        if all(is_number(row.get(attribute, "")) for row in raw_rows)
    ]
    categorical_attributes = [attribute for attribute in attributes if attribute not in numeric_attributes]

    thresholds = {}
    for attribute in numeric_attributes:
        values = [float(row[attribute]) for row in raw_rows]
        thresholds[attribute] = {
            "low": quantile(values, 1 / 3),
            "high": quantile(values, 2 / 3),
        }

    rows = []
    for row in raw_rows:
        categorized = {
            attribute: category_for(attribute, row[attribute], thresholds)
            for attribute in numeric_attributes
        }
        for attribute in categorical_attributes:
            categorized[attribute] = row[attribute]
        rows.append(categorized)

    domains = {attribute: sorted({row[attribute] for row in rows}) for attribute in attributes}
    world_counts: dict[tuple[str, ...], int] = {}
    for row in rows:
        key = tuple(row[attribute] for attribute in attributes)
        world_counts[key] = world_counts.get(key, 0) + 1

    worlds = [
        {"values": dict(zip(attributes, key)), "count": count}
        for key, count in world_counts.items()
    ]
    apriori = mine_apriori_rules(
        worlds,
        len(rows),
        min_support=APRIORI_MIN_SUPPORT,
        min_confidence=APRIORI_MIN_CONFIDENCE,
        max_itemset_size=APRIORI_MAX_ITEMSET_SIZE,
    )
    return {
        "datasetPath": str(path.resolve()),
        "attributes": attributes,
        "numericAttributes": numeric_attributes,
        "categoricalAttributes": categorical_attributes,
        "rows": rows,
        "worlds": worlds,
        "domains": domains,
        "thresholds": thresholds,
        "total": len(rows),
        "apriori": apriori,
    }


def validate_conditions(
    conditions: list[dict[str, str]],
    domains: dict[str, list[str]],
) -> list[dict[str, str]]:
    cleaned = []
    for condition in conditions:
        attribute = condition["attribute"]
        value = condition["value"]
        if attribute not in domains:
            raise ValueError(f"Atributo invalido: {attribute}")
        if value not in domains[attribute]:
            valid_values = ", ".join(domains[attribute])
            raise ValueError(f"Valor invalido para {attribute}: {value}. Use: {valid_values}")
        cleaned.append({"attribute": attribute, "value": value})
    return cleaned


def normalize_base_conditions(
    base: list[dict[str, str]],
    target: dict[str, str],
) -> list[dict[str, str]]:
    normalized = []
    seen: set[tuple[str, str]] = set()
    target_key = (target["attribute"], target["value"])
    for condition in base:
        key = (condition["attribute"], condition["value"])
        if key == target_key or key in seen:
            continue
        seen.add(key)
        normalized.append(condition)
    return normalized


def matches(row: dict[str, str], conditions: list[dict[str, str]]) -> bool:
    return all(row[condition["attribute"]] == condition["value"] for condition in conditions)


def probability(rows: list[dict[str, str]], conditions: list[dict[str, str]]) -> float:
    if not conditions:
        return 1.0
    return sum(1 for row in rows if matches(row, conditions)) / len(rows)


def probability_count(rows: list[dict[str, str]], conditions: list[dict[str, str]]) -> int:
    if not conditions:
        return len(rows)
    return sum(1 for row in rows if matches(row, conditions))


def world_mask(worlds: list[dict[str, Any]], conditions: list[dict[str, str]]) -> list[float]:
    return [1.0 if matches(world["values"], conditions) else 0.0 for world in worlds]


def sparse_world_mask(
    worlds: list[dict[str, Any]],
    conditions: list[dict[str, str]],
) -> dict[int, float]:
    return {
        index: 1.0
        for index, world in enumerate(worlds)
        if matches(world["values"], conditions)
    }


def add_sparse_vectors(
    left: dict[int, float],
    right: dict[int, float],
    right_scale: float = 1.0,
) -> dict[int, float]:
    result = dict(left)
    for index, value in right.items():
        combined = result.get(index, 0.0) + (right_scale * value)
        if abs(combined) <= 1e-15:
            result.pop(index, None)
        else:
            result[index] = combined
    return result


def complete_unobserved_query_worlds(
    data: dict[str, Any],
    target: dict[str, str],
    base: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], int]:
    worlds = data["worlds"]
    both = [*base, target]
    if any(matches(world["values"], both) for world in worlds):
        return worlds, 0

    fixed: dict[str, str] = {}
    for condition in both:
        previous = fixed.get(condition["attribute"])
        if previous is not None and previous != condition["value"]:
            return worlds, 0
        fixed[condition["attribute"]] = condition["value"]
    free_attributes = [attribute for attribute in data["attributes"] if attribute not in fixed]
    observed_keys = {
        tuple(world["values"][attribute] for attribute in data["attributes"])
        for world in worlds
    }
    additions = []
    for values in itertools.product(*(data["domains"][attribute] for attribute in free_attributes)):
        completed = dict(fixed)
        completed.update(dict(zip(free_attributes, values)))
        key = tuple(completed[attribute] for attribute in data["attributes"])
        if key not in observed_keys:
            additions.append(
                {"values": completed, "count": 0, "queryCompletion": True}
            )
    return [*worlds, *additions], len(additions)


def probability_interval(value: float, width: float = 0.001) -> tuple[float, float]:
    value = float(value)
    return max(0.0, value - width), min(1.0, value + width)


def clean_probability(value: float | None) -> float | None:
    if value is None:
        return None
    if abs(value) < 1e-9:
        return 0.0
    return min(1.0, max(0.0, value))


def fmt_probability(value: float | None) -> str:
    cleaned = clean_probability(value)
    if cleaned is None:
        return "-"
    return f"{cleaned:.3f}"


def event_key(conditions: list[dict[str, str]]) -> str:
    if not conditions:
        return "verdadeiro"
    return ", ".join(f"{item['attribute']}={item['value']}" for item in conditions)


def condition_signature(conditions: list[dict[str, str]]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((item["attribute"], item["value"]) for item in conditions))


def mine_association_rules(
    data: dict[str, Any],
) -> list[dict[str, Any]]:
    return list(data["apriori"]["rules"])


def find_released_association_rule(
    rules: list[dict[str, Any]],
    antecedent: list[dict[str, str]],
    consequent: list[dict[str, str]],
) -> dict[str, Any] | None:
    antecedent_signature = condition_signature(antecedent)
    consequent_signature = condition_signature(consequent)
    for rule in rules:
        if (
            condition_signature(rule["antecedent"]) == antecedent_signature
            and condition_signature(rule["consequent"]) == consequent_signature
        ):
            return rule
    return None


def query_association_rule(
    rules: list[dict[str, Any]],
    antecedent: list[dict[str, str]],
    consequent: list[dict[str, str]],
) -> dict[str, Any] | None:
    return find_released_association_rule(rules, antecedent, consequent)


def build_linear_constraints(
    data: dict[str, Any],
    target: dict[str, str],
    base: list[dict[str, str]],
) -> tuple[list[dict[int, float]], list[float], dict[str, int]]:
    # Monta as mesmas restricoes do projeto principal: marginais, conjuntas por
    # pares, suportes Apriori e confiancas das regras fortes.
    rows = data["rows"]
    worlds = data["worlds"]
    a_ub: list[dict[int, float]] = []
    b_ub: list[float] = []
    summary = {
        "marginal": 0,
        "pairwiseJoint": 0,
        "aprioriRuleSupport": 0,
        "aprioriRuleConfidence": 0,
        "aprioriRuleConfidenceThreshold": APRIORI_LP_MIN_CONFIDENCE,
        "aprioriRuleConfidenceFilteredOut": 0,
        "aprioriRules": 0,
    }
    interval_events: set[tuple[tuple[str, str], ...]] = set()
    mask_cache: dict[tuple[tuple[str, str], ...], dict[int, float]] = {}
    probability_cache: dict[tuple[tuple[str, str], ...], float] = {}
    total_weight = sum(int(world.get("count", 1)) for world in worlds)

    def cached_sparse_mask(conditions: list[dict[str, str]]) -> dict[int, float]:
        signature = condition_signature(conditions)
        mask = mask_cache.get(signature)
        if mask is None:
            mask = sparse_world_mask(worlds, conditions)
            mask_cache[signature] = mask
        return mask

    def weighted_probability(conditions: list[dict[str, str]]) -> float:
        signature = condition_signature(conditions)
        cached = probability_cache.get(signature)
        if cached is not None:
            return cached
        mask = cached_sparse_mask(conditions)
        value = (
            sum(int(worlds[index].get("count", 1)) for index in mask)
            / total_weight
        )
        probability_cache[signature] = value
        return value

    def add_interval(kind: str, conditions: list[dict[str, str]], value: float) -> bool:
        signature = condition_signature(conditions)
        if signature in interval_events:
            return False
        interval_events.add(signature)
        lower, upper = probability_interval(value)
        mask = cached_sparse_mask(conditions)
        a_ub.append(mask)
        b_ub.append(upper)
        a_ub.append({index: -value for index, value in mask.items()})
        b_ub.append(-lower)
        summary[kind] += 1
        return True

    def add_apriori_rule(rule: dict[str, Any]) -> None:
        # Suporte ancora P(R e S). Confianca vira duas desigualdades lineares.
        # Lift e apenas descritivo e nao participa da formulacao.
        antecedent = rule["antecedent"]
        consequent = rule["consequent"]
        both = [*antecedent, *consequent]
        add_interval("aprioriRuleSupport", both, rule["support"])
        if rule["confidence"] < APRIORI_LP_MIN_CONFIDENCE:
            summary["aprioriRuleConfidenceFilteredOut"] += 1
            return
        lower, upper = probability_interval(rule["confidence"])
        antecedent_mask = cached_sparse_mask(antecedent)
        both_mask = cached_sparse_mask(both)
        a_ub.append(add_sparse_vectors(both_mask, antecedent_mask, -upper))
        b_ub.append(0.0)
        negative_both = {index: -value for index, value in both_mask.items()}
        a_ub.append(add_sparse_vectors(negative_both, antecedent_mask, lower))
        b_ub.append(0.0)
        summary["aprioriRuleConfidence"] += 1

    for attribute in data["attributes"]:
        for value in data["domains"][attribute]:
            conditions = [{"attribute": attribute, "value": value}]
            add_interval("marginal", conditions, weighted_probability(conditions))

    for index, left_attribute in enumerate(data["attributes"]):
        for right_attribute in data["attributes"][index + 1 :]:
            for left_value in data["domains"][left_attribute]:
                for right_value in data["domains"][right_attribute]:
                    conditions = [
                        {"attribute": left_attribute, "value": left_value},
                        {"attribute": right_attribute, "value": right_value},
                    ]
                    add_interval("pairwiseJoint", conditions, weighted_probability(conditions))

    learned_rules = mine_association_rules(data)
    for rule in learned_rules:
        add_apriori_rule(rule)
    summary["aprioriRules"] = len(learned_rules)

    return a_ub, b_ub, summary


@lru_cache(maxsize=1)
def cached_linear_constraint_model(
) -> tuple[list[dict[int, float]], list[float], dict[str, int]]:
    data = load_dataset(DEFAULT_DATASET)
    return build_linear_constraints(data, {}, [])


def solve_linear_interval(
    data: dict[str, Any],
    target: dict[str, str],
    base: list[dict[str, str]],
    solver_method: str = "highs-ipm",
    solver_name: str = "SciPy HiGHS Interior Point",
) -> dict[str, Any]:
    # Resolve dois LPs: um minimiza e outro maximiza P(A | B). A razao e
    # linearizada por Charnes-Cooper antes da chamada ao scipy.optimize.linprog.
    rows = data["rows"]
    observed_worlds = data["worlds"]
    denominator_probability = probability(rows, base)
    denominator_count = probability_count(rows, base)
    if denominator_count == 0 or denominator_probability <= 0:
        return {
            "ok": False,
            "reason": "zero_denominator",
            "error": (
                "A consulta condicional nao pode ser resolvida porque P(B)=0 na base. "
                "Nenhum registro satisfaz todas as afirmacoes escolhidas."
            ),
            "baseProbability": denominator_probability,
            "baseCount": denominator_count,
        }

    try:
        from scipy.optimize import linprog
        from scipy.sparse import coo_matrix, csr_matrix, hstack
    except Exception as error:
        return {"ok": False, "error": f"scipy indisponivel: {error}"}

    worlds, completion_count = complete_unobserved_query_worlds(data, target, base)
    n = len(worlds)
    if data.get("datasetPath") == str(DEFAULT_DATASET.resolve()) and completion_count == 0:
        a_ub, b_ub, summary = cached_linear_constraint_model()
    else:
        model_data = {**data, "worlds": worlds}
        a_ub, b_ub, summary = build_linear_constraints(model_data, target, base)

    both = [*base, target]

    denominator_mask = world_mask(worlds, base)
    numerator_mask = world_mask(worlds, both)

    # Charnes-Cooper transforma P(A e B) / P(B) em objetivos lineares.
    sparse_values: list[float] = []
    sparse_rows: list[int] = []
    sparse_columns: list[int] = []
    for row_index, row in enumerate(a_ub):
        for column_index, value in row.items():
            sparse_rows.append(row_index)
            sparse_columns.append(column_index)
            sparse_values.append(value)
    base_matrix = coo_matrix(
        (sparse_values, (sparse_rows, sparse_columns)),
        shape=(len(a_ub), n),
    ).tocsr()
    transformed_a_ub = hstack(
        [base_matrix, csr_matrix([[-limit] for limit in b_ub])],
        format="csr",
    )
    transformed_b_ub = [0.0] * len(a_ub)
    transformed_a_eq = csr_matrix(
        [
            [*([1.0] * n), -1.0],
            [*denominator_mask, 0.0],
        ]
    )
    transformed_b_eq = [0.0, 1.0]
    transformed_bounds = [(0.0, None)] * (n + 1)

    def optimize(objective: list[float]) -> Any:
        return linprog(
            c=objective,
            A_ub=transformed_a_ub,
            b_ub=transformed_b_ub,
            A_eq=transformed_a_eq,
            b_eq=transformed_b_eq,
            bounds=transformed_bounds,
            method=solver_method,
        )

    lower_result = optimize([*numerator_mask, 0.0])
    upper_result = optimize([-item for item in [*numerator_mask, 0.0]])
    if not lower_result.success or not upper_result.success:
        return {
            "ok": False,
            "error": lower_result.message if not lower_result.success else upper_result.message,
        }

    return {
        "ok": True,
        "lower": clean_probability(float(lower_result.fun)),
        "upper": clean_probability(float(-upper_result.fun)),
        "variables": n + 1,
        "worldVariables": n,
        "observedWorldVariables": len(observed_worlds),
        "queryCompletionWorlds": completion_count,
        "empiricalJointZero": probability_count(rows, [*base, target]) == 0,
        "solverVariables": n + 1,
        "constraints": transformed_a_ub.shape[0] + transformed_a_eq.shape[0],
        "baseConstraints": len(a_ub),
        "constraintSummary": summary,
        "solver": f"scipy.optimize.linprog {solver_method}",
        "solverMethod": solver_method,
        "solverName": solver_name,
    }


def linear_program_text(
    target: dict[str, str],
    base: list[dict[str, str]],
    p_a: float,
    p_b: float,
    p_ab: float,
    lp: dict[str, Any],
) -> str:
    numerator = f"soma(x_w onde {event_key([*base, target])})"
    denominator = f"soma(x_w onde {event_key(base)})"
    lines = [
        "MODELO MATEMATICO DA CONSULTA",
        "",
        "1. Eventos da consulta:",
        f"  A = {event_key([target])}",
        f"  B = {event_key(base)}",
        f"  A e B = {event_key([*base, target])}",
        "",
        "2. Variaveis:",
        "  Cada mundo possivel w e uma combinacao categorica completa.",
        "  Mundos observados preservam a contagem; completamentos da consulta tem contagem zero.",
        "  x_w >= 0 representa a massa de probabilidade atribuida ao mundo w.",
        "",
        "3. Normalizacao probabilistica:",
        "  soma(x_w) = 1",
        "",
        "4. Evidencias globais usadas como restricoes intervalares:",
        "  O LP inclui marginais de cada valor e conjuntas por pares de valores.",
        "  Cada faixa usa o valor empirico completo p: max(0, p - 0.001) <= P(E) <= min(1, p + 0.001).",
        "  Nenhuma chamada round() altera p ou os coeficientes enviados ao solver.",
        f"  P(A) empirico completo para auditoria = {p_a!r}",
        f"  P(B) empirico completo para auditoria = {p_b!r}",
        f"  P(A e B) empirico completo para auditoria = {p_ab!r}",
        "  A consulta nao injeta P(A e B) como resposta pronta no LP.",
        "",
        "5. Mineracao Apriori e regras lineares:",
        "  O Apriori recebe os mundos observados de Omega como transacoes ponderadas.",
        "  Suporte ancora P(R e S) para todas as regras geradas.",
        "  Confianca ancora P(R e S) = confianca.P(R) quando confianca >= 0.700.",
        "  Regras abaixo desse limiar continuam visiveis; lift permanece descritivo.",
        "",
        "6. Papel do lift:",
        "  Lift e apenas descritivo: nao mede acuracia e nao entra no LP.",
        "",
        "7. Consulta condicional:",
        "  P(A | B) = P(A e B) / P(B)",
        "",
        "8. Resolucao linear:",
        "  A razao P(A e B) / P(B) e convertida por Charnes-Cooper:",
        "  x_w = y_w / t",
        "  P(B) em y e fixado como 1",
        "  minimizar soma(y_w onde A e B) para o limite inferior",
        "  maximizar soma(y_w onde A e B) para o limite superior",
    ]
    if lp.get("ok"):
        lines.extend(
            [
                "",
                f"Solver: {lp['solver']}",
                f"Variaveis: {lp['variables']}",
                f"Restricoes: {lp['constraints']}",
                f"Intervalo retornado: {fmt_probability(lp['lower'])} <= P(A | B) <= {fmt_probability(lp['upper'])}",
            ]
        )
    elif lp.get("reason") == "zero_denominator":
        lines.extend(
            [
                "",
                "Consulta nao resolvida pelo solver:",
                "  P(B)=0 na base, entao P(A | B) teria denominador zero.",
                "  Escolha menos afirmacoes ou uma combinacao B que exista no dataset.",
            ]
        )
    else:
        lines.extend(["", f"Solver indisponivel: {lp.get('error', 'erro desconhecido')}"])
    return "\n".join(lines)


def conclusion_text(
    target: dict[str, str],
    base: list[dict[str, str]],
    support: float,
    confidence: float | None,
    lift: float | None,
    p_b: float,
    count_base: int,
    count_both: int,
    lp: dict[str, Any],
) -> str:
    target_label = event_key([target])
    base_label = event_key(base)
    if count_base == 0 or p_b <= 0:
        return (
            f"Nao foi possivel estimar P({target_label} | {base_label}) porque nenhuma "
            "linha do dataset satisfaz todas as afirmacoes escolhidas. A conclusao tecnica "
            "e que essa combinacao nao tem evidencia empirica na base."
        )

    interval_sentence = ""
    if lp.get("ok"):
        interval_sentence = (
            f" Pelo modelo linear, P(A | B) fica entre {fmt_probability(lp['lower'])} "
            f"e {fmt_probability(lp['upper'])}."
        )

    confidence_sentence = (
        "suporte e confianca alimentam restricoes probabilisticas"
        if confidence >= APRIORI_LP_MIN_CONFIDENCE
        else (
            "o suporte alimenta as restricoes; a confianca fica descritiva por ser "
            f"menor que {APRIORI_LP_MIN_CONFIDENCE:.2f}"
        )
    )
    return (
        f"O Apriori gerou a regra {base_label} -> {target_label} com suporte "
        f"{support:.3f} e confianca {confidence:.3f}; {confidence_sentence}. "
        f"O lift {lift:.3f} e apenas descritivo e nao mede acuracia. "
        f"Na base, existem {count_both} ocorrencias conjuntas em {count_base} casos de B."
        f"{interval_sentence}"
    )


def compute_query(
    data: dict[str, Any],
    target: dict[str, str],
    conditions: list[dict[str, str]],
    solver_method: str = "highs",
    solver_name: str = "SciPy HiGHS",
) -> dict[str, Any]:
    # Ponto de entrada reutilizado pela CLI e pela comparacao do Flask.
    # Calcula probabilidades, regra da consulta, intervalo linear e tempo.
    started_at = datetime.now(timezone.utc)
    started_perf = time.perf_counter()
    rows = data["rows"]
    conditions = normalize_base_conditions(conditions, target)
    both = [*conditions, target]
    p_a = probability(rows, [target])
    p_b = probability(rows, conditions)
    p_ab = probability(rows, both)
    count_both = probability_count(rows, both)
    count_base = probability_count(rows, conditions)
    learned_rules = mine_association_rules(data)
    released_rule = query_association_rule(learned_rules, conditions, [target])
    rule_support = released_rule["support"] if released_rule else None
    rule_confidence = released_rule["confidence"] if released_rule else None
    rule_lift = released_rule["lift"] if released_rule else None
    lp = solve_linear_interval(data, target, conditions, solver_method, solver_name)
    finished_at = datetime.now(timezone.utc)
    duration_seconds = time.perf_counter() - started_perf
    return {
        "ok": True,
        "processing": {
            "startedAt": started_at.isoformat(),
            "finishedAt": finished_at.isoformat(),
            "durationSeconds": duration_seconds,
            "durationMilliseconds": round(duration_seconds * 1000, 3),
        },
        "target": target,
        "conditions": conditions,
        "support": rule_support,
        "confidence": rule_confidence,
        "lift": rule_lift,
        "pAB": p_ab,
        "pA": p_a,
        "pB": p_b,
        "countBoth": count_both,
        "countBase": count_base,
        "total": data["total"],
        "aprioriMining": {
            key: value
            for key, value in data["apriori"].items()
            if key != "rules"
        },
        "linear": lp,
        "linearProgram": linear_program_text(target, conditions, p_a, p_b, p_ab, lp),
        "conclusion": (
            conclusion_text(target, conditions, rule_support, rule_confidence, rule_lift, p_b, count_base, count_both, lp)
            if released_rule
            else (
                f"A consulta {event_key(conditions)} -> {event_key([target])} nao foi gerada pelo Apriori. "
                "As medidas de regra ficam vazias, mas o intervalo continua sendo inferido pelas "
                "restricoes globais."
            )
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Resolve uma consulta probabilistica condicional com programacao linear, "
            "sem depender da interface web."
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Caminho do CSV categorizavel. Padrao: data/Crop_recommendation.csv",
    )
    parser.add_argument("--target", type=parse_condition, help="Evento A. Exemplo: label=rice")
    parser.add_argument(
        "--condition",
        action="append",
        default=[],
        type=parse_condition,
        help="Condicao B. Pode repetir. Exemplo: --condition ph=acido",
    )
    parser.add_argument("--output", type=Path, help="Caminho opcional para salvar o JSON.")
    parser.add_argument(
        "--solver-method",
        choices=[engine["method"] for engine in SOLVER_ENGINES],
        default="highs-ipm",
        help="Metodo do SciPy/HiGHS: highs, highs-ds ou highs-ipm.",
    )
    parser.add_argument("--show-domains", action="store_true", help="Mostra atributos e valores validos.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    data = load_dataset(args.dataset)

    if args.show_domains:
        print(
            json.dumps(
                {
                    "dataset": str(args.dataset),
                    "total": data["total"],
                    "attributes": data["attributes"],
                    "numericAttributes": data["numericAttributes"],
                    "categoricalAttributes": data["categoricalAttributes"],
                    "domains": data["domains"],
                    "thresholds": data["thresholds"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        if not args.target:
            return 0

    if not args.target:
        parser.error("informe --target atributo=valor ou use --show-domains para listar os valores")

    try:
        target = validate_conditions([args.target], data["domains"])[0]
        conditions = validate_conditions(args.condition, data["domains"])
        solver_name = next(
            (engine["name"] for engine in SOLVER_ENGINES if engine["method"] == args.solver_method),
            args.solver_method,
        )
        result = compute_query(data, target, conditions, args.solver_method, solver_name)
    except ValueError as error:
        print(f"Erro: {error}")
        print("Use --show-domains para ver atributos e valores validos.")
        return 2

    output_text = json.dumps(result, ensure_ascii=False, indent=2)
    print(output_text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output_text + "\n", encoding="utf-8")
        print(f"\nResultado salvo em: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
