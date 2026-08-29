from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import os
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request, send_from_directory

from apriori_rules import mine_apriori_rules


# MAPA DO EXERCICIO
#
# Este arquivo e o backend principal do trabalho. Ele implementa o roteiro passado
# no quadro:
# 1. carrega a base e transforma atributos numericos em categorias;
# 2. forma Ω com os mundos observados e minera itemsets/regras com Apriori;
# 3. escreve marginais, conjuntas e regras Apriori como restricoes lineares;
# 4. recebe da interface uma pergunta P(A | B), escolhida pelo usuario;
# 5. transforma a razao P(A e B) / P(B) por Charnes-Cooper e resolve com HiGHS;
# 6. gera textos, PDFs e comparacao com o solver separado.
ROOT = Path(__file__).resolve().parent
DATASET_PATH = ROOT / "data" / "Crop_recommendation.csv"
GENERATED_REPORT_DIR = ROOT / "reports" / "generated"
QUERY_REPORT_PATH = GENERATED_REPORT_DIR / "relatorio_consulta_atual.pdf"
SOLVER_COMPARISON_REPORT_PATH = GENERATED_REPORT_DIR / "relatorio_comparacao_solver.pdf"
FULL_LINEAR_PROGRAM_PATH = GENERATED_REPORT_DIR / "programa_linear_completo.txt"
APRIORI_MIN_SUPPORT = 0.01
APRIORI_MIN_CONFIDENCE = 0.0
APRIORI_MAX_ITEMSET_SIZE = 3
APRIORI_RULE_PREVIEW_LIMIT = 50

# Solvers realmente executados na comparacao. Todos usam scipy.optimize.linprog,
# variando o metodo do HiGHS para medir consistencia e tempo.
SOLVER_ENGINES = [
    {
        "id": "highs",
        "name": "SciPy HiGHS",
        "method": "highs",
        "engine": "scipy.optimize.linprog(method='highs')",
        "status": "Executado na comparacao pelo script separado",
        "comparison": "Comparacao numerica ativa com os parametros da interface",
        "notes": "Escolha automatica do HiGHS usada como referencia comparativa.",
    },
    {
        "id": "highs-ds",
        "name": "HiGHS Dual Simplex",
        "method": "highs-ds",
        "engine": "scipy.optimize.linprog(method='highs-ds')",
        "status": "Executado no script separado",
        "comparison": "Comparacao de metricas contra a mesma consulta da interface",
        "notes": "Usa a estrategia dual simplex do HiGHS.",
    },
    {
        "id": "highs-ipm",
        "name": "HiGHS Interior Point",
        "method": "highs-ipm",
        "engine": "scipy.optimize.linprog(method='highs-ipm')",
        "status": "Executado no projeto principal",
        "comparison": "Comparacao de metricas contra a mesma consulta da interface",
        "notes": "Usa o metodo de pontos interiores do HiGHS; o resultado principal e reaproveitado na tabela comparativa.",
    },
]
DOCUMENTED_SOLVERS = [
    {
        "id": "gurobi",
        "name": "Gurobi",
        "engine": "gurobipy",
        "status": "Comparacao documental",
        "comparison": "Nao executado no Render por depender de instalacao/licenca",
        "notes": "Solver comercial forte para LP/MILP; referencia de benchmark futuro.",
    },
    {
        "id": "lp-solve",
        "name": "lp_solve",
        "engine": "lp_solve",
        "status": "Comparacao documental",
        "comparison": "Nao executado no Render nesta versao",
        "notes": "Solver livre tradicional para LP/MILP; util como comparacao externa futura.",
    },
    {
        "id": "cupdlp-c",
        "name": "cuPDLP-C",
        "engine": "COPT-Public/cuPDLP-C",
        "status": "Comparacao documental",
        "comparison": "Nao executado no Render nesta versao",
        "notes": "Referencia moderna para LP em larga escala.",
    },
]
KNOWN_LABELS = {
    "N": "Nitrogenio",
    "P": "Fosforo",
    "K": "Potassio",
    "temperature": "Temperatura",
    "humidity": "Umidade",
    "ph": "pH",
    "rainfall": "Chuva",
    "label": "Cultura",
}

app = Flask(__name__, static_folder=None)


def quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    base = math.floor(position)
    rest = position - base
    if base + 1 >= len(ordered):
        return ordered[base]
    return ordered[base] + rest * (ordered[base + 1] - ordered[base])


def label_for(attribute: str) -> str:
    return KNOWN_LABELS.get(attribute, attribute.replace("_", " ").title())


def is_number(value: str) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


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


def event_key(conditions: list[dict[str, str]]) -> str:
    if not conditions:
        return "verdadeiro"
    return ", ".join(f"{item['attribute']}={item['value']}" for item in conditions)


def valid_conditions(conditions: list[dict[str, Any]], domains: dict[str, list[str]]) -> list[dict[str, str]]:
    cleaned = []
    for condition in conditions:
        attribute = str(condition.get("attribute", "")).strip()
        value = str(condition.get("value", "")).strip()
        if attribute not in domains or value not in domains[attribute]:
            raise ValueError("Condicao invalida.")
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


@lru_cache(maxsize=1)
def load_dataset() -> dict[str, Any]:
    # Item 1a do exercicio: preparar a base categorica.
    # A base original tem atributos numericos de solo/clima. Para permitir
    # perguntas logicas do tipo "N=alto" e "ph=alcalino", cada valor numerico
    # e convertido para uma faixa categorica. O resultado e guardado em cache
    # porque as consultas usam sempre a mesma base.
    with DATASET_PATH.open("r", encoding="utf-8", newline="") as file:
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
        # Para atributos numericos gerais, os tercis dividem os valores em
        # baixo, medio e alto. O pH usa uma regra propria por ter interpretacao
        # quimica natural: acido, neutro e alcalino.
        values = [float(row[attribute]) for row in raw_rows]
        thresholds[attribute] = {
            "low": quantile(values, 1 / 3),
            "high": quantile(values, 2 / 3),
        }

    categorical_rows = []
    for row in raw_rows:
        categorized = {
            attribute: category_for(attribute, row[attribute], thresholds)
            for attribute in numeric_attributes
        }
        for attribute in categorical_attributes:
            categorized[attribute] = row[attribute]
        categorical_rows.append(categorized)

    domains = {
        attribute: sorted({row[attribute] for row in categorical_rows})
        for attribute in attributes
    }
    world_counts: dict[tuple[str, ...], int] = {}
    for row in categorical_rows:
        key = tuple(row[attribute] for attribute in attributes)
        world_counts[key] = world_counts.get(key, 0) + 1
    # Cada combinacao categorica observada vira um "mundo possivel" w.
    # No programa linear, cada mundo recebe uma variavel x_w.
    worlds = [
        {"values": dict(zip(attributes, key)), "count": count}
        for key, count in world_counts.items()
    ]

    return {
        "attributes": attributes,
        "numericAttributes": numeric_attributes,
        "categoricalAttributes": categorical_attributes,
        "rows": categorical_rows,
        "worlds": worlds,
        "domains": domains,
        "labels": {attribute: label_for(attribute) for attribute in attributes},
        "thresholds": thresholds,
        "total": len(categorical_rows),
    }


def matches(row: dict[str, str], conditions: list[dict[str, str]]) -> bool:
    return all(row[condition["attribute"]] == condition["value"] for condition in conditions)


def probability(rows: list[dict[str, str]], conditions: list[dict[str, str]]) -> float:
    # Frequencia empirica usada como probabilidade: P(condicoes) =
    # quantidade de linhas que satisfazem as condicoes / total de linhas.
    if not conditions:
        return 1.0
    return sum(1 for row in rows if matches(row, conditions)) / len(rows)


def probability_count(rows: list[dict[str, str]], conditions: list[dict[str, str]]) -> int:
    if not conditions:
        return len(rows)
    return sum(1 for row in rows if matches(row, conditions))


def rounded_interval(value: float, width: float = 0.001) -> tuple[float, float]:
    # Item 1a: trabalhar com intervalos evita que arredondamentos e ponto
    # flutuante deixem o PL artificialmente inviavel. Ex.: 0.976 vira uma faixa
    # pequena em torno do valor arredondado.
    rounded = round(value, 3)
    return max(0.0, rounded - width), min(1.0, rounded + width)


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


def world_mask(worlds: list[dict[str, Any]], conditions: list[dict[str, str]]) -> list[float]:
    # Vetor indicador do evento. Se o mundo w satisfaz as condicoes, a posicao
    # vale 1; caso contrario, 0. Assim soma(mask[w] * x_w) representa P(evento).
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
    worlds: list[dict[str, Any]],
    rows: list[dict[str, str]],
    target: dict[str, str],
    base: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], int]:
    """Add zero-count completions when A and B never occur together.

    The empirical dataset remains unchanged: added worlds have count zero. They
    only give the LP variables for logically possible A-and-B combinations, so
    empirical P(A and B)=0 does not force the optimized upper bound to zero.
    """
    both = [*base, target]
    if any(matches(world["values"], both) for world in worlds):
        return worlds, 0

    attributes = list(rows[0].keys())
    domains = {
        attribute: sorted({row[attribute] for row in rows})
        for attribute in attributes
    }
    fixed: dict[str, str] = {}
    for condition in both:
        previous = fixed.get(condition["attribute"])
        if previous is not None and previous != condition["value"]:
            return worlds, 0
        fixed[condition["attribute"]] = condition["value"]

    free_attributes = [attribute for attribute in attributes if attribute not in fixed]
    observed_keys = {
        tuple(world["values"][attribute] for attribute in attributes)
        for world in worlds
    }
    additions: list[dict[str, Any]] = []
    for values in itertools.product(*(domains[attribute] for attribute in free_attributes)):
        completed = dict(fixed)
        completed.update(dict(zip(free_attributes, values)))
        key = tuple(completed[attribute] for attribute in attributes)
        if key in observed_keys:
            continue
        additions.append(
            {
                "values": completed,
                "count": 0,
                "queryCompletion": True,
            }
        )
    return [*worlds, *additions], len(additions)


def mask_expression(conditions: list[dict[str, str]]) -> str:
    return f"soma(x_w onde {event_key(conditions)})"


def fmt_lp_number(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.3f}"


def association_rule_description(
    rows: list[dict[str, str]],
    antecedent: list[dict[str, str]],
    consequent: list[dict[str, str]],
) -> str:
    p_antecedent = probability(rows, antecedent)
    p_consequent = probability(rows, consequent)
    p_both = probability(rows, [*antecedent, *consequent])
    confidence = p_both / p_antecedent if p_antecedent > 0 else None
    lift = confidence / p_consequent if confidence is not None and p_consequent > 0 else None
    return (
        f"{event_key(antecedent)} -> {event_key(consequent)} | "
        f"P(antecedente)={fmt_lp_number(p_antecedent)}, "
        f"P(consequente)={fmt_lp_number(p_consequent)}, "
        f"P(antecedente e consequente)={fmt_lp_number(p_both)}, "
        f"confianca={fmt_lp_number(confidence)}, "
        f"lift={fmt_lp_number(lift)}"
    )


def association_rule_status(
    rows: list[dict[str, str]],
    antecedent: list[dict[str, str]],
    consequent: list[dict[str, str]],
) -> dict[str, Any]:
    # Diagnostico empirico. Estes numeros nao tornam a consulta uma regra e nao
    # medem acuracia de classificacao. A regra so entra no PL quando o Apriori a
    # gera a partir de um itemset frequente.
    p_antecedent = probability(rows, antecedent)
    p_consequent = probability(rows, consequent)
    p_both = probability(rows, [*antecedent, *consequent])
    confidence = p_both / p_antecedent if p_antecedent > 0 else None
    lift = confidence / p_consequent if confidence is not None and p_consequent > 0 else None
    return {
        "accepted": False,
        "reason": "diagnostico empirico; a inclusao depende da mineracao Apriori",
        "pAntecedent": p_antecedent,
        "pConsequent": p_consequent,
        "pBoth": p_both,
        "confidence": confidence,
        "lift": lift,
    }


def condition_signature(conditions: list[dict[str, str]]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((item["attribute"], item["value"]) for item in conditions))


def mine_association_rules(
    rows: list[dict[str, str]],
    domains: dict[str, list[str]],
) -> list[dict[str, Any]]:
    # Compatibilidade para relatorios que fornecem linhas em vez de Ω. O
    # algoritmo Apriori propriamente dito recebe mundos unicos com pesos.
    attributes = list(domains.keys())
    world_counts: dict[tuple[str, ...], int] = {}
    for row in rows:
        key = tuple(row[attribute] for attribute in attributes)
        world_counts[key] = world_counts.get(key, 0) + 1
    worlds = [
        {"values": dict(zip(attributes, key)), "count": count}
        for key, count in world_counts.items()
    ]
    return list(
        mine_apriori_rules(
            worlds,
            len(rows),
            min_support=APRIORI_MIN_SUPPORT,
            min_confidence=APRIORI_MIN_CONFIDENCE,
            max_itemset_size=APRIORI_MAX_ITEMSET_SIZE,
        )["rules"]
    )


@lru_cache(maxsize=1)
def learned_association_mining() -> dict[str, Any]:
    data = load_dataset()
    return mine_apriori_rules(
        data["worlds"],
        data["total"],
        min_support=APRIORI_MIN_SUPPORT,
        min_confidence=APRIORI_MIN_CONFIDENCE,
        max_itemset_size=APRIORI_MAX_ITEMSET_SIZE,
    )


@lru_cache(maxsize=1)
def learned_association_rules() -> tuple[dict[str, Any], ...]:
    return tuple(learned_association_mining()["rules"])


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


def association_rule_payload(
    rule: dict[str, Any] | None,
    antecedent: list[dict[str, str]],
    consequent: list[dict[str, str]],
) -> dict[str, Any]:
    thresholds = {
        "support": APRIORI_MIN_SUPPORT,
        "confidence": APRIORI_MIN_CONFIDENCE,
        "maxItemsetSize": APRIORI_MAX_ITEMSET_SIZE,
    }
    if rule is None:
        return {
            "antecedent": antecedent,
            "consequent": consequent,
            "support": None,
            "confidence": None,
            "lift": None,
            "released": False,
            "reason": "regra nao gerada pelo Apriori com os limiares configurados",
            "thresholds": thresholds,
        }
    return {
        "antecedent": rule["antecedent"],
        "consequent": rule["consequent"],
        "support": rule["support"],
        "confidence": rule["confidence"],
        "lift": rule["lift"],
        "released": True,
        "reason": rule.get("source", "regra gerada pelo Apriori"),
        "thresholds": thresholds,
    }


def query_association_rule(
    rules: list[dict[str, Any]],
    antecedent: list[dict[str, str]],
    consequent: list[dict[str, str]],
) -> dict[str, Any] | None:
    # A consulta B -> A so recebe medidas de regra quando essa regra pertence a
    # saida do Apriori. Nao recalculamos uma regra ad hoc para preencher cards.
    return find_released_association_rule(rules, antecedent, consequent)


def build_linear_constraints(
    worlds: list[dict[str, Any]],
    rows: list[dict[str, str]],
    target: dict[str, str],
    base: list[dict[str, str]],
) -> tuple[list[dict[int, float]], list[float], list[dict[str, Any]]]:
    # Item 1b: converter probabilidades empiricas em restricoes lineares.
    # A variavel do PL e x_w, a massa de probabilidade de cada mundo possivel.
    # Para cada evento E, criamos:
    #   lower <= soma(x_w onde E) <= upper
    # usando intervalos em torno das frequencias observadas na base.
    a_ub: list[dict[int, float]] = []
    b_ub: list[float] = []
    records: list[dict[str, Any]] = []
    interval_events: set[tuple[tuple[str, str], ...]] = set()
    attributes = list(rows[0].keys())
    domains = {
        attribute: sorted({row[attribute] for row in rows})
        for attribute in attributes
    }
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
        mask = cached_sparse_mask(conditions)
        lower, upper = rounded_interval(value)
        row_indexes = [len(a_ub), len(a_ub) + 1]
        a_ub.append(mask)
        b_ub.append(upper)
        a_ub.append({index: -value for index, value in mask.items()})
        b_ub.append(-lower)
        records.append(
            {
                "kind": kind,
                "conditions": conditions,
                "lower": lower,
                "upper": upper,
                "value": value,
                "expression": mask_expression(conditions),
                "rowIndexes": row_indexes,
            }
        )
        return True

    def add_apriori_rule(rule: dict[str, Any]) -> None:
        # Uma regra Apriori R -> S fornece duas probabilidades para o PL:
        #   suporte s = P(R e S), uma restricao linear direta; e
        #   confianca c = P(S | R), linearizada pelas desigualdades abaixo.
        # Lift permanece metadado descritivo e nao entra como coeficiente.
        #   lower <= P(R e S) / P(R) <= upper
        #   P(R e S) - upper.P(R) <= 0
        #  -P(R e S) + lower.P(R) <= 0
        antecedent = rule["antecedent"]
        consequent = rule["consequent"]
        both = [*antecedent, *consequent]
        support_constraint_added = add_interval(
            "apriori_rule_support",
            both,
            rule["support"],
        )
        confidence = rule["confidence"]
        lower, upper = rounded_interval(confidence)
        antecedent_mask = cached_sparse_mask(antecedent)
        both_mask = cached_sparse_mask(both)
        confidence_row_indexes = [len(a_ub), len(a_ub) + 1]
        a_ub.append(add_sparse_vectors(both_mask, antecedent_mask, -upper))
        b_ub.append(0.0)
        negative_both = {index: -value for index, value in both_mask.items()}
        a_ub.append(add_sparse_vectors(negative_both, antecedent_mask, lower))
        b_ub.append(0.0)
        records.append(
            {
                "kind": "apriori_rule_confidence",
                "antecedent": antecedent,
                "consequent": consequent,
                "lower": lower,
                "upper": upper,
                "value": confidence,
                "support": rule["support"],
                "lift": rule["lift"],
                "source": "saida do algoritmo Apriori",
                "supportConstraintAdded": support_constraint_added,
                "liftUsage": "descritivo; nao usado como restricao nem acuracia",
                "rowIndexes": confidence_row_indexes,
                "expression": (
                    f"{lower:.3f} <= {mask_expression(both)} / "
                    f"{mask_expression(antecedent)} <= {upper:.3f}"
                ),
            }
        )

    for attribute in attributes:
        for value in domains[attribute]:
            # Probabilidades marginais de cada valor de cada variavel:
            # P(N=alto), P(label=banana), P(ph=alcalino), etc.
            conditions = [{"attribute": attribute, "value": value}]
            add_interval("marginal", conditions, weighted_probability(conditions))

    for index, left_attribute in enumerate(attributes):
        for right_attribute in attributes[index + 1 :]:
            for left_value in domains[left_attribute]:
                for right_value in domains[right_attribute]:
                    # Probabilidades conjuntas por pares de valores:
                    # P(N=alto e P=alto), P(label=banana e ph=neutro), etc.
                    conditions = [
                        {"attribute": left_attribute, "value": left_value},
                        {"attribute": right_attribute, "value": right_value},
                    ]
                    add_interval("pairwise_joint", conditions, weighted_probability(conditions))

    learned_rules = list(learned_association_rules())
    for rule in learned_rules:
        add_apriori_rule(rule)

    # A consulta nao injeta P(A e B) observado como resposta pronta. O solver
    # infere seus limites usando marginais, conjuntas por pares e todas as
    # restricoes produzidas pelas regras Apriori.

    return a_ub, b_ub, records


@lru_cache(maxsize=1)
def cached_linear_constraint_model(
) -> tuple[list[dict[int, float]], list[float], list[dict[str, Any]]]:
    """Build the dataset-wide constraints once per server process."""
    data = load_dataset()
    return build_linear_constraints(data["worlds"], data["rows"], {}, [])


def _canonical_sparse_rows(rows: list[dict[int, float]]) -> list[list[list[float | int]]]:
    """Return a deterministic coordinate representation for hashing/export."""
    return [
        [[column, float(value)] for column, value in sorted(row.items())]
        for row in rows
    ]


def _constraint_sources(
    records: list[dict[str, Any]],
    row_count: int,
) -> list[dict[str, Any]]:
    """Map every numeric A_ub row back to the evidence that created it."""
    sources: list[dict[str, Any]] = [
        {"kind": "dataset_constraint", "side": "unknown", "description": "restricao global"}
        for _ in range(row_count)
    ]
    for record in records:
        row_indexes = record.get("rowIndexes", [])
        if "conditions" in record:
            description = event_key(record["conditions"])
        else:
            description = (
                f"{event_key(record.get('antecedent', []))} -> "
                f"{event_key(record.get('consequent', []))}"
            )
        for position, row_index in enumerate(row_indexes):
            if 0 <= row_index < row_count:
                sources[row_index] = {
                    "kind": record["kind"],
                    "side": "upper" if position == 0 else "lower",
                    "description": description,
                }
    return sources


def build_transformed_linear_program(
    worlds: list[dict[str, Any]],
    rows: list[dict[str, str]],
    target: dict[str, str],
    base: list[dict[str, str]],
) -> dict[str, Any]:
    """Build the exact Charnes-Cooper model consumed by ``linprog``.

    The returned sparse rows, right-hand sides, objectives and bounds are the
    single source of truth for both optimization and the auditable TXT export.
    """
    n_worlds = len(worlds)
    t_index = n_worlds
    if any(world.get("queryCompletion") for world in worlds):
        # The numerical limits still come exclusively from the observed rows;
        # zero-count completions only add eligible LP columns.
        a_ub, b_ub, records = build_linear_constraints(worlds, rows, target, base)
    else:
        a_ub, b_ub, records = cached_linear_constraint_model()
    both = [*base, target]
    denominator_mask = sparse_world_mask(worlds, base)
    numerator_mask = sparse_world_mask(worlds, both)

    transformed_a_ub: list[dict[int, float]] = []
    for row, limit in zip(a_ub, b_ub):
        transformed = dict(row)
        if abs(limit) > 1e-15:
            transformed[t_index] = -float(limit)
        transformed_a_ub.append(transformed)

    transformed_a_eq = [
        {**{index: 1.0 for index in range(n_worlds)}, t_index: -1.0},
        dict(denominator_mask),
    ]
    objective_lower = dict(numerator_mask)
    objective_upper_as_min = {index: -value for index, value in numerator_mask.items()}
    variable_names = [f"y_{index + 1:04d}" for index in range(n_worlds)] + ["t"]
    bounds: list[tuple[float, None]] = [(0.0, None)] * (n_worlds + 1)

    digest_payload = {
        "query": {"target": target, "base": base},
        "worlds": [
            {"values": world["values"], "count": int(world.get("count", 1))}
            for world in worlds
        ],
        "variableNames": variable_names,
        "objectiveLower": _canonical_sparse_rows([objective_lower])[0],
        "objectiveUpperAsMin": _canonical_sparse_rows([objective_upper_as_min])[0],
        "aUb": _canonical_sparse_rows(transformed_a_ub),
        "bUb": [0.0] * len(transformed_a_ub),
        "aEq": _canonical_sparse_rows(transformed_a_eq),
        "bEq": [0.0, 1.0],
        "bounds": [[0.0, None] for _ in bounds],
    }
    canonical = json.dumps(
        digest_payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    return {
        "worldVariables": n_worlds,
        "solverVariables": n_worlds + 1,
        "tIndex": t_index,
        "variableNames": variable_names,
        "objectiveLower": objective_lower,
        "objectiveUpperAsMin": objective_upper_as_min,
        "aUb": transformed_a_ub,
        "bUb": [0.0] * len(transformed_a_ub),
        "aEq": transformed_a_eq,
        "bEq": [0.0, 1.0],
        "bounds": bounds,
        "rowSources": _constraint_sources(records, len(transformed_a_ub)),
        "records": records,
        "digest": hashlib.sha256(canonical).hexdigest(),
    }


def solve_linear_interval(
    worlds: list[dict[str, Any]],
    rows: list[dict[str, str]],
    target: dict[str, str],
    base: list[dict[str, str]],
    solver_method: str = "highs-ipm",
    solver_name: str = "SciPy HiGHS Interior Point",
) -> dict[str, Any]:
    # Itens 1c, 1d e 1e: resolver a pergunta do usuario.
    # A interface escolhe A e B; o objetivo matematico e minimizar e maximizar
    # P(A | B) = P(A e B) / P(B), respeitando todas as restricoes montadas.
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
        from scipy.sparse import coo_matrix
    except Exception as error:
        return {"ok": False, "error": f"scipy indisponivel: {error}"}

    model_worlds, completion_count = complete_unobserved_query_worlds(
        worlds,
        rows,
        target,
        base,
    )
    model = build_transformed_linear_program(model_worlds, rows, target, base)
    variable_count = model["solverVariables"]

    def sparse_matrix(sparse_rows: list[dict[int, float]]) -> Any:
        values: list[float] = []
        row_indexes: list[int] = []
        column_indexes: list[int] = []
        for row_index, row in enumerate(sparse_rows):
            for column_index, value in row.items():
                row_indexes.append(row_index)
                column_indexes.append(column_index)
                values.append(value)
        return coo_matrix(
            (values, (row_indexes, column_indexes)),
            shape=(len(sparse_rows), variable_count),
        ).tocsr()

    transformed_a_ub = sparse_matrix(model["aUb"])
    transformed_a_eq = sparse_matrix(model["aEq"])

    def dense_objective(sparse_objective: dict[int, float]) -> list[float]:
        return [sparse_objective.get(index, 0.0) for index in range(variable_count)]

    def optimize(sparse_objective: dict[int, float]) -> Any:
        return linprog(
            c=dense_objective(sparse_objective),
            A_ub=transformed_a_ub,
            b_ub=model["bUb"],
            A_eq=transformed_a_eq,
            b_eq=model["bEq"],
            bounds=model["bounds"],
            method=solver_method,
        )

    lower_result = optimize(model["objectiveLower"])
    upper_result = optimize(model["objectiveUpperAsMin"])
    if not lower_result.success or not upper_result.success:
        return {
            "ok": False,
            "error": lower_result.message if not lower_result.success else upper_result.message,
        }

    lower_hi = clean_probability(float(lower_result.fun))
    upper_lo = clean_probability(float(-upper_result.fun))
    records = model["records"]

    return {
        "ok": True,
        "lower": lower_hi,
        "upper": upper_lo,
        "variables": model["solverVariables"],
        "worldVariables": model["worldVariables"],
        "observedWorldVariables": len(worlds),
        "queryCompletionWorlds": completion_count,
        "empiricalJointZero": probability_count(rows, [*base, target]) == 0,
        "solverVariables": model["solverVariables"],
        "constraints": transformed_a_ub.shape[0] + transformed_a_eq.shape[0],
        "baseConstraints": len(model["aUb"]),
        "constraintSummary": {
            "marginal": sum(1 for item in records if item["kind"] == "marginal"),
            "pairwiseJoint": sum(1 for item in records if item["kind"] == "pairwise_joint"),
            "aprioriRuleSupport": sum(1 for item in records if item["kind"] == "apriori_rule_support"),
            "aprioriRuleConfidence": sum(1 for item in records if item["kind"] == "apriori_rule_confidence"),
            "aprioriRules": len(learned_association_rules()),
        },
        "solver": f"scipy.optimize.linprog {solver_method}",
        "solverMethod": solver_method,
        "solverName": solver_name,
        "modelDigest": model["digest"],
    }


def linear_program_text(
    target: dict[str, str],
    base: list[dict[str, str]],
    p_a: float,
    p_b: float,
    p_ab: float,
    lp: dict[str, Any],
) -> str:
    lines = [
        "FORMULACAO MATEMATICA RESUMIDA DA CONSULTA",
        "Esta e uma explicacao didatica; nao e a matriz numerica enviada ao solver.",
        "Use o TXT auditavel para consultar A_ub, b_ub, A_eq, b_eq, objetivos e limites exatos.",
        "",
        "1. Eventos da consulta:",
        f"  A = {event_key([target])}",
        f"  B = {event_key(base)}",
        f"  A e B = {event_key([*base, target])}",
        "",
        "2. Variaveis:",
        "  Cada mundo possivel w e uma combinacao categorica completa.",
        "  Mundos observados preservam sua contagem; completamentos da consulta tem contagem zero.",
        "  x_w >= 0 representa a massa de probabilidade atribuida ao mundo w.",
        "",
        "3. Normalizacao probabilistica:",
        "  soma(x_w) = 1",
        "",
        "4. Evidencias globais usadas como restricoes intervalares:",
        "  O LP inclui marginais de cada valor e conjuntas por pares de valores.",
        "  P(A), P(B) e P(A e B) abaixo sao exibidos para auditoria da consulta:",
        f"  P(A) empirico = {p_a:.3f}",
        f"  P(B) empirico = {p_b:.3f}",
        f"  P(A e B) empirico = {p_ab:.3f}",
        "  A consulta nao injeta P(A e B) como resposta pronta no LP.",
        "",
        "5. Mineracao Apriori e regras lineares:",
        "  O Apriori recebe os mundos observados de Omega como transacoes ponderadas.",
        "  Para cada regra R -> S gerada, o suporte ancora P(R e S).",
        "  A confianca ancora P(R e S) = confianca.P(R), em forma linear intervalar.",
        "  Todas as regras que atingem suporte e confianca minimos entram no LP.",
        "",
        "6. Papel do lift:",
        "  Lift e apenas uma medida descritiva da associacao retornada pelo Apriori.",
        "  Ele nao mede acuracia, nao seleciona as regras e nao vira coeficiente do LP.",
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
                f"Variaveis de mundos: {lp.get('worldVariables', lp['variables'])}",
                f"Mundos observados: {lp.get('observedWorldVariables', '-')}",
                f"Mundos completados para a consulta: {lp.get('queryCompletionWorlds', 0)}",
                f"Variaveis enviadas ao solver (y_w e t): {lp.get('solverVariables', lp['variables'])}",
                f"Restricoes: {lp['constraints']}",
                f"SHA-256 do modelo numerico: {lp.get('modelDigest', '-')}",
                f"Resumo: {lp.get('constraintSummary', {})}",
                f"Intervalo retornado: {fmt_probability(lp['lower'])} <= P(A | B) <= {fmt_probability(lp['upper'])}",
            ]
        )
    elif lp.get("reason") == "zero_denominator":
        lines.extend(
            [
                "",
                "Consulta nao resolvida pelo solver:",
                "  P(B)=0 na base, entao P(A | B) = P(A e B) / P(B) teria denominador zero.",
                "  Escolha menos afirmacoes ou uma combinacao B que exista no dataset.",
            ]
        )
    else:
        lines.extend(["", f"Solver indisponivel: {lp.get('error', 'erro desconhecido')}"])
    return "\n".join(lines)


def full_linear_program_text(
    worlds: list[dict[str, Any]],
    rows: list[dict[str, str]],
    target: dict[str, str],
    base: list[dict[str, str]],
    lp: dict[str, Any],
) -> str:
    model_worlds, completion_count = complete_unobserved_query_worlds(
        worlds,
        rows,
        target,
        base,
    )
    model = build_transformed_linear_program(model_worlds, rows, target, base)
    solver_digest = lp.get("modelDigest")
    if solver_digest and solver_digest != model["digest"]:
        raise RuntimeError(
            "O modelo reconstruido para exportacao difere daquele resolvido pelo solver."
        )

    def exact_number(value: float) -> str:
        if abs(value) <= 1e-15:
            return "0"
        return format(float(value), ".17g")

    def sparse_row(row: dict[int, float]) -> str:
        if not row:
            return "{}"
        entries = []
        for column, value in sorted(row.items()):
            entries.append(
                f"{column}:{model['variableNames'][column]}:{exact_number(value)}"
            )
        return "{" + ";".join(entries) + "}"

    lines = [
        "PROGRAMA LINEAR NUMERICO AUDITAVEL",
        "",
        "Este TXT e gerado a partir do mesmo objeto numerico convertido em matrizes CSR",
        "e entregue a scipy.optimize.linprog. Nao e uma formula resumida ou ilustrativa.",
        "Indices de linhas e colunas abaixo comecam em zero, como no codigo Python.",
        "",
        "IDENTIFICACAO",
        f"consulta=P({event_key([target])} | {event_key(base)})",
        f"solver={lp.get('solver', 'scipy.optimize.linprog highs-ipm')}",
        f"sha256_modelo={model['digest']}",
        f"sha256_confirmado_pelo_solver={solver_digest or model['digest']}",
        f"mundos={model['worldVariables']}",
        f"mundos_observados={len(worlds)}",
        f"mundos_completados_para_consulta={completion_count}",
        f"variaveis_solver={model['solverVariables']}",
        f"linhas_A_ub={len(model['aUb'])}",
        f"linhas_A_eq={len(model['aEq'])}",
        f"restricoes_totais={len(model['aUb']) + len(model['aEq'])}",
        "",
        "FORMATO DAS LINHAS ESPARSAS",
        "{coluna:nome_variavel:coeficiente;...}",
        "A_ub[i] usa <= b_ub[i]; A_eq[i] usa = b_eq[i].",
        "",
        "MAPEAMENTO EXATO DAS VARIAVEIS",
    ]
    for column, world in enumerate(model_worlds):
        values = json.dumps(world["values"], ensure_ascii=False, sort_keys=True)
        lines.append(
            f"col={column};var={model['variableNames'][column]};"
            f"mundo=w_{column + 1:04d};contagem={int(world.get('count', 1))};"
            f"origem={'completado_para_consulta' if world.get('queryCompletion') else 'observado'};"
            f"valores={values};relacao_original=x_{column + 1:04d}=y_{column + 1:04d}/t"
        )
    lines.append(
        f"col={model['tIndex']};var=t;descricao=escala de Charnes-Cooper;limite_inferior=0"
    )

    lines.extend(
        [
            "",
            "OBJETIVOS EXATOS ENTREGUES AO LINPROG",
            f"c_lower={sparse_row(model['objectiveLower'])}",
            f"c_upper_as_min={sparse_row(model['objectiveUpperAsMin'])}",
            "limite_inferior=fun(c_lower)",
            "limite_superior=-fun(c_upper_as_min)",
            "",
            "DESIGUALDADES EXATAS",
        ]
    )
    for row_index, (row, rhs, source) in enumerate(
        zip(model["aUb"], model["bUb"], model["rowSources"])
    ):
        lines.append(
            f"A_ub[{row_index}]={sparse_row(row)} <= "
            f"b_ub[{row_index}]={exact_number(rhs)} | "
            f"origem={source['kind']}:{source['side']}:{source['description']}"
        )

    lines.extend(["", "IGUALDADES EXATAS"])
    equality_sources = [
        "normalizacao_transformada: soma(y_w)-t=0",
        f"denominador_transformado: evento B ({event_key(base)})=1",
    ]
    for row_index, (row, rhs, source) in enumerate(
        zip(model["aEq"], model["bEq"], equality_sources)
    ):
        lines.append(
            f"A_eq[{row_index}]={sparse_row(row)} = "
            f"b_eq[{row_index}]={exact_number(rhs)} | origem={source}"
        )

    lines.extend(["", "LIMITES EXATOS DAS VARIAVEIS"])
    for column, (lower, upper) in enumerate(model["bounds"]):
        upper_text = "None" if upper is None else exact_number(upper)
        lines.append(
            f"bounds[{column}:{model['variableNames'][column]}]="
            f"({exact_number(lower)},{upper_text})"
        )

    lines.extend(
        [
            "",
            "CHAMADA REPRODUZIDA",
            "linprog(c=c_lower, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method=solver_method)",
            "linprog(c=c_upper_as_min, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method=solver_method)",
            "",
            "RESULTADO DO MESMO MODELO",
        ]
    )
    if lp.get("ok"):
        lines.extend(
            [
                f"limite_inferior={exact_number(lp['lower'])}",
                f"limite_superior={exact_number(lp['upper'])}",
                f"intervalo={fmt_probability(lp['lower'])} <= P(A | B) <= {fmt_probability(lp['upper'])}",
            ]
        )
    else:
        lines.append(f"status=nao_resolvido;motivo={lp.get('error', 'erro desconhecido')}")

    learned_rules = list(learned_association_rules())
    released_rule = query_association_rule(learned_rules, base, [target])
    lines.extend(["", "STATUS DA REGRA CONSULTADA"])
    if released_rule is None:
        lines.extend(
            [
                "status=nao_se_aplica_como_regra_minerada",
                "motivo=a regra B -> A nao foi gerada pelo Apriori com os parametros atuais",
                "efeito=o intervalo do solver permanece valido e usa as restricoes globais",
            ]
        )
    else:
        lines.extend(
            [
                "status=regra_gerada_pelo_apriori",
                f"suporte={exact_number(released_rule['support'])}",
                f"confianca={exact_number(released_rule['confidence'])}",
                f"lift_descritivo={exact_number(released_rule['lift'])}",
            ]
        )

    return "\n".join(lines) + "\n"


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
            "linha do dataset satisfaz todas as afirmacoes escolhidas. A conclusao "
            "tecnica e que essa combinacao nao tem evidencia empirica na base; reduza "
            "a quantidade de condicoes ou escolha valores mais frequentes."
        )

    interval_sentence = ""
    if lp.get("ok"):
        interval_sentence = (
            f" Pelo modelo linear intervalar, P(A | B) fica entre {fmt_probability(lp['lower'])} "
            f"e {fmt_probability(lp['upper'])}."
        )

    return (
        f"O Apriori gerou a regra {base_label} -> {target_label} com suporte "
        f"{support:.3f} e confianca {confidence:.3f}. Esses valores foram convertidos "
        "em restricoes lineares de probabilidade. "
        f"O lift {fmt_lp_number(lift)} fica apenas como descricao da associacao: nao e "
        "acuracia, nao seleciona a regra e nao entra como coeficiente do programa linear. "
        f"Na base, existem {count_both} ocorrencias conjuntas em {count_base} casos de B."
        f"{interval_sentence}"
    )


@app.get("/")
def home():
    response = send_from_directory(ROOT, "index.html")
    response.cache_control.no_cache = True
    response.cache_control.max_age = 0
    return response


@app.get("/styles.css")
def styles():
    response = send_from_directory(ROOT, "styles.css")
    response.cache_control.no_cache = True
    response.cache_control.max_age = 0
    return response


@app.get("/script.js")
def scripts():
    response = send_from_directory(ROOT, "script.js")
    response.cache_control.no_cache = True
    response.cache_control.max_age = 0
    return response


@app.get("/healthz")
def healthz():
    return jsonify(
        {
            "ok": True,
            # O Render preenche esta variavel com o commit realmente implantado.
            # O robo usa o valor para nao testar por engano uma versao anterior.
            "commit": os.environ.get("RENDER_GIT_COMMIT"),
        }
    )


@app.get("/<path:path>")
def static_files(path: str):
    if not path.startswith("reports/"):
        return jsonify({"ok": False, "error": "Arquivo nao encontrado."}), 404
    return send_from_directory(ROOT, path)


@app.get("/api/metadata")
def metadata():
    data = load_dataset()
    mining = learned_association_mining()
    return jsonify(
        {
            "attributes": data["attributes"],
            "numericAttributes": data["numericAttributes"],
            "categoricalAttributes": data["categoricalAttributes"],
            "total": data["total"],
            "domains": data["domains"],
            "labels": data["labels"],
            "thresholds": data["thresholds"],
            "omegaWorlds": len(data["worlds"]),
            "apriori": {
                key: value
                for key, value in mining.items()
                if key != "rules"
            },
        }
    )


def _compute_query_uncached(
    payload: dict[str, Any],
    solver_method: str = "highs-ipm",
    solver_name: str = "SciPy HiGHS Interior Point",
) -> dict[str, Any]:
    # Ponto central chamado pelo botao "Consultar".
    # Ele valida A e B, calcula probabilidades empiricas, minera/seleciona
    # regras, resolve o PL e devolve tudo que o frontend exibe.
    started_at = datetime.now(timezone.utc)
    started_perf = time.perf_counter()
    data = load_dataset()
    target = valid_conditions([payload.get("target", {})], data["domains"])[0]
    base = normalize_base_conditions(
        valid_conditions(payload.get("conditions", []), data["domains"]),
        target,
    )

    rows = data["rows"]
    both = [*base, target]
    p_a = probability(rows, [target])
    p_b = probability(rows, base)
    p_ab = probability(rows, both)
    # Estes tres valores sao exatamente as evidencias da pergunta:
    # P(A), P(B) e P(A e B). O intervalo do solver fica sobre P(A | B).
    count_a = probability_count(rows, [target])
    count_both = probability_count(rows, both)
    count_base = probability_count(rows, base)
    mining = learned_association_mining()
    learned_rules = list(mining["rules"])
    released_rule = query_association_rule(learned_rules, base, [target])
    queried_rule = association_rule_payload(released_rule, base, [target])
    rule_support = released_rule["support"] if released_rule else None
    rule_confidence = released_rule["confidence"] if released_rule else None
    rule_lift = released_rule["lift"] if released_rule else None
    lp = solve_linear_interval(data["worlds"], rows, target, base, solver_method, solver_name)
    finished_at = datetime.now(timezone.utc)
    duration_seconds = time.perf_counter() - started_perf
    conclusion = (
        conclusion_text(target, base, rule_support, rule_confidence, rule_lift, p_b, count_base, count_both, lp)
        if released_rule
        else (
            f"A consulta {event_key(base)} -> {event_key([target])} nao aparece entre as regras "
            "geradas pelo Apriori. Por isso suporte, confianca e lift nao sao preenchidos como "
            "medidas de uma regra; o intervalo continua sendo inferido pelas restricoes globais."
        )
    )

    return {
        "ok": True,
        "processing": {
            "startedAt": started_at.isoformat(),
            "finishedAt": finished_at.isoformat(),
            "durationSeconds": duration_seconds,
            "durationMilliseconds": round(duration_seconds * 1000, 3),
        },
        "target": target,
        "conditions": base,
        "support": rule_support,
        "confidence": rule_confidence,
        "lift": rule_lift,
        "pAB": p_ab,
        "pA": p_a,
        "pB": p_b,
        "countBoth": count_both,
        "countBase": count_base,
        "countA": count_a,
        "total": data["total"],
        "linear": lp,
        "learnedAssociationRules": [
            {
                "antecedent": rule["antecedent"],
                "consequent": rule["consequent"],
                "support": rule["support"],
                "confidence": rule["confidence"],
                "lift": rule["lift"],
            }
            for rule in learned_rules[:APRIORI_RULE_PREVIEW_LIMIT]
        ],
        "aprioriMining": {
            key: value
            for key, value in mining.items()
            if key != "rules"
        },
        "queriedAssociationRule": queried_rule,
        "releasedAssociationRule": (
            association_rule_payload(released_rule, base, [target])
            if released_rule
            else None
        ),
        "linearProgramSummary": linear_program_text(target, base, p_a, p_b, p_ab, lp),
        "linearProgram": linear_program_text(target, base, p_a, p_b, p_ab, lp),
        "linearProgramFullAvailable": True,
        "conclusion": conclusion,
    }


@lru_cache(maxsize=32)
def _cached_query_result(
    payload_json: str,
    solver_method: str,
    solver_name: str,
) -> dict[str, Any]:
    return _compute_query_uncached(
        json.loads(payload_json),
        solver_method=solver_method,
        solver_name=solver_name,
    )


def compute_query(
    payload: dict[str, Any],
    solver_method: str = "highs-ipm",
    solver_name: str = "SciPy HiGHS Interior Point",
) -> dict[str, Any]:
    # A interface consulta primeiro e compara os solvers depois. Reutilizar o
    # resultado principal elimina duas resolucoes identicas do caminho critico
    # da comparacao, o que mantem a rota abaixo do limite do Render.
    payload_json = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return _cached_query_result(payload_json, solver_method, solver_name)


def compare_number(main_value: float | int | None, solver_value: float | int | None) -> dict[str, Any]:
    if main_value is None or solver_value is None:
        return {
            "main": main_value,
            "solver": solver_value,
            "difference": None,
            "match": main_value == solver_value,
        }
    difference = abs(float(main_value) - float(solver_value))
    return {
        "main": main_value,
        "solver": solver_value,
        "difference": difference,
        "match": difference <= 1e-9,
    }


def compare_solver_result(main_result: dict[str, Any], solver_result: dict[str, Any]) -> dict[str, Any]:
    fields = ["pA", "pB", "pAB", "support", "confidence", "lift", "countBoth", "countBase"]
    metrics = {
        field: compare_number(main_result.get(field), solver_result.get(field))
        for field in fields
    }
    main_linear = main_result.get("linear", {})
    solver_linear = solver_result.get("linear", {})
    metrics["linearLower"] = compare_number(main_linear.get("lower"), solver_linear.get("lower"))
    metrics["linearUpper"] = compare_number(main_linear.get("upper"), solver_linear.get("upper"))
    metrics["variables"] = compare_number(main_linear.get("variables"), solver_linear.get("variables"))
    metrics["constraints"] = compare_number(main_linear.get("constraints"), solver_linear.get("constraints"))
    metrics["durationSeconds"] = compare_number(
        main_result.get("processing", {}).get("durationSeconds"),
        solver_result.get("processing", {}).get("durationSeconds"),
    )
    comparable_metrics = {
        key: item
        for key, item in metrics.items()
        if key != "durationSeconds"
    }
    return {
        "allMatch": all(item["match"] for item in comparable_metrics.values()),
        "metrics": metrics,
    }


def compare_solver_timing(main_result: dict[str, Any], solver_result: dict[str, Any]) -> dict[str, Any]:
    main_seconds = main_result.get("processing", {}).get("durationSeconds")
    solver_seconds = solver_result.get("processing", {}).get("durationSeconds")
    if main_seconds is None or solver_seconds is None:
        return {
            "available": False,
            "mainSeconds": main_seconds,
            "solverSeconds": solver_seconds,
            "message": "Tempo de execucao indisponivel para comparacao.",
        }

    difference = float(main_seconds) - float(solver_seconds)
    percent = (abs(difference) / float(main_seconds) * 100) if float(main_seconds) > 0 else None
    if abs(difference) <= 1e-9:
        message = "Os dois caminhos tiveram praticamente o mesmo tempo de execucao."
        faster = "equal"
    elif difference > 0:
        message = (
            f"O solver separado foi {difference:.3f} segundos mais rapido "
            f"({percent:.1f}% de reducao no tempo)."
        )
        faster = "standaloneSolver"
    else:
        message = (
            f"O projeto principal foi {abs(difference):.3f} segundos mais rapido "
            f"({percent:.1f}% de reducao no tempo)."
        )
        faster = "main"

    return {
        "available": True,
        "mainSeconds": main_seconds,
        "solverSeconds": solver_seconds,
        "differenceSeconds": difference,
        "absoluteDifferenceSeconds": abs(difference),
        "improvementPercent": percent,
        "faster": faster,
        "message": message,
    }


def solver_catalog() -> list[dict[str, str]]:
    return [*SOLVER_ENGINES, *DOCUMENTED_SOLVERS]


def solver_engine_summary(
    engine: dict[str, str],
    main_result: dict[str, Any],
    solver_result: dict[str, Any],
) -> dict[str, Any]:
    comparison = compare_solver_result(main_result, solver_result)
    linear = solver_result.get("linear", {})
    return {
        "id": engine["id"],
        "name": engine["name"],
        "method": engine["method"],
        "status": "ok" if linear.get("ok") else "erro",
        "pA": solver_result.get("pA"),
        "pB": solver_result.get("pB"),
        "support": solver_result.get("support"),
        "confidence": solver_result.get("confidence"),
        "lift": solver_result.get("lift"),
        "lower": linear.get("lower"),
        "upper": linear.get("upper"),
        "variables": linear.get("variables"),
        "constraints": linear.get("constraints"),
        "durationSeconds": solver_result.get("processing", {}).get("durationSeconds"),
        "allMatch": comparison["allMatch"],
        "error": linear.get("error"),
    }


def solver_engine_for_method(method: str) -> dict[str, str]:
    for engine in SOLVER_ENGINES:
        if engine["method"] == method:
            return engine
    valid_methods = ", ".join(engine["method"] for engine in SOLVER_ENGINES)
    raise ValueError(f"Metodo de solver invalido: {method}. Use: {valid_methods}")


@lru_cache(maxsize=48)
def _cached_standalone_solver_result(
    payload_json: str,
    solver_method: str,
    solver_name: str,
) -> dict[str, Any]:
    from scripts import solve_query as standalone_solver

    payload = json.loads(payload_json)
    solver_data = standalone_solver.load_dataset(standalone_solver.DEFAULT_DATASET)
    solver_target = standalone_solver.validate_conditions(
        [payload.get("target", {})],
        solver_data["domains"],
    )[0]
    solver_conditions = standalone_solver.validate_conditions(
        payload.get("conditions", []),
        solver_data["domains"],
    )
    return standalone_solver.compute_query(
        solver_data,
        solver_target,
        solver_conditions,
        solver_method=solver_method,
        solver_name=solver_name,
    )


def _build_solver_comparison_uncached(payload: dict[str, Any]) -> dict[str, Any]:
    main_result = compute_query(payload)
    normalized_payload_json = json.dumps(
        {"target": main_result["target"], "conditions": main_result["conditions"]},
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    engine_results = []
    default_solver_result = None
    for engine in SOLVER_ENGINES:
        # O projeto principal ja executou de fato o HiGHS-IPM acima. Reutilizar
        # esse resultado evita resolver os mesmos dois PLs uma quarta vez sem
        # retirar nenhum dos tres metodos reais da comparacao.
        if engine["method"] == main_result.get("linear", {}).get("solverMethod"):
            solver_result = main_result
        else:
            solver_result = _cached_standalone_solver_result(
                normalized_payload_json,
                engine["method"],
                engine["name"],
            )
        if engine["id"] == "highs":
            default_solver_result = solver_result
        engine_results.append(solver_engine_summary(engine, main_result, solver_result))

    solver_result = default_solver_result or _cached_standalone_solver_result(
        normalized_payload_json,
        "highs",
        "SciPy HiGHS",
    )

    return {
        "ok": True,
        "main": main_result,
        "standaloneSolver": solver_result,
        "solverEngineResults": engine_results,
        "comparison": compare_solver_result(main_result, solver_result),
        "timing": compare_solver_timing(main_result, solver_result),
        "solverCatalog": solver_catalog(),
        "message": "Comparacao executada com 3 metodos reais do HiGHS: highs, highs-ds e highs-ipm, todos usando os mesmos parametros escolhidos na interface.",
    }


@lru_cache(maxsize=16)
def _cached_solver_comparison(payload_json: str) -> dict[str, Any]:
    return _build_solver_comparison_uncached(json.loads(payload_json))


def build_solver_comparison(payload: dict[str, Any]) -> dict[str, Any]:
    # O botao de comparacao e o PDF normalmente recebem a mesma consulta em
    # sequencia. A chave canonica permite que o PDF reutilize os tres resultados
    # ja calculados, em vez de repetir uma operacao pesada no Render gratuito.
    payload_json = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return _cached_solver_comparison(payload_json)


@app.post("/api/query")
def query():
    try:
        result = compute_query(request.get_json(force=True))
    except ValueError as error:
        return jsonify({"ok": False, "error": str(error)}), 400
    except Exception as error:
        return jsonify({"ok": False, "error": f"Erro ao executar consulta: {error}"}), 500
    return jsonify(result)


@app.post("/api/linear-program/full")
def full_linear_program():
    try:
        payload = request.get_json(force=True)
        data = load_dataset()
        target = valid_conditions([payload.get("target", {})], data["domains"])[0]
        base = normalize_base_conditions(
            valid_conditions(payload.get("conditions", []), data["domains"]),
            target,
        )
        lp = solve_linear_interval(data["worlds"], data["rows"], target, base)
        GENERATED_REPORT_DIR.mkdir(parents=True, exist_ok=True)
        FULL_LINEAR_PROGRAM_PATH.write_text(
            full_linear_program_text(data["worlds"], data["rows"], target, base, lp),
            encoding="utf-8",
        )
    except ValueError as error:
        return jsonify({"ok": False, "error": str(error)}), 400
    except Exception as error:
        return jsonify({"ok": False, "error": f"Erro ao gerar modelo numerico auditavel: {error}"}), 500

    return jsonify(
        {
            "ok": True,
            "fileUrl": "/reports/generated/programa_linear_completo.txt",
            "downloadUrl": "/api/linear-program/full/download",
            "message": "Modelo numerico auditavel gerado a partir das mesmas matrizes do solver.",
            "modelDigest": lp.get("modelDigest"),
            "solverVariables": lp.get("solverVariables"),
            "constraints": lp.get("constraints"),
        }
    )


@app.get("/api/linear-program/full/download")
def download_full_linear_program():
    if not FULL_LINEAR_PROGRAM_PATH.exists():
        return jsonify({"ok": False, "error": "Gere o modelo numerico auditavel antes de baixar o TXT."}), 404
    return send_from_directory(
        GENERATED_REPORT_DIR,
        FULL_LINEAR_PROGRAM_PATH.name,
        as_attachment=True,
        download_name="programa_linear_completo.txt",
        mimetype="text/plain; charset=utf-8",
    )


@app.post("/api/solver/compare")
def solver_compare():
    try:
        result = build_solver_comparison(request.get_json(force=True))
    except ValueError as error:
        return jsonify({"ok": False, "error": str(error)}), 400
    except Exception as error:
        return jsonify({"ok": False, "error": f"Solver separado indisponivel: {error}"}), 500

    return jsonify(result)


@app.post("/api/solver/run")
def solver_run():
    """Executa um metodo por requisicao para respeitar o limite do Render."""
    try:
        payload = request.get_json(force=True)
        solver_method = str(payload.get("solverMethod", ""))
        engine = solver_engine_for_method(solver_method)
        query_payload = {
            "target": payload.get("target", {}),
            "conditions": payload.get("conditions", []),
        }
        main_result = compute_query(query_payload)
        normalized_payload_json = json.dumps(
            {"target": main_result["target"], "conditions": main_result["conditions"]},
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        if solver_method == main_result.get("linear", {}).get("solverMethod"):
            solver_result = main_result
        else:
            solver_result = _cached_standalone_solver_result(
                normalized_payload_json,
                engine["method"],
                engine["name"],
            )
        summary = solver_engine_summary(engine, main_result, solver_result)
    except ValueError as error:
        return jsonify({"ok": False, "error": str(error)}), 400
    except Exception as error:
        return jsonify({"ok": False, "error": f"Erro ao executar metodo do solver: {error}"}), 500
    return jsonify({"ok": True, "solverEngineResult": summary})


def format_report_probability(value: float | None) -> str:
    return fmt_probability(value).replace(".", ",")


def format_report_interval_probability(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{clean_probability(value):.6f}".replace(".", ",")


def format_report_number(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.3f}".replace(".", ",")


def format_report_metric(value: float | int | None) -> str:
    if value is None:
        return "-"
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return f"{number:.3f}".replace(".", ",")


def format_report_difference(value: float | None) -> str:
    if value is None:
        return "-"
    if abs(value) <= 1e-12:
        return "0"
    return f"{value:.3e}".replace(".", ",")


def write_query_report(result: dict[str, Any]) -> None:
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import cm
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    except Exception as error:
        raise RuntimeError(f"ReportLab indisponivel para gerar PDF: {error}") from error

    GENERATED_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "TitleCustom",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        alignment=1,
        textColor=colors.HexColor("#111827"),
    )
    body = ParagraphStyle(
        "BodyCustom",
        parent=styles["BodyText"],
        fontSize=10,
        leading=14,
        spaceAfter=6,
    )
    code = ParagraphStyle(
        "CodeCustom",
        parent=styles["Code"],
        fontName="Courier",
        fontSize=8,
        leading=10,
        backColor=colors.HexColor("#f3f4f6"),
        borderPadding=6,
    )

    def table(rows: list[list[str]]) -> Table:
        created = Table(rows, colWidths=[5.5 * cm, 10.1 * cm])
        created.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f766e")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#e5e7eb")),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7faf9")]),
                ]
            )
        )
        return created

    target = event_key([result["target"]])
    conditions = event_key(result["conditions"])
    linear = result["linear"]
    interval = (
        f"{format_report_interval_probability(linear.get('lower'))} <= P(A | B) <= {format_report_interval_probability(linear.get('upper'))}"
        if linear.get("ok")
        else f"Nao resolvido: {linear.get('error', 'motivo nao informado')}"
    )
    missing_rule_text = "Nao se aplica: a regra B -> A nao foi gerada pelo Apriori"
    support_text = (
        format_report_probability(result["support"])
        if result["support"] is not None
        else missing_rule_text
    )
    confidence_text = (
        format_report_probability(result["confidence"])
        if result["confidence"] is not None
        else missing_rule_text
    )
    lift_text = (
        format_report_number(result["lift"])
        if result["lift"] is not None
        else missing_rule_text
    )
    model_digest = str(linear.get("modelDigest", "-"))
    model_digest_text: Any = (
        Paragraph(f"{model_digest[:32]}<br/>{model_digest[32:]}", body)
        if len(model_digest) == 64
        else model_digest
    )
    story = [
        Paragraph("Relatorio da Consulta Atual", title),
        Spacer(1, 0.2 * cm),
        Paragraph(f"Consulta: P({target} | {conditions})", body),
        table(
            [
                ["Metrica", "Valor"],
                ["P(A)", format_report_probability(result["pA"])],
                ["P(B)", format_report_probability(result["pB"])],
                ["P(A e B) empirico (somente auditoria da base)", format_report_probability(result["pAB"])],
                ["Suporte da regra Apriori", support_text],
                ["Confianca da regra Apriori", confidence_text],
                ["Lift descritivo", lift_text],
                ["Instancias", f"{result['countBoth']} / {result['countBase']}"],
                ["Resultado de P(A | B) calculado pelo programa linear", interval],
                ["Mundos observados", str(linear.get("observedWorldVariables", "-"))],
                ["Mundos completados para a consulta", str(linear.get("queryCompletionWorlds", 0))],
                ["Variaveis de mundos", str(linear.get("worldVariables", "-"))],
                ["Variaveis do solver (y_w e t)", str(linear.get("solverVariables", "-"))],
                ["Restricoes do solver", str(linear.get("constraints", "-"))],
                ["SHA-256 do modelo numerico", model_digest_text],
                ["Inicio", result["processing"]["startedAt"]],
                ["Fim", result["processing"]["finishedAt"]],
                ["Duracao", f"{result['processing']['durationSeconds']:.3f} segundos"],
            ]
        ),
        Spacer(1, 0.18 * cm),
        Paragraph(
            (
                "Os valores P(A), P(B) e P(A e B) acima sao frequencias empiricas usadas para auditoria da base; "
                "nao sao o resultado do programa linear. Quando P(A e B) empirico e zero, mundos possiveis de "
                "contagem zero completam a consulta e permitem ao solver calcular um limite superior pequeno."
            ),
            body,
        ),
        Spacer(1, 0.18 * cm),
        Paragraph("Mineracao Apriori e restricoes", styles["Heading2"]),
        Paragraph(
            (
                f"Omega contem {result['aprioriMining']['omegaWorlds']} mundos observados. "
                f"O Apriori encontrou {result['aprioriMining']['frequentItemsets']} itemsets "
                f"frequentes e gerou {result['aprioriMining']['ruleCount']} regras. Suporte e "
                "confianca sao transformados em restricoes lineares; lift e apenas descritivo "
                "e nao representa acuracia."
            ),
            body,
        ),
        Spacer(1, 0.18 * cm),
        Paragraph("Conclusao", styles["Heading2"]),
        Paragraph(result["conclusion"], body),
        Paragraph("Formulacao matematica resumida", styles["Heading2"]),
        Paragraph(
            "A formulacao abaixo e didatica. O TXT auditavel, gerado na interface, "
            "contem os vetores objetivo, A_ub, b_ub, A_eq, b_eq e limites exatos "
            "do mesmo modelo identificado pelo SHA-256 acima.",
            body,
        ),
        Paragraph(result["linearProgram"].replace("\n", "<br/>"), code),
        Paragraph("Justificativa dos intervalos", styles["Heading2"]),
        Paragraph(
            "As probabilidades foram representadas por intervalos para reduzir efeitos de "
            "arredondamento e permitir modelagem consistente das restricoes lineares.",
            body,
        ),
        Paragraph("Referencias bibliograficas", styles["Heading2"]),
        Paragraph(
            "Nilsson, N. J. Probabilistic Logic. Artificial Intelligence, 1986.<br/>"
            "Charnes, A.; Cooper, W. W. Programming with linear fractional functionals. Naval Research Logistics Quarterly, 1962.<br/>"
            "Tessem, B. Interval probability propagation. International Journal of Approximate Reasoning, 1992.<br/>"
            "Agrawal, R.; Srikant, R. Fast algorithms for mining association rules. VLDB, 1994.",
            body,
        ),
    ]

    doc = SimpleDocTemplate(
        str(QUERY_REPORT_PATH),
        pagesize=A4,
        rightMargin=1.8 * cm,
        leftMargin=1.8 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.4 * cm,
        title="Relatorio da Consulta Atual",
    )
    doc.build(story)


def write_solver_comparison_report(result: dict[str, Any]) -> None:
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import cm
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    except Exception as error:
        raise RuntimeError(f"ReportLab indisponivel para gerar PDF: {error}") from error

    GENERATED_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "TitleCustom",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        alignment=1,
        textColor=colors.HexColor("#111827"),
    )
    body = ParagraphStyle(
        "BodyCustom",
        parent=styles["BodyText"],
        fontSize=10,
        leading=14,
        spaceAfter=6,
    )

    labels = {
        "pA": "P(A)",
        "pB": "P(B)",
        "pAB": "P(A e B) usado no PL",
        "support": "Suporte da regra",
        "confidence": "Confianca da regra",
        "lift": "Lift",
        "countBoth": "Casos A e B",
        "countBase": "Casos B",
        "linearLower": "Limite linear inferior",
        "linearUpper": "Limite linear superior",
        "variables": "Variaveis",
        "constraints": "Restricoes",
        "durationSeconds": "Tempo de resolucao (s)",
    }
    comparison = result["comparison"]
    rows = [["Metrica", "Projeto", "Solver separado", "Diferenca", "Status"]]
    for key, item in comparison["metrics"].items():
        rows.append(
            [
                labels.get(key, key),
                format_report_metric(item["main"]),
                format_report_metric(item["solver"]),
                format_report_difference(item["difference"]),
                "Medido" if key == "durationSeconds" else "Igual" if item["match"] else "Diferente",
            ]
        )

    table = Table(rows, colWidths=[4.2 * cm, 2.8 * cm, 3.2 * cm, 2.7 * cm, 2.7 * cm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f766e")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#e5e7eb")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7faf9")]),
            ]
        )
    )

    catalog_rows = [["Solver", "Status no projeto", "Comparacao"]]
    for solver in result.get("solverCatalog", solver_catalog()):
        catalog_rows.append([solver["name"], solver["status"], solver["comparison"]])
    catalog_table = Table(catalog_rows, colWidths=[3.9 * cm, 5.8 * cm, 6.0 * cm])
    catalog_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#111827")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#e5e7eb")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 7.7),
                ("LEADING", (0, 0), (-1, -1), 9.4),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7faf9")]),
            ]
        )
    )

    engine_rows = [["Solver", "Metodo SciPy", "Suporte", "Confianca", "Lift", "Intervalo", "Tempo (s)", "Status"]]
    for engine in result.get("solverEngineResults", []):
        interval = (
            f"{format_report_metric(engine.get('lower'))} - {format_report_metric(engine.get('upper'))}"
            if engine.get("status") == "ok"
            else engine.get("error", "Erro")
        )
        engine_rows.append(
            [
                engine["name"],
                engine.get("method", "-"),
                format_report_metric(engine.get("support")),
                format_report_metric(engine.get("confidence")),
                format_report_metric(engine.get("lift")),
                interval,
                format_report_metric(engine.get("durationSeconds")),
                "Igual ao projeto" if engine.get("allMatch") else "Diferente",
            ]
        )
    engine_table = Table(
        engine_rows,
        colWidths=[3.0 * cm, 2.0 * cm, 1.6 * cm, 1.7 * cm, 1.4 * cm, 2.5 * cm, 1.6 * cm, 1.9 * cm],
    )
    engine_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f766e")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#e5e7eb")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 7.2),
                ("LEADING", (0, 0), (-1, -1), 8.8),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7faf9")]),
            ]
        )
    )

    main_result = result["main"]
    target = event_key([main_result["target"]])
    conditions = event_key(main_result["conditions"])
    status = (
        "Todos os valores comparados coincidem entre o projeto e o solver separado."
        if comparison["allMatch"]
        else "Foram encontradas diferencas entre o projeto e o solver separado."
    )
    timing = result.get("timing", {})
    story = [
        Paragraph("Relatorio Comparativo: Projeto x Solver Separado", title),
        Spacer(1, 0.2 * cm),
        Paragraph(f"Consulta comparada: P({target} | {conditions})", body),
        Paragraph(
            "Os dados de entrada usados neste relatorio sao exatamente os mesmos informados "
            "na interface do projeto: o evento A escolhido na pergunta e as afirmacoes B "
            "selecionadas pelo usuario.",
            body,
        ),
        Paragraph(
            "O projeto principal calcula as probabilidades pela API Flask da interface. "
            "O solver separado executa o modulo scripts/solve_query.py, usando o mesmo dataset, "
            "a mesma categorizacao e o solver HiGHS via scipy.optimize.linprog.",
            body,
        ),
        Paragraph(
            "A programacao linear e resolvida apos a transformacao de Charnes-Cooper, "
            "que converte a razao P(A e B) / P(B) em objetivos lineares de minimizacao e maximizacao.",
            body,
        ),
        Paragraph(
            "Nesta versao, o botao executa tres metodos reais do HiGHS no script separado: "
            "HiGHS automatico, HiGHS Dual Simplex e HiGHS Interior Point. Todos usam exatamente "
            "os mesmos parametros escolhidos pelo usuario na interface. Gurobi, lp_solve e "
            "cuPDLP-C permanecem como referencias tecnicas para benchmark futuro.",
            body,
        ),
        Spacer(1, 0.12 * cm),
        Paragraph("Comparacao entre 3 metodos de solver executados", styles["Heading2"]),
        Paragraph(
            "A tabela abaixo nao e apenas uma lista de referencias: estes tres metodos foram "
            "executados pelo script separado com a mesma consulta da interface.",
            body,
        ),
        engine_table,
        Spacer(1, 0.16 * cm),
        Paragraph("Solvers considerados", styles["Heading2"]),
        catalog_table,
        Spacer(1, 0.16 * cm),
        Paragraph("Comparacao numerica executada", styles["Heading2"]),
        table,
        Spacer(1, 0.18 * cm),
        Paragraph("Conclusao da comparacao", styles["Heading2"]),
        Paragraph(status, body),
        Paragraph("Tempo de execucao", styles["Heading2"]),
        Paragraph(
            (
                f"Projeto principal: {format_report_metric(timing.get('mainSeconds'))} segundos.<br/>"
                f"Solver separado: {format_report_metric(timing.get('solverSeconds'))} segundos.<br/>"
                f"{timing.get('message', 'Tempo de execucao indisponivel para comparacao.')}"
            ),
            body,
        ),
        Paragraph(
            "Essa comparacao evidencia que a interface nao apenas exibe resultados: ela esta alinhada "
            "ao solver independente usado para reproduzir a formulacao matematica da consulta.",
            body,
        ),
    ]

    doc = SimpleDocTemplate(
        str(SOLVER_COMPARISON_REPORT_PATH),
        pagesize=A4,
        rightMargin=1.7 * cm,
        leftMargin=1.7 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.4 * cm,
        title="Relatorio Comparativo do Solver",
    )
    doc.build(story)


@app.post("/api/report/query")
def query_report():
    try:
        result = compute_query(request.get_json(force=True))
        write_query_report(result)
    except ValueError as error:
        return jsonify({"ok": False, "error": str(error)}), 400
    except RuntimeError as error:
        return jsonify({"ok": False, "error": str(error)}), 500
    return jsonify(
        {
            "ok": True,
            "reportUrl": "/reports/generated/relatorio_consulta_atual.pdf",
            "message": "Relatorio gerado com sucesso.",
        }
    )


@app.post("/api/report/solver-comparison")
def solver_comparison_report():
    try:
        result = build_solver_comparison(request.get_json(force=True))
        write_solver_comparison_report(result)
    except ValueError as error:
        return jsonify({"ok": False, "error": str(error)}), 400
    except RuntimeError as error:
        return jsonify({"ok": False, "error": str(error)}), 500
    except Exception as error:
        return jsonify({"ok": False, "error": f"Erro ao comparar solver: {error}"}), 500
    return jsonify(
        {
            "ok": True,
            "reportUrl": "/reports/generated/relatorio_comparacao_solver.pdf",
            "message": "Relatorio comparativo gerado com sucesso.",
            "main": result["main"],
            "standaloneSolver": result["standaloneSolver"],
            "comparison": result["comparison"],
            "timing": result["timing"],
        }
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "1000"))
    app.run(host="0.0.0.0", port=port, debug=True)
