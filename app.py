from __future__ import annotations

import csv
import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request, send_from_directory


ROOT = Path(__file__).resolve().parent
DATASET_PATH = ROOT / "data" / "Crop_recommendation.csv"
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


@lru_cache(maxsize=1)
def load_dataset() -> dict[str, Any]:
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
    if not conditions:
        return 1.0
    return sum(1 for row in rows if matches(row, conditions)) / len(rows)


def probability_count(rows: list[dict[str, str]], conditions: list[dict[str, str]]) -> int:
    if not conditions:
        return len(rows)
    return sum(1 for row in rows if matches(row, conditions))


def rounded_interval(value: float, width: float = 0.001) -> tuple[float, float]:
    rounded = round(value, 3)
    return max(0.0, rounded - width), min(1.0, rounded + width)


def world_mask(worlds: list[dict[str, Any]], conditions: list[dict[str, str]]) -> list[float]:
    return [1.0 if matches(world["values"], conditions) else 0.0 for world in worlds]


def solve_linear_interval(
    worlds: list[dict[str, Any]],
    rows: list[dict[str, str]],
    target: dict[str, str],
    base: list[dict[str, str]],
) -> dict[str, Any]:
    try:
        from scipy.optimize import linprog
    except Exception as error:
        return {"ok": False, "error": f"scipy indisponivel: {error}"}

    n = len(worlds)
    a_ub: list[list[float]] = []
    b_ub: list[float] = []

    def add_interval(mask: list[float], value: float) -> None:
        lower, upper = rounded_interval(value)
        a_ub.append(mask)
        b_ub.append(upper)
        a_ub.append([-item for item in mask])
        b_ub.append(-lower)

    # Marginais de todos os valores observados: conhecimento probabilistico da base.
    attributes = list(rows[0].keys())
    for attribute in attributes:
        for value in sorted({row[attribute] for row in rows}):
            conditions = [{"attribute": attribute, "value": value}]
            add_interval(world_mask(worlds, conditions), probability(rows, conditions))

    both = [*base, target]
    add_interval(world_mask(worlds, [target]), probability(rows, [target]))
    add_interval(world_mask(worlds, base), probability(rows, base))
    add_interval(world_mask(worlds, both), probability(rows, both))

    denominator_mask = world_mask(worlds, base)
    numerator_mask = world_mask(worlds, both)

    # Charnes-Cooper: x = y / t. The conditional objective becomes linear
    # because P(B) is fixed as denominator_mask . y = 1.
    transformed_a_ub = []
    transformed_b_ub = []
    for row, limit in zip(a_ub, b_ub):
        transformed_a_ub.append([*row, -limit])
        transformed_b_ub.append(0.0)
    transformed_a_eq = [
        [*([1.0] * n), -1.0],
        [*denominator_mask, 0.0],
    ]
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
            method="highs",
        )

    lower_result = optimize([*numerator_mask, 0.0])
    upper_result = optimize([-item for item in [*numerator_mask, 0.0]])
    if not lower_result.success or not upper_result.success:
        return {
            "ok": False,
            "error": lower_result.message if not lower_result.success else upper_result.message,
        }

    lower_hi = float(lower_result.fun)
    upper_lo = float(-upper_result.fun)

    return {
        "ok": True,
        "lower": lower_hi,
        "upper": upper_lo,
        "variables": n,
        "constraints": len(transformed_a_ub) + len(transformed_a_eq),
        "solver": "scipy.optimize.linprog highs",
    }


def linear_program_text(
    target: dict[str, str],
    base: list[dict[str, str]],
    p_a: float,
    p_b: float,
    p_ab: float,
    lp: dict[str, Any],
) -> str:
    i_a = rounded_interval(p_a)
    i_b = rounded_interval(p_b)
    i_ab = rounded_interval(p_ab)
    numerator = f"soma(x_w onde {event_key([*base, target])})"
    denominator = f"soma(x_w onde {event_key(base)})"
    lines = [
        "Variaveis:",
        "  x_w >= 0 para cada mundo possivel w da base categorizada",
        "",
        "Restricao de normalizacao:",
        "  soma(x_w) = 1",
        "",
        "Restricoes extraidas da base:",
        f"  {i_a[0]:.3f} <= P(A) = soma(x_w onde {event_key([target])}) <= {i_a[1]:.3f}",
        f"  {i_b[0]:.3f} <= P(B) = {denominator} <= {i_b[1]:.3f}",
        f"  {i_ab[0]:.3f} <= P(A e B) = {numerator} <= {i_ab[1]:.3f}",
        "",
        "Consulta:",
        "  P(A | B) = P(A e B) / P(B)",
        "",
        "Resolucao linear:",
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
                f"Intervalo retornado: {lp['lower']:.3f} <= P(A | B) <= {lp['upper']:.3f}",
            ]
        )
    else:
        lines.extend(["", f"Solver indisponivel: {lp.get('error', 'erro desconhecido')}"])
    return "\n".join(lines)


@app.get("/")
def home():
    return send_from_directory(ROOT, "index.html")


@app.get("/healthz")
def healthz():
    return jsonify({"ok": True})


@app.get("/<path:path>")
def static_files(path: str):
    return send_from_directory(ROOT, path)


@app.get("/api/metadata")
def metadata():
    data = load_dataset()
    return jsonify(
        {
            "attributes": data["attributes"],
            "numericAttributes": data["numericAttributes"],
            "categoricalAttributes": data["categoricalAttributes"],
            "total": data["total"],
            "domains": data["domains"],
            "labels": data["labels"],
            "thresholds": data["thresholds"],
        }
    )


@app.post("/api/query")
def query():
    try:
        payload = request.get_json(force=True)
        data = load_dataset()
        base = valid_conditions(payload.get("conditions", []), data["domains"])
        target = valid_conditions([payload.get("target", {})], data["domains"])[0]
    except ValueError as error:
        return jsonify({"ok": False, "error": str(error)}), 400

    rows = data["rows"]
    both = [*base, target]
    p_a = probability(rows, [target])
    p_b = probability(rows, base)
    p_ab = probability(rows, both)
    confidence = p_ab / p_b if p_b > 0 else None
    lift = confidence / p_a if confidence is not None and p_a > 0 else None
    lp = solve_linear_interval(data["worlds"], rows, target, base)

    return jsonify(
        {
            "ok": True,
            "target": target,
            "conditions": base,
            "support": p_ab,
            "confidence": confidence,
            "lift": lift,
            "pA": p_a,
            "pB": p_b,
            "countBoth": probability_count(rows, both),
            "countBase": probability_count(rows, base),
            "total": data["total"],
            "linear": lp,
            "linearProgram": linear_program_text(target, base, p_a, p_b, p_ab, lp),
        }
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "1000"))
    app.run(host="0.0.0.0", port=port, debug=True)
