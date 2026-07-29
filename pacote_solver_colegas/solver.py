from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = ROOT / "data" / "Crop_recommendation.csv"
DEFAULT_OUTPUT = ROOT / "results" / "solver_resultado_padrao.json"
DEFAULT_TARGET = {"attribute": "label", "value": "rice"}
DEFAULT_CONDITIONS = [
    {"attribute": "ph", "value": "acido"},
    {"attribute": "rainfall", "value": "alto"},
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
    return {
        "attributes": attributes,
        "numericAttributes": numeric_attributes,
        "categoricalAttributes": categorical_attributes,
        "rows": rows,
        "worlds": worlds,
        "domains": domains,
        "thresholds": thresholds,
        "total": len(rows),
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


def rounded_interval(value: float, width: float = 0.001) -> tuple[float, float]:
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


def event_key(conditions: list[dict[str, str]]) -> str:
    if not conditions:
        return "verdadeiro"
    return ", ".join(f"{item['attribute']}={item['value']}" for item in conditions)


def solve_linear_interval(
    data: dict[str, Any],
    target: dict[str, str],
    base: list[dict[str, str]],
) -> dict[str, Any]:
    rows = data["rows"]
    worlds = data["worlds"]
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

    for attribute in data["attributes"]:
        for value in data["domains"][attribute]:
            add_interval(
                world_mask(worlds, [{"attribute": attribute, "value": value}]),
                probability(rows, [{"attribute": attribute, "value": value}]),
            )

    both = [*base, target]
    add_interval(world_mask(worlds, [target]), probability(rows, [target]))
    add_interval(world_mask(worlds, base), probability(rows, base))
    add_interval(world_mask(worlds, both), probability(rows, both))

    denominator_mask = world_mask(worlds, base)
    numerator_mask = world_mask(worlds, both)

    # Charnes-Cooper transforma P(A e B) / P(B) em objetivos lineares.
    transformed_a_ub = [[*row, -limit] for row, limit in zip(a_ub, b_ub)]
    transformed_b_ub = [0.0 for _ in transformed_a_ub]
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

    return {
        "ok": True,
        "lower": clean_probability(float(lower_result.fun)),
        "upper": clean_probability(float(-upper_result.fun)),
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

    confidence_label = "baixa"
    if confidence is not None and confidence >= 0.7:
        confidence_label = "alta"
    elif confidence is not None and confidence >= 0.3:
        confidence_label = "moderada"

    if lift is None:
        lift_sentence = "O lift nao foi calculado."
    elif lift > 1.2:
        lift_sentence = f"O lift de {lift:.3f} indica associacao positiva."
    elif lift < 0.8:
        lift_sentence = f"O lift de {lift:.3f} indica associacao negativa."
    else:
        lift_sentence = f"O lift de {lift:.3f} indica associacao fraca ou proxima da independencia."

    interval_sentence = ""
    if lp.get("ok"):
        interval_sentence = (
            f" Pelo modelo linear, P(A | B) fica entre {fmt_probability(lp['lower'])} "
            f"e {fmt_probability(lp['upper'])}."
        )

    return (
        f"Para a regra {base_label} -> {target_label}, foram encontrados {count_both} "
        f"casos favoraveis dentro de {count_base} casos que satisfazem B. A confianca "
        f"empirica e {confidence:.3f}, considerada {confidence_label}, e o suporte e "
        f"{support:.3f}. {lift_sentence}{interval_sentence}"
    )


def compute_query(
    data: dict[str, Any],
    target: dict[str, str],
    conditions: list[dict[str, str]],
) -> dict[str, Any]:
    started_at = datetime.now(timezone.utc)
    started_perf = time.perf_counter()
    rows = data["rows"]
    both = [*conditions, target]
    p_a = probability(rows, [target])
    p_b = probability(rows, conditions)
    p_ab = probability(rows, both)
    confidence = p_ab / p_b if p_b > 0 else None
    lift = confidence / p_a if confidence is not None and p_a > 0 else None
    count_both = probability_count(rows, both)
    count_base = probability_count(rows, conditions)
    lp = solve_linear_interval(data, target, conditions)
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
        "support": p_ab,
        "confidence": confidence,
        "lift": lift,
        "pA": p_a,
        "pB": p_b,
        "countBoth": count_both,
        "countBase": count_base,
        "total": data["total"],
        "linear": lp,
        "linearProgram": linear_program_text(target, conditions, p_a, p_b, p_ab, lp),
        "conclusion": conclusion_text(target, conditions, p_ab, confidence, lift, p_b, count_base, count_both, lp),
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
    parser.add_argument("--show-domains", action="store_true", help="Mostra atributos e valores validos.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if len(sys.argv) == 1:
        args.target = DEFAULT_TARGET
        args.condition = DEFAULT_CONDITIONS
        args.output = DEFAULT_OUTPUT

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
        result = compute_query(data, target, conditions)
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
