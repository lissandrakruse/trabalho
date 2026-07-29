from __future__ import annotations

import csv
import math
import os
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request, send_from_directory


ROOT = Path(__file__).resolve().parent
DATASET_PATH = ROOT / "data" / "Crop_recommendation.csv"
GENERATED_REPORT_DIR = ROOT / "reports" / "generated"
QUERY_REPORT_PATH = GENERATED_REPORT_DIR / "relatorio_consulta_atual.pdf"
SOLVER_COMPARISON_REPORT_PATH = GENERATED_REPORT_DIR / "relatorio_comparacao_solver.pdf"
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
    return [1.0 if matches(world["values"], conditions) else 0.0 for world in worlds]


def solve_linear_interval(
    worlds: list[dict[str, Any]],
    rows: list[dict[str, str]],
    target: dict[str, str],
    base: list[dict[str, str]],
) -> dict[str, Any]:
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

    lower_hi = clean_probability(float(lower_result.fun))
    upper_lo = clean_probability(float(-upper_result.fun))

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

    confidence_label = "baixa"
    if confidence is not None and confidence >= 0.7:
        confidence_label = "alta"
    elif confidence is not None and confidence >= 0.3:
        confidence_label = "moderada"

    lift_sentence = "O lift nao foi calculado."
    if lift is not None:
        if lift > 1.2:
            lift_sentence = (
                f"O lift de {lift:.3f} indica associacao positiva: o evento A fica mais provavel "
                "quando as condicoes B ocorrem."
            )
        elif lift < 0.8:
            lift_sentence = (
                f"O lift de {lift:.3f} indica associacao negativa: o evento A fica menos provavel "
                "quando as condicoes B ocorrem."
            )
        else:
            lift_sentence = (
                f"O lift de {lift:.3f} indica associacao fraca ou proxima da independencia."
            )

    interval_sentence = ""
    if lp.get("ok"):
        interval_sentence = (
            f" Pelo modelo linear intervalar, P(A | B) fica entre {fmt_probability(lp['lower'])} "
            f"e {fmt_probability(lp['upper'])}."
        )

    return (
        f"Para a regra {base_label} -> {target_label}, foram encontrados {count_both} "
        f"casos favoraveis dentro de {count_base} casos que satisfazem B. A confianca "
        f"empirica e {confidence:.3f}, considerada {confidence_label}, e o suporte e "
        f"{support:.3f}. {lift_sentence}{interval_sentence}"
    )


@app.get("/")
def home():
    return send_from_directory(ROOT, "index.html")


@app.get("/styles.css")
def styles():
    return send_from_directory(ROOT, "styles.css")


@app.get("/script.js")
def scripts():
    return send_from_directory(ROOT, "script.js")


@app.get("/healthz")
def healthz():
    return jsonify({"ok": True})


@app.get("/<path:path>")
def static_files(path: str):
    if not path.startswith("reports/"):
        return jsonify({"ok": False, "error": "Arquivo nao encontrado."}), 404
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


def compute_query(payload: dict[str, Any]) -> dict[str, Any]:
    started_at = datetime.now(timezone.utc)
    started_perf = time.perf_counter()
    data = load_dataset()
    base = valid_conditions(payload.get("conditions", []), data["domains"])
    target = valid_conditions([payload.get("target", {})], data["domains"])[0]

    rows = data["rows"]
    both = [*base, target]
    p_a = probability(rows, [target])
    p_b = probability(rows, base)
    p_ab = probability(rows, both)
    confidence = p_ab / p_b if p_b > 0 else None
    lift = confidence / p_a if confidence is not None and p_a > 0 else None
    count_both = probability_count(rows, both)
    count_base = probability_count(rows, base)
    lp = solve_linear_interval(data["worlds"], rows, target, base)
    finished_at = datetime.now(timezone.utc)
    duration_seconds = time.perf_counter() - started_perf
    conclusion = conclusion_text(target, base, p_ab, confidence, lift, p_b, count_base, count_both, lp)

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
        "support": p_ab,
        "confidence": confidence,
        "lift": lift,
        "pA": p_a,
        "pB": p_b,
        "countBoth": count_both,
        "countBase": count_base,
        "total": data["total"],
        "linear": lp,
        "linearProgram": linear_program_text(target, base, p_a, p_b, p_ab, lp),
        "conclusion": conclusion,
    }


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
    fields = ["pA", "pB", "support", "confidence", "lift", "countBoth", "countBase"]
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


def build_solver_comparison(payload: dict[str, Any]) -> dict[str, Any]:
    main_result = compute_query(payload)

    from scripts import solve_query as standalone_solver

    solver_data = standalone_solver.load_dataset(standalone_solver.DEFAULT_DATASET)
    solver_target = standalone_solver.validate_conditions([main_result["target"]], solver_data["domains"])[0]
    solver_conditions = standalone_solver.validate_conditions(main_result["conditions"], solver_data["domains"])
    solver_result = standalone_solver.compute_query(solver_data, solver_target, solver_conditions)

    return {
        "ok": True,
        "main": main_result,
        "standaloneSolver": solver_result,
        "comparison": compare_solver_result(main_result, solver_result),
        "message": "Os mesmos dados da consulta foram resolvidos pelo solver separado scripts/solve_query.py e comparados com o projeto.",
    }


@app.post("/api/query")
def query():
    try:
        result = compute_query(request.get_json(force=True))
    except ValueError as error:
        return jsonify({"ok": False, "error": str(error)}), 400
    return jsonify(result)


@app.post("/api/solver/compare")
def solver_compare():
    try:
        result = build_solver_comparison(request.get_json(force=True))
    except ValueError as error:
        return jsonify({"ok": False, "error": str(error)}), 400
    except Exception as error:
        return jsonify({"ok": False, "error": f"Solver separado indisponivel: {error}"}), 500

    return jsonify(result)


def format_report_probability(value: float | None) -> str:
    return fmt_probability(value).replace(".", ",")


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
        f"{format_report_probability(linear.get('lower'))} <= P(A | B) <= {format_report_probability(linear.get('upper'))}"
        if linear.get("ok")
        else linear.get("error", "Nao calculado")
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
                ["Suporte P(A e B)", format_report_probability(result["support"])],
                ["Confianca P(A | B)", format_report_probability(result["confidence"])],
                ["Lift", format_report_number(result["lift"])],
                ["Instancias", f"{result['countBoth']} / {result['countBase']}"],
                ["Intervalo linear", interval],
                ["Inicio", result["processing"]["startedAt"]],
                ["Fim", result["processing"]["finishedAt"]],
                ["Duracao", f"{result['processing']['durationSeconds']:.3f} segundos"],
            ]
        ),
        Spacer(1, 0.18 * cm),
        Paragraph("Conclusao", styles["Heading2"]),
        Paragraph(result["conclusion"], body),
        Paragraph("Programa linear", styles["Heading2"]),
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
            "Artigos e materiais disponibilizados pelo professor no Classroom.",
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
        "support": "P(A e B)",
        "confidence": "Confianca P(A | B)",
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

    main_result = result["main"]
    target = event_key([main_result["target"]])
    conditions = event_key(main_result["conditions"])
    status = (
        "Todos os valores comparados coincidem entre o projeto e o solver separado."
        if comparison["allMatch"]
        else "Foram encontradas diferencas entre o projeto e o solver separado."
    )
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
        Spacer(1, 0.16 * cm),
        table,
        Spacer(1, 0.18 * cm),
        Paragraph("Conclusao da comparacao", styles["Heading2"]),
        Paragraph(status, body),
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
        }
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "1000"))
    app.run(host="0.0.0.0", port=port, debug=True)
