from __future__ import annotations

import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports" / "relatorio_saida_programacao_linear.pdf"
sys.path.insert(0, str(ROOT))

from app import (  # noqa: E402
    load_dataset,
    linear_program_text,
    probability,
    probability_count,
    solve_linear_interval,
)


def fmt(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.3f}".replace(".", ",")


def add_heading(story: list, text: str, style: ParagraphStyle) -> None:
    story.append(Spacer(1, 0.18 * cm))
    story.append(Paragraph(text, style))
    story.append(Spacer(1, 0.08 * cm))


def build_report() -> None:
    data = load_dataset()
    rows = data["rows"]
    target = {"attribute": "label", "value": "rice"}
    conditions = [
        {"attribute": "ph", "value": "acido"},
        {"attribute": "rainfall", "value": "alto"},
    ]
    both = [*conditions, target]

    p_a = probability(rows, [target])
    p_b = probability(rows, conditions)
    p_ab = probability(rows, both)
    confidence = p_ab / p_b if p_b > 0 else None
    lift = confidence / p_a if confidence is not None and p_a > 0 else None
    lp = solve_linear_interval(data["worlds"], rows, target, conditions)
    lp_text = linear_program_text(target, conditions, p_a, p_b, p_ab, lp)

    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "TitleCustom",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#17352f"),
        spaceAfter=10,
    )
    h2 = ParagraphStyle(
        "HeadingCustom",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#176b5b"),
    )
    body = ParagraphStyle(
        "BodyCustom",
        parent=styles["BodyText"],
        fontName="Helvetica",
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
        backColor=colors.HexColor("#f2f5f4"),
        borderPadding=6,
    )

    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        rightMargin=1.6 * cm,
        leftMargin=1.6 * cm,
        topMargin=1.4 * cm,
        bottomMargin=1.4 * cm,
        title="Relatorio - Saida e Programacao Linear",
    )

    story: list = []
    story.append(Paragraph("Relatorio da Saida do Sistema", title))
    story.append(
        Paragraph(
            "Consulta probabilistica com dataset categorico, regras de associacao "
            "e resolucao por programacao linear.",
            body,
        )
    )

    add_heading(story, "1. Objetivo", h2)
    story.append(
        Paragraph(
            "O sistema permite que o usuario escolha afirmacoes na interface e formule "
            "uma pergunta condicional. A aplicacao Python le o dataset, extrai "
            "probabilidades, monta restricoes lineares e usa um solver para obter um "
            "intervalo de resposta para P(A | B).",
            body,
        )
    )

    add_heading(story, "2. Dataset e dinamica da interface", h2)
    story.append(
        Paragraph(
            f"O dataset carregado possui {data['total']} registros. O sistema detecta "
            "automaticamente as colunas do CSV, identifica atributos numericos, converte "
            "esses atributos em categorias e envia os dominios para o frontend. Assim, "
            "os campos da interface sao montados dinamicamente a partir da propria base.",
            body,
        )
    )

    metadata_table = Table(
        [
            ["Item", "Valor"],
            ["Atributos", ", ".join(data["attributes"])],
            ["Numericos categorizados", ", ".join(data["numericAttributes"])],
            ["Categoricos originais", ", ".join(data["categoricalAttributes"]) or "-"],
            ["Mundos observados", str(len(data["worlds"]))],
        ],
        colWidths=[5 * cm, 11 * cm],
    )
    metadata_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#176b5b")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#b9c7c3")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7faf9")]),
            ]
        )
    )
    story.append(metadata_table)

    add_heading(story, "3. Exemplo de pergunta do usuario", h2)
    story.append(
        Paragraph(
            "Na interface, o usuario pode escolher uma pergunta A e uma ou mais "
            "condicoes B. No exemplo abaixo:",
            body,
        )
    )
    story.append(Paragraph("A = label=rice<br/>B = ph=acido, rainfall=alto", code))
    story.append(Paragraph("Consulta: P(label=rice | ph=acido, rainfall=alto)", body))

    add_heading(story, "4. Saida probabilistica", h2)
    metrics_table = Table(
        [
            ["Metrica", "Valor", "Interpretacao"],
            ["P(A)", fmt(p_a), "Probabilidade marginal da cultura rice."],
            ["P(B)", fmt(p_b), "Probabilidade das condicoes escolhidas."],
            ["Suporte P(A e B)", fmt(p_ab), "Frequencia da regra completa na base."],
            ["Confianca P(A | B)", fmt(confidence), "Probabilidade de A quando B ocorre."],
            ["Lift", fmt(lift), "Forca da regra comparada com P(A)."],
            [
                "Instancias",
                f"{probability_count(rows, both)} / {probability_count(rows, conditions)}",
                "Casos que satisfazem A e B sobre casos que satisfazem B.",
            ],
        ],
        colWidths=[4.2 * cm, 3 * cm, 8.8 * cm],
    )
    metrics_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17352f")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#b9c7c3")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7faf9")]),
            ]
        )
    )
    story.append(metrics_table)

    add_heading(story, "5. Programacao linear", h2)
    story.append(
        Paragraph(
            "Cada mundo possivel w recebe uma variavel x_w, que representa a "
            "probabilidade daquela combinacao categorizada. As probabilidades extraidas "
            "do dataset sao convertidas em restricoes intervalares com tres casas "
            "decimais. A consulta condicional e resolvida como um problema "
            "linear-fracionario transformado em programacao linear.",
            body,
        )
    )
    story.append(Paragraph(lp_text.replace("\n", "<br/>"), code))

    add_heading(story, "6. Saida final apresentada ao usuario", h2)
    if lp.get("ok"):
        interval_text = f"{fmt(lp['lower'])} <= P(A | B) <= {fmt(lp['upper'])}"
    else:
        interval_text = "Solver indisponivel"
    story.append(
        Paragraph(
            "A interface apresenta o valor empirico da confianca, as metricas de regra "
            "de associacao e o intervalo obtido pelo solver. Para o exemplo:",
            body,
        )
    )
    story.append(
        Paragraph(
            f"P(label=rice | ph=acido, rainfall=alto) = {fmt(confidence)}<br/>"
            f"Intervalo linear: {interval_text}<br/>"
            f"Solver: {lp.get('solver', '-')}",
            code,
        )
    )

    add_heading(story, "7. Relacao com o enunciado", h2)
    story.append(
        Paragraph(
            "A saida comprova que o sistema extrai conhecimento probabilistico "
            "da base, expressa suporte e confianca como probabilidades, transforma "
            "as probabilidades em restricoes lineares, permite perguntas condicionais "
            "pela interface e resolve a consulta com solver de programacao linear.",
            body,
        )
    )

    doc.build(story)


if __name__ == "__main__":
    build_report()
    print(OUTPUT)
