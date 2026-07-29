from __future__ import annotations

import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports" / "conclusao_resultados_programacao_linear.pdf"
sys.path.insert(0, str(ROOT))

from app import (  # noqa: E402
    conclusion_text,
    load_dataset,
    probability,
    probability_count,
    solve_linear_interval,
)


INK = colors.HexColor("#111827")
MUTED = colors.HexColor("#374151")
GRID = colors.HexColor("#e5e7eb")
TEAL = colors.HexColor("#0f766e")
SOFT = colors.HexColor("#f7faf9")


def fmt(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.3f}".replace(".", ",")


def styled_table(rows: list[list[str]], widths: list[float], header_color=TEAL) -> Table:
    table = Table(rows, colWidths=widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), header_color),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.4, GRID),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.8),
                ("LEADING", (0, 0), (-1, -1), 11),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, SOFT]),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return table


def page_style(canvas, doc) -> None:
    canvas.saveState()
    width, height = A4
    canvas.setStrokeColor(GRID)
    canvas.line(doc.leftMargin, height - 1.05 * cm, width - doc.rightMargin, height - 1.05 * cm)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(doc.leftMargin, 0.75 * cm, "Conclusao - Consulta Probabilistica e Programacao Linear")
    canvas.drawRightString(width - doc.rightMargin, 0.75 * cm, f"Pagina {doc.page}")
    canvas.restoreState()


def build_case(data: dict, target: dict[str, str], conditions: list[dict[str, str]]) -> dict:
    rows = data["rows"]
    both = [*conditions, target]
    p_a = probability(rows, [target])
    p_b = probability(rows, conditions)
    p_ab = probability(rows, both)
    confidence = p_ab / p_b if p_b > 0 else None
    lift = confidence / p_a if confidence is not None and p_a > 0 else None
    count_base = probability_count(rows, conditions)
    count_both = probability_count(rows, both)
    lp = solve_linear_interval(data["worlds"], rows, target, conditions)
    conclusion = conclusion_text(target, conditions, p_ab, confidence, lift, p_b, count_base, count_both, lp)
    return {
        "target": target,
        "conditions": conditions,
        "pA": p_a,
        "pB": p_b,
        "support": p_ab,
        "confidence": confidence,
        "lift": lift,
        "countBase": count_base,
        "countBoth": count_both,
        "linear": lp,
        "conclusion": conclusion,
    }


def build_report() -> None:
    data = load_dataset()
    valid_case = build_case(
        data,
        {"attribute": "label", "value": "rice"},
        [{"attribute": "ph", "value": "acido"}, {"attribute": "rainfall", "value": "alto"}],
    )
    zero_case = build_case(
        data,
        {"attribute": "label", "value": "banana"},
        [
            {"attribute": "N", "value": "baixo"},
            {"attribute": "humidity", "value": "alto"},
            {"attribute": "P", "value": "medio"},
            {"attribute": "rainfall", "value": "alto"},
            {"attribute": "K", "value": "alto"},
        ],
    )

    styles = getSampleStyleSheet()
    title = ParagraphStyle("TitleCustom", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=19, leading=23, alignment=1, textColor=INK)
    subtitle = ParagraphStyle("SubtitleCustom", parent=styles["BodyText"], fontSize=10, leading=14, alignment=1, textColor=MUTED, spaceAfter=12)
    h2 = ParagraphStyle("HeadingCustom", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=13, leading=16, textColor=TEAL)
    body = ParagraphStyle("BodyCustom", parent=styles["BodyText"], fontSize=10, leading=14, spaceAfter=6)
    code = ParagraphStyle("CodeCustom", parent=styles["Code"], fontName="Courier", fontSize=8, leading=10, textColor=INK, backColor=colors.HexColor("#f3f4f6"), borderPadding=6)

    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        rightMargin=1.8 * cm,
        leftMargin=1.8 * cm,
        topMargin=1.55 * cm,
        bottomMargin=1.35 * cm,
        title="Conclusao dos Resultados",
    )

    story: list = []
    story.append(Paragraph("Conclusao dos Resultados", title))
    story.append(Paragraph("Interpretacao da saida probabilistica e do intervalo resolvido por programacao linear.", subtitle))

    story.append(Paragraph("1. Conclusao geral", h2))
    story.append(
        Paragraph(
            "O sistema consegue transformar uma pergunta feita na interface em uma consulta probabilistica "
            "sobre o dataset. A resposta combina a evidencia empirica da base com um intervalo obtido por "
            "programacao linear. Assim, a saida nao mostra apenas um numero: ela explica a forca da regra, "
            "a frequencia do evento, a confianca da associacao e a validade matematica da consulta.",
            body,
        )
    )

    story.append(Paragraph("2. Exemplo com resultado calculavel", h2))
    story.append(Paragraph("Consulta: P(label=rice | ph=acido, rainfall=alto)", code))
    story.append(
        styled_table(
            [
                ["Metrica", "Valor"],
                ["P(A)", fmt(valid_case["pA"])],
                ["P(B)", fmt(valid_case["pB"])],
                ["Suporte P(A e B)", fmt(valid_case["support"])],
                ["Confianca P(A | B)", fmt(valid_case["confidence"])],
                ["Lift", fmt(valid_case["lift"])],
                ["Instancias", f"{valid_case['countBoth']} / {valid_case['countBase']}"],
                ["Intervalo linear", f"{fmt(valid_case['linear'].get('lower'))} <= P(A | B) <= {fmt(valid_case['linear'].get('upper'))}"],
            ],
            [6 * cm, 9.6 * cm],
            header_color=INK,
        )
    )
    story.append(Spacer(1, 0.12 * cm))
    story.append(Paragraph("<b>Conclusao automatica:</b>", body))
    story.append(Paragraph(valid_case["conclusion"], body))

    story.append(Paragraph("3. Exemplo com P(B)=0", h2))
    story.append(
        Paragraph(
            "Quando nenhuma linha do dataset satisfaz todas as afirmacoes escolhidas, P(B)=0. "
            "Nesse caso, a probabilidade condicional P(A | B) nao pode ser calculada, porque "
            "a formula teria denominador zero.",
            body,
        )
    )
    story.append(
        Paragraph(
            "Consulta: P(label=banana | N=baixo, humidity=alto, P=medio, rainfall=alto, K=alto)",
            code,
        )
    )
    story.append(
        styled_table(
            [
                ["Metrica", "Valor"],
                ["P(B)", fmt(zero_case["pB"])],
                ["Suporte", fmt(zero_case["support"])],
                ["Confianca", fmt(zero_case["confidence"])],
                ["Instancias", f"{zero_case['countBoth']} / {zero_case['countBase']}"],
                ["Resultado linear", zero_case["linear"].get("error", "-")],
            ],
            [6 * cm, 9.6 * cm],
            header_color=INK,
        )
    )
    story.append(Spacer(1, 0.12 * cm))
    story.append(Paragraph("<b>Conclusao automatica:</b>", body))
    story.append(Paragraph(zero_case["conclusion"], body))

    story.append(Paragraph("4. Como interpretar a conclusao", h2))
    story.append(
        styled_table(
            [
                ["Elemento", "Interpretacao"],
                ["Suporte", "Mostra a frequencia da regra completa no dataset."],
                ["Confianca", "Mostra a probabilidade de A acontecer quando B acontece."],
                ["Lift > 1", "Indica associacao positiva entre B e A."],
                ["Lift proximo de 1", "Indica associacao fraca ou proxima da independencia."],
                ["Lift < 1", "Indica que A fica menos provavel quando B ocorre."],
                ["Intervalo linear", "Mostra limites minimo e maximo respeitando as restricoes probabilisticas."],
                ["P(B)=0", "A consulta nao tem evidencia empirica suficiente e nao deve ser interpretada como probabilidade valida."],
            ],
            [4.8 * cm, 10.8 * cm],
        )
    )

    story.append(Paragraph("5. Conclusao para apresentar", h2))
    story.append(
        Paragraph(
            "Conclui-se que o sistema atende ao objetivo do trabalho porque permite formular perguntas "
            "condicionais sobre uma base categorizada, extrai probabilidades e regras de associacao, "
            "transforma essas informacoes em restricoes lineares e resolve a consulta com solver. "
            "A conclusao automatica torna a saida compreensivel para o usuario, distinguindo casos "
            "com evidencia suficiente de casos em que P(B)=0 e a consulta nao e matematicamente definida.",
            body,
        )
    )

    doc.build(story, onFirstPage=page_style, onLaterPages=page_style)


if __name__ == "__main__":
    build_report()
    print(OUTPUT)
