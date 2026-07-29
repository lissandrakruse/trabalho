from __future__ import annotations

import sys
import json
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
from reportlab.graphics.shapes import Drawing, Line, Rect, String


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports" / "relatorio_saida_programacao_linear.pdf"
JSON_OUTPUT = ROOT / "reports" / "saida_calculada_programacao_linear.json"
sys.path.insert(0, str(ROOT))

from app import (  # noqa: E402
    load_dataset,
    linear_program_text,
    probability,
    probability_count,
    solve_linear_interval,
)

INK = colors.HexColor("#111827")
MUTED = colors.HexColor("#374151")
GRID = colors.HexColor("#e5e7eb")
TEAL = colors.HexColor("#0f766e")
TEAL_DARK = colors.HexColor("#176b5b")
SOFT = colors.HexColor("#f7faf9")


def fmt(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.3f}".replace(".", ",")


def add_heading(story: list, text: str, style: ParagraphStyle) -> None:
    story.append(Spacer(1, 0.22 * cm))
    story.append(Paragraph(text, style))
    story.append(Spacer(1, 0.08 * cm))


def styled_table(rows: list[list[str]], col_widths: list[float], header_color=TEAL_DARK) -> Table:
    table = Table(rows, colWidths=col_widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), header_color),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.4, GRID),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.7),
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


def metric_bar_chart(metrics: list[tuple[str, float, colors.Color]]) -> Drawing:
    width = 460
    height = 185
    left = 82
    bottom = 34
    chart_width = 330
    chart_height = 110
    drawing = Drawing(width, height)
    drawing.add(String(width / 2, height - 18, "Grafico das probabilidades calculadas", textAnchor="middle", fontSize=11, fillColor=INK))
    drawing.add(Line(left, bottom, left, bottom + chart_height, strokeColor=INK, strokeWidth=1))
    drawing.add(Line(left, bottom, left + chart_width, bottom, strokeColor=INK, strokeWidth=1))
    for tick in range(0, 6):
        value = tick / 5
        y = bottom + value * chart_height
        drawing.add(Line(left - 3, y, left + chart_width, y, strokeColor=GRID, strokeWidth=0.6))
        drawing.add(String(left - 10, y - 3, f"{value:.1f}", textAnchor="end", fontSize=7.5, fillColor=MUTED))

    bar_width = 42
    gap = 34
    for index, (label, value, color) in enumerate(metrics):
        x = left + 22 + index * (bar_width + gap)
        bar_height = max(1, min(value, 1) * chart_height)
        drawing.add(Rect(x, bottom, bar_width, bar_height, fillColor=color, strokeColor=color))
        drawing.add(String(x + bar_width / 2, bottom + bar_height + 7, f"{value:.3f}", textAnchor="middle", fontSize=8, fillColor=INK))
        drawing.add(String(x + bar_width / 2, bottom - 15, label, textAnchor="middle", fontSize=8, fillColor=MUTED))
    return drawing


def interval_chart(empirical: float | None, lower: float | None, upper: float | None) -> Drawing:
    width = 460
    height = 125
    left = 64
    y = 58
    axis_width = 350
    drawing = Drawing(width, height)
    drawing.add(String(width / 2, height - 18, "Intervalo da consulta condicional pelo solver", textAnchor="middle", fontSize=11, fillColor=INK))
    drawing.add(Line(left, y, left + axis_width, y, strokeColor=INK, strokeWidth=1.2))
    for tick in range(0, 6):
        value = tick / 5
        x = left + value * axis_width
        drawing.add(Line(x, y - 4, x, y + 4, strokeColor=INK, strokeWidth=0.8))
        drawing.add(String(x, y - 18, f"{value:.1f}", textAnchor="middle", fontSize=7.5, fillColor=MUTED))

    if lower is not None and upper is not None:
        x1 = left + max(0, min(lower, 1)) * axis_width
        x2 = left + max(0, min(upper, 1)) * axis_width
        drawing.add(Rect(x1, y - 7, max(2, x2 - x1), 14, fillColor=colors.HexColor("#99f6e4"), strokeColor=TEAL))
        drawing.add(String(x1, y + 16, f"min {lower:.3f}", textAnchor="middle", fontSize=8, fillColor=TEAL_DARK))
        drawing.add(String(x2, y + 16, f"max {upper:.3f}", textAnchor="middle", fontSize=8, fillColor=TEAL_DARK))

    if empirical is not None:
        xe = left + max(0, min(empirical, 1)) * axis_width
        drawing.add(Line(xe, y - 18, xe, y + 22, strokeColor=colors.HexColor("#b91c1c"), strokeWidth=1.4))
        drawing.add(String(xe, y + 31, f"empirico {empirical:.3f}", textAnchor="middle", fontSize=8, fillColor=colors.HexColor("#b91c1c")))
    return drawing


def page_style(canvas, doc) -> None:
    canvas.saveState()
    width, height = A4
    canvas.setStrokeColor(GRID)
    canvas.setLineWidth(0.7)
    canvas.line(doc.leftMargin, height - 1.05 * cm, width - doc.rightMargin, height - 1.05 * cm)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(doc.leftMargin, 0.75 * cm, "Trabalho - Probabilidades e Programacao Linear")
    canvas.drawRightString(width - doc.rightMargin, 0.75 * cm, f"Pagina {doc.page}")
    canvas.restoreState()


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
    lp_lower = lp.get("lower") if lp.get("ok") else None
    lp_upper = lp.get("upper") if lp.get("ok") else None

    JSON_OUTPUT.write_text(
        json.dumps(
            {
                "consulta": "P(label=rice | ph=acido, rainfall=alto)",
                "eventoA": target,
                "condicoesB": conditions,
                "probabilidades": {
                    "pA": p_a,
                    "pB": p_b,
                    "suporte_pAeB": p_ab,
                    "confianca_pAdadoB": confidence,
                    "lift": lift,
                },
                "contagens": {
                    "total": data["total"],
                    "instancias_B": probability_count(rows, conditions),
                    "instancias_A_e_B": probability_count(rows, both),
                    "mundos_observados": len(data["worlds"]),
                },
                "programacaoLinear": lp,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "TitleCustom",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=19,
        leading=23,
        alignment=1,
        textColor=INK,
        spaceAfter=5,
    )
    subtitle = ParagraphStyle(
        "SubtitleCustom",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        alignment=1,
        textColor=MUTED,
        spaceAfter=12,
    )
    h2 = ParagraphStyle(
        "HeadingCustom",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        textColor=TEAL,
        spaceBefore=4,
        spaceAfter=4,
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
        textColor=INK,
        backColor=colors.HexColor("#f3f4f6"),
        borderPadding=6,
    )

    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        rightMargin=1.8 * cm,
        leftMargin=1.8 * cm,
        topMargin=1.55 * cm,
        bottomMargin=1.35 * cm,
        title="Relatorio - Saida e Programacao Linear",
    )

    story: list = []
    story.append(Paragraph("Relatorio da Saida do Sistema", title))
    story.append(
        Paragraph(
            "Consulta probabilistica com dataset categorico, regras de associacao "
            "e resolucao por programacao linear.",
            subtitle,
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

    metadata_table = styled_table(
        [
            ["Item", "Valor"],
            ["Atributos", ", ".join(data["attributes"])],
            ["Numericos categorizados", ", ".join(data["numericAttributes"])],
            ["Categoricos originais", ", ".join(data["categoricalAttributes"]) or "-"],
            ["Mundos observados", str(len(data["worlds"]))],
        ],
        [5 * cm, 10.6 * cm],
    )
    story.append(metadata_table)

    add_heading(story, "3. Algoritmos matematicos usados", h2)
    story.append(
        Paragraph(
            "Abaixo estao os principais algoritmos matematicos implementados no "
            "backend Python. Eles explicam como o dataset vira probabilidades, "
            "restricoes e uma consulta resolvida por programacao linear.",
            body,
        )
    )
    algorithms_table = styled_table(
        [
            ["Etapa", "Formula/algoritmo", "Finalidade"],
            [
                "Discretizacao",
                "atributo numerico <= Q1 -> baixo; <= Q2 -> medio; > Q2 -> alto",
                "Transformar variaveis numericas em categorias para montar mundos possiveis.",
            ],
            [
                "Probabilidade marginal",
                "P(E) = contagem(E) / N",
                "Medir a frequencia de um evento simples, como P(label=rice).",
            ],
            [
                "Probabilidade conjunta",
                "P(A e B) = contagem(A e B) / N",
                "Medir a frequencia de eventos que acontecem ao mesmo tempo.",
            ],
            [
                "Probabilidade condicional",
                "P(A | B) = P(A e B) / P(B)",
                "Responder a pergunta feita pelo usuario na interface.",
            ],
            [
                "Suporte",
                "suporte(B -> A) = P(A e B)",
                "Indicar quanto a regra completa aparece na base.",
            ],
            [
                "Confianca/precisao",
                "confianca(B -> A) = P(A | B)",
                "Indicar a chance de A acontecer quando B acontece.",
            ],
            [
                "Lift",
                "lift(B -> A) = P(A | B) / P(A)",
                "Comparar a regra com a probabilidade marginal de A.",
            ],
            [
                "Intervalo",
                "round(p, 3) - 0,001 <= P(E) <= round(p, 3) + 0,001",
                "Evitar rigidez numerica e representar probabilidade intervalar.",
            ],
            [
                "Variavel linear",
                "x_w >= 0 para cada mundo possivel w; soma(x_w) = 1",
                "Representar uma distribuicao de probabilidade valida.",
            ],
            [
                "Restricao linear",
                "L <= soma(x_w onde E ocorre) <= U",
                "Converter probabilidades extraidas da base em restricoes do LP.",
            ],
            [
                "Charnes-Cooper",
                "P(A e B)/P(B) -> x_w = y_w/t, com P(B) em y = 1",
                "Transformar a consulta condicional fracionaria em problema linear.",
            ],
        ],
        [3.6 * cm, 6.4 * cm, 5.6 * cm],
        header_color=INK,
    )
    story.append(algorithms_table)
    story.append(Spacer(1, 0.12 * cm))
    story.append(
        Paragraph(
            "Pseudocodigo resumido:<br/>"
            "1. Ler CSV e detectar colunas.<br/>"
            "2. Converter atributos numericos para categorias.<br/>"
            "3. Agrupar linhas em mundos possiveis w.<br/>"
            "4. Calcular P(A), P(B), P(A e B), suporte, confianca e lift.<br/>"
            "5. Criar restricoes intervalares para as probabilidades observadas.<br/>"
            "6. Resolver minimo e maximo da consulta condicional com scipy.optimize.linprog.",
            code,
        )
    )

    add_heading(story, "4. Exemplo de pergunta do usuario", h2)
    story.append(
        Paragraph(
            "Na interface, o usuario pode escolher uma pergunta A e uma ou mais "
            "condicoes B. No exemplo abaixo:",
            body,
        )
    )
    story.append(Paragraph("A = label=rice<br/>B = ph=acido, rainfall=alto", code))
    story.append(Paragraph("Consulta: P(label=rice | ph=acido, rainfall=alto)", body))

    add_heading(story, "5. Saida probabilistica", h2)
    metrics_table = styled_table(
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
        [4.1 * cm, 2.7 * cm, 8.8 * cm],
        header_color=INK,
    )
    story.append(metrics_table)
    story.append(Spacer(1, 0.18 * cm))
    story.append(
        metric_bar_chart(
            [
                ("P(A)", p_a, TEAL),
                ("P(B)", p_b, colors.HexColor("#2563eb")),
                ("Suporte", p_ab, colors.HexColor("#7c3aed")),
                ("Confianca", confidence or 0, colors.HexColor("#b45309")),
            ]
        )
    )

    add_heading(story, "6. Grafico do intervalo linear", h2)
    story.append(
        Paragraph(
            "O grafico abaixo compara o valor empirico da confianca com o intervalo "
            "calculado pelo solver. O intervalo aparece porque as probabilidades do "
            "dataset sao usadas como restricoes aproximadas, com arredondamento em "
            "tres casas decimais.",
            body,
        )
    )
    story.append(interval_chart(confidence, lp_lower, lp_upper))

    add_heading(story, "7. Probabilidades lineares consideradas", h2)
    story.append(
        Paragraph(
            "No modelo atual entram automaticamente as probabilidades lineares mais "
            "importantes para responder a consulta escolhida. Elas sao somas de "
            "variaveis x_w, portanto podem ser escritas diretamente como restricoes "
            "lineares.",
            body,
        )
    )
    linear_table = styled_table(
        [
            ["Tipo", "Forma", "Status no sistema"],
            ["Marginal", "P(valor de atributo)", "Incluida para todos os valores de todos os atributos."],
            ["Evento A", "P(A)", "Incluida conforme a pergunta do usuario."],
            ["Condicoes B", "P(B)", "Incluida conforme as afirmacoes do usuario."],
            ["Conjunta da regra", "P(A e B)", "Incluida para calcular suporte e condicional."],
            ["Suporte", "P(A e B)", "Linear direto."],
            ["Condicional", "P(A | B)", "Resolvida por transformacao linear-fracionaria."],
            ["Conjuntas de pares", "P(X=x, Y=y)", "Possivel extensao; nao entra por padrao para evitar explosao combinatoria."],
            ["Conjuntas de trios ou mais", "P(X=x, Y=y, Z=z)", "Possivel extensao; aumenta muito variaveis/restricoes."],
        ],
        [4.2 * cm, 4.8 * cm, 6.6 * cm],
        header_color=TEAL,
    )
    story.append(linear_table)

    add_heading(story, "8. Programacao linear", h2)
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

    add_heading(story, "9. Saida final apresentada ao usuario", h2)
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

    add_heading(story, "10. Arquivo de saida calculada", h2)
    story.append(
        Paragraph(
            "Alem do PDF, o gerador salva um JSON com a consulta, probabilidades, "
            "contagens e resultado do solver. Esse arquivo pode ser usado como "
            "evidencia objetiva da execucao do sistema.",
            body,
        )
    )
    story.append(Paragraph("reports/saida_calculada_programacao_linear.json", code))

    add_heading(story, "11. Relacao com o enunciado", h2)
    story.append(
        Paragraph(
            "A saida comprova que o sistema extrai conhecimento probabilistico "
            "da base, expressa suporte e confianca como probabilidades, transforma "
            "as probabilidades em restricoes lineares, permite perguntas condicionais "
            "pela interface e resolve a consulta com solver de programacao linear.",
            body,
        )
    )

    doc.build(story, onFirstPage=page_style, onLaterPages=page_style)


if __name__ == "__main__":
    build_report()
    print(OUTPUT)
