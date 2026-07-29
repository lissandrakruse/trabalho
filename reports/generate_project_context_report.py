from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports" / "contexto_completo_projeto.pdf"

INK = colors.HexColor("#111827")
MUTED = colors.HexColor("#374151")
GRID = colors.HexColor("#e5e7eb")
TEAL = colors.HexColor("#0f766e")
SOFT = colors.HexColor("#f7faf9")


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
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("LEADING", (0, 0), (-1, -1), 10.8),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, SOFT]),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
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
    canvas.drawString(doc.leftMargin, 0.75 * cm, "Contexto completo - Projeto de Probabilidades e Programacao Linear")
    canvas.drawRightString(width - doc.rightMargin, 0.75 * cm, f"Pagina {doc.page}")
    canvas.restoreState()


def heading(story: list, text: str, style: ParagraphStyle) -> None:
    story.append(Spacer(1, 0.18 * cm))
    story.append(Paragraph(text, style))
    story.append(Spacer(1, 0.08 * cm))


def build_report() -> None:
    styles = getSampleStyleSheet()
    title = ParagraphStyle("TitleCustom", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=19, leading=23, alignment=1, textColor=INK)
    subtitle = ParagraphStyle("SubtitleCustom", parent=styles["BodyText"], fontSize=10, leading=14, alignment=1, textColor=MUTED, spaceAfter=12)
    h2 = ParagraphStyle("HeadingCustom", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=13, leading=16, textColor=TEAL)
    body = ParagraphStyle("BodyCustom", parent=styles["BodyText"], fontSize=9.8, leading=13.5, spaceAfter=6)
    code = ParagraphStyle("CodeCustom", parent=styles["Code"], fontName="Courier", fontSize=8, leading=10, textColor=INK, backColor=colors.HexColor("#f3f4f6"), borderPadding=6)

    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        rightMargin=1.8 * cm,
        leftMargin=1.8 * cm,
        topMargin=1.55 * cm,
        bottomMargin=1.35 * cm,
        title="Contexto Completo do Projeto",
    )

    story: list = []
    story.append(Paragraph("Contexto Completo do Projeto", title))
    story.append(Paragraph("Sistema dinamico para probabilidades, regras de associacao e programacao linear.", subtitle))

    heading(story, "1. Contexto do trabalho", h2)
    story.append(
        Paragraph(
            "O projeto foi desenvolvido a partir do enunciado do professor Jose Carlos Ferreira da Rocha. "
            "A proposta era usar uma base com variaveis categoricas para extrair conhecimento probabilistico, "
            "representar esse conhecimento como restricoes de um programa linear e permitir que o usuario "
            "fizesse perguntas condicionais do tipo P(A | B).",
            body,
        )
    )

    heading(story, "2. Objetivo implementado", h2)
    story.append(
        Paragraph(
            "Foi construida uma aplicacao web em Python/Flask com frontend dinamico. O usuario escolhe "
            "afirmacoes e uma pergunta na interface. O backend calcula probabilidades, monta o modelo linear, "
            "resolve com solver e retorna metricas, intervalo linear, tempo de processamento e conclusao automatica.",
            body,
        )
    )

    heading(story, "3. Dataset usado", h2)
    story.append(
        styled_table(
            [
                ["Item", "Descricao"],
                ["Arquivo", "data/Crop_recommendation.csv"],
                ["Origem conceitual", "Dataset de recomendacao de culturas agricolas usado como base de solo/cultura."],
                ["Colunas", "N, P, K, temperature, humidity, ph, rainfall, label"],
                ["Tratamento", "Colunas numericas sao categorizadas automaticamente; label permanece como classe/cultura."],
                ["Dinamismo", "O sistema detecta as colunas do CSV e monta a interface com os dominios reais da base."],
            ],
            [4.2 * cm, 11.4 * cm],
            header_color=INK,
        )
    )

    heading(story, "4. Arquitetura do sistema", h2)
    story.append(
        styled_table(
            [
                ["Camada", "Responsabilidade", "Arquivos"],
                ["Frontend", "Interface para escolher A, B e visualizar resultados.", "index.html, styles.css, script.js"],
                ["API Python", "Calcula probabilidades, valida consulta e chama solver.", "app.py"],
                ["Solver", "Resolve o programa linear via SciPy linprog/HiGHS.", "requirements.txt, app.py"],
                ["Relatorios", "Gera PDFs e JSONs com explicacoes e saidas calculadas.", "reports/"],
                ["Deploy", "Configura Render com Python 3.11.9 e Gunicorn.", "render.yaml, Procfile, .python-version"],
                ["Compatibilidade", "Alias WSGI para start command antigo do Render.", "your_application/wsgi.py"],
            ],
            [3.3 * cm, 7.0 * cm, 5.3 * cm],
        )
    )

    heading(story, "5. Fluxo de uso", h2)
    story.append(
        Paragraph(
            "1. O usuario abre a interface.<br/>"
            "2. A interface consulta /api/metadata para buscar atributos e valores do dataset.<br/>"
            "3. O usuario escolhe o evento A e as afirmacoes B.<br/>"
            "4. A interface envia a consulta para /api/query.<br/>"
            "5. O backend calcula P(A), P(B), P(A e B), suporte, confianca e lift.<br/>"
            "6. O backend monta restricoes lineares e resolve o intervalo de P(A | B).<br/>"
            "7. A interface mostra resultado, programa linear, tempo e conclusao.",
            code,
        )
    )

    heading(story, "6. Matematica do projeto", h2)
    story.append(
        styled_table(
            [
                ["Conceito", "Formula", "Uso"],
                ["Marginal", "P(A) = count(A) / N", "Probabilidade de um valor isolado."],
                ["Conjunta", "P(A e B) = count(A e B) / N", "Suporte da regra."],
                ["Condicional", "P(A | B) = P(A e B) / P(B)", "Pergunta feita pelo usuario."],
                ["Confianca", "confidence(B -> A) = P(A | B)", "Precisao da regra."],
                ["Lift", "lift = P(A | B) / P(A)", "Forca da associacao."],
                ["Intervalo", "round(p,3) +/- 0,001", "Probabilidade intervalar para evitar rigidez numerica."],
                ["Variavel LP", "x_w >= 0", "Probabilidade de cada mundo possivel."],
                ["Normalizacao", "soma(x_w) = 1", "Distribuicao de probabilidade valida."],
                ["Restricao", "L <= soma(x_w onde E) <= U", "Conhecimento probabilistico da base."],
                ["Charnes-Cooper", "P(A e B)/P(B) -> objetivo linear", "Permite resolver condicional com linprog."],
            ],
            [3.6 * cm, 5.6 * cm, 6.4 * cm],
            header_color=INK,
        )
    )

    heading(story, "7. Solver", h2)
    story.append(
        Paragraph(
            "O solver nao e separado do projeto. Ele e uma dependencia instalada pelo requirements.txt. "
            "O backend importa scipy.optimize.linprog e executa o solver dentro da propria aplicacao. "
            "No Render, o deploy instala SciPy e Gunicorn; depois inicia o Flask app.",
            body,
        )
    )
    story.append(Paragraph("scipy.optimize.linprog(method='highs')", code))

    heading(story, "8. Saidas do sistema", h2)
    story.append(
        styled_table(
            [
                ["Saida", "Conteudo"],
                ["Interface", "Metricas, intervalo linear, programa linear e conclusao automatica."],
                ["API /api/query", "JSON dinamico com probabilidades, solver, tempo e conclusao."],
                ["relatorio_saida_programacao_linear.pdf", "Graficos, calculos, algoritmos e explicacao."],
                ["saida_calculada_programacao_linear.json", "Calculos numericos dos algoritmos."],
                ["roadmap_base_cientifica.pdf", "Roadmap e base cientifica/técnica."],
                ["conclusao_resultados_programacao_linear.pdf", "Conclusao interpretativa dos resultados."],
                ["contexto_completo_projeto.pdf", "Este documento com o contexto total do projeto."],
            ],
            [6.2 * cm, 9.4 * cm],
        )
    )

    heading(story, "9. Tratamento de casos especiais", h2)
    story.append(
        Paragraph(
            "Quando P(B)=0, a consulta condicional nao e matematicamente definida, pois haveria divisao por zero. "
            "O sistema detecta esse caso antes do solver, explica a inviabilidade e orienta o usuario a reduzir "
            "as condicoes ou escolher uma combinacao existente no dataset.",
            body,
        )
    )

    heading(story, "10. Deploy no Render", h2)
    story.append(
        Paragraph(
            "O projeto foi preparado como Web Service Python no Render. A versao Python foi fixada em 3.11.9 "
            "para evitar falha de instalacao do SciPy no Python 3.14. O start correto e gunicorn app:app, "
            "mas tambem foi criado um modulo your_application.wsgi para compatibilidade com start command antigo.",
            body,
        )
    )
    story.append(
        Paragraph(
            "Build: python -m pip install --upgrade pip setuptools wheel && python -m pip install -r requirements.txt<br/>"
            "Start: gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120<br/>"
            "Health check: /healthz",
            code,
        )
    )

    heading(story, "11. Base cientifica", h2)
    story.append(
        Paragraph(
            "O projeto se apoia em raciocinio sob incerteza, probabilidades intervalares, regras de associacao, "
            "estimativas por frequencia/verossimilhanca e programacao linear. Os links indicados pelo professor "
            "foram usados para justificar a formulacao por restricoes, o uso de intervalos, a possibilidade de "
            "smoothing e a comparacao futura de solvers.",
            body,
        )
    )

    heading(story, "12. Limites e proximos passos", h2)
    story.append(
        Paragraph(
            "O sistema ja responde consultas condicionais e gera conclusoes. Como proximos passos, e possivel "
            "adicionar Laplace smoothing na interface, gerar conjuntas de pares/trios como nivel opcional de "
            "restricao, comparar varios solvers, medir curvas de tempo e calcular acuracia quando o atributo "
            "perguntado for uma classe.",
            body,
        )
    )

    heading(story, "13. Conclusao final", h2)
    story.append(
        Paragraph(
            "Conclui-se que o projeto implementa o ciclo completo solicitado: extracao de conhecimento "
            "probabilistico, transformacao em restricoes lineares, pergunta condicional do usuario, resolucao "
            "com solver e apresentacao interpretavel do resultado. A aplicacao tambem registra tempos de "
            "processamento, gera relatorios em PDF/JSON e esta pronta para deploy no Render.",
            body,
        )
    )

    doc.build(story, onFirstPage=page_style, onLaterPages=page_style)


if __name__ == "__main__":
    build_report()
    print(OUTPUT)
