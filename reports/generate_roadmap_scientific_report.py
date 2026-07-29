from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports" / "roadmap_base_cientifica.pdf"

INK = colors.HexColor("#111827")
MUTED = colors.HexColor("#374151")
GRID = colors.HexColor("#e5e7eb")
TEAL = colors.HexColor("#0f766e")
TEAL_DARK = colors.HexColor("#176b5b")
SOFT = colors.HexColor("#f7faf9")


def styled_table(rows: list[list[str]], widths: list[float], header_color=TEAL_DARK) -> Table:
    table = Table(rows, colWidths=widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), header_color),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.4, GRID),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.4),
                ("LEADING", (0, 0), (-1, -1), 10.5),
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
    canvas.setLineWidth(0.7)
    canvas.line(doc.leftMargin, height - 1.05 * cm, width - doc.rightMargin, height - 1.05 * cm)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(doc.leftMargin, 0.75 * cm, "Roadmap - Probabilidades, Regras e Programacao Linear")
    canvas.drawRightString(width - doc.rightMargin, 0.75 * cm, f"Pagina {doc.page}")
    canvas.restoreState()


def add_heading(story: list, text: str, style: ParagraphStyle) -> None:
    story.append(Spacer(1, 0.18 * cm))
    story.append(Paragraph(text, style))
    story.append(Spacer(1, 0.08 * cm))


def build_report() -> None:
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
        fontSize=9.8,
        leading=13.5,
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
        title="Roadmap e Base Cientifica",
    )

    story: list = []
    story.append(Paragraph("Roadmap do Projeto e Base Cientifica", title))
    story.append(
        Paragraph(
            "Sistema web em Python/Flask para extrair conhecimento probabilistico "
            "de uma base categorica, montar restricoes de programacao linear e "
            "responder perguntas condicionais feitas pelo usuario.",
            subtitle,
        )
    )

    add_heading(story, "1. Objetivo geral", h2)
    story.append(
        Paragraph(
            "O objetivo implementado foi construir uma aplicacao dinamica em que o "
            "usuario escolhe afirmacoes e perguntas na interface. O backend calcula "
            "probabilidades, suporte, confianca, lift, intervalos e resolve a consulta "
            "por programacao linear usando o dataset de recomendacao de culturas.",
            body,
        )
    )

    add_heading(story, "2. Roadmap do que foi feito", h2)
    roadmap = styled_table(
        [
            ["Etapa", "Implementacao realizada", "Evidencia no projeto"],
            ["Dataset", "Foi mantido apenas o dataset do solo/culturas no repositorio trabalho.", "data/Crop_recommendation.csv"],
            ["Interface", "Tela para escolher evento A e condicoes B dinamicamente.", "index.html, styles.css, script.js"],
            ["Backend Python", "API Flask para metadados, consulta e health check.", "app.py"],
            ["Deteccao dinamica", "O CSV e lido; colunas numericas e categoricas sao detectadas automaticamente.", "load_dataset()"],
            ["Discretizacao", "Atributos numericos viram categorias baixo, medio e alto; pH vira acido, neutro e alcalino.", "category_for()"],
            ["Probabilidades", "Calculo de P(A), P(B), P(A e B) e contagens.", "probability(), probability_count()"],
            ["Regras", "Suporte, confianca/precisao e lift sao calculados para B -> A.", "/api/query"],
            ["Intervalos", "Probabilidades arredondadas em tres casas viram limites inferior/superior.", "rounded_interval()"],
            ["Programacao linear", "Cada mundo possivel vira x_w; marginais e consulta viram restricoes.", "solve_linear_interval()"],
            ["Solver", "SciPy linprog com metodo HiGHS resolve minimo e maximo.", "scipy.optimize.linprog"],
            ["Saida", "Interface mostra metricas, programa linear e tempo de processamento.", "script.js e /api/query"],
            ["Relatorios", "PDF com graficos, explicacao e JSON dos calculos matematicos.", "reports/"],
            ["Deploy", "Configuracao Render Web Service Python com Python 3.11.9 fixado.", "render.yaml, .python-version"],
        ],
        [3.4 * cm, 7.0 * cm, 5.2 * cm],
        header_color=INK,
    )
    story.append(roadmap)

    add_heading(story, "3. Fluxo de processamento", h2)
    story.append(
        Paragraph(
            "O fluxo operacional do sistema e:",
            body,
        )
    )
    story.append(
        Paragraph(
            "1. Usuario escolhe A e B na interface.<br/>"
            "2. Frontend envia JSON para /api/query.<br/>"
            "3. Python valida os valores contra os dominios reais do CSV.<br/>"
            "4. Python calcula P(A), P(B), P(A e B), suporte, confianca e lift.<br/>"
            "5. Python monta restricoes lineares com intervalos de tres casas decimais.<br/>"
            "6. O solver calcula minimo e maximo de P(A | B).<br/>"
            "7. A interface apresenta metricas, intervalo, formulacao linear e tempo.",
            code,
        )
    )

    add_heading(story, "4. Algoritmos matematicos usados", h2)
    algorithms = styled_table(
        [
            ["Algoritmo", "Formula", "Uso no projeto"],
            ["Discretizacao por quantis", "x <= Q1: baixo; x <= Q2: medio; x > Q2: alto", "Transforma base numerica em base categorica."],
            ["Marginal", "P(E) = count(E) / N", "Calcula frequencia de qualquer valor de atributo."],
            ["Conjunta", "P(A e B) = count(A e B) / N", "Calcula suporte e restricoes da regra."],
            ["Condicional", "P(A | B) = P(A e B) / P(B)", "Representa a pergunta do usuario."],
            ["Suporte", "support(B -> A) = P(A e B)", "Mostra quanto a regra aparece na base."],
            ["Confianca", "confidence(B -> A) = P(A | B)", "Mostra precisao da regra."],
            ["Lift", "lift = P(A | B) / P(A)", "Compara a regra com a chance geral de A."],
            ["Intervalo", "round(p,3)-0,001 <= P(E) <= round(p,3)+0,001", "Evita rigidez numerica e representa incerteza intervalar."],
            ["LP", "min c^T x sujeito a A_ub x <= b_ub, A_eq x = b_eq", "Formato usado pelo scipy.optimize.linprog."],
            ["Charnes-Cooper", "P(A e B)/P(B) -> x_w = y_w/t, P(B) em y = 1", "Transforma razao condicional em objetivo linear."],
        ],
        [3.5 * cm, 5.9 * cm, 6.2 * cm],
        header_color=TEAL,
    )
    story.append(algorithms)

    add_heading(story, "5. Base cientifica e tecnica dos links", h2)
    sources = styled_table(
        [
            ["Fonte indicada", "Ideia aproveitada", "Aplicacao no projeto"],
            [
                "Mathematical programming models for reasoning under uncertainty",
                "Raciocinio sob incerteza pode ser formulado por modelos de programacao matematica.",
                "Probabilidades extraidas do dataset sao tratadas como restricoes e a consulta e resolvida por otimizacao.",
            ],
            [
                "ScienceDirect: Artificial Intelligence, PII 0004370294900795",
                "Base teorica para representar conhecimento incerto por restricoes e inferencia.",
                "Justifica calcular limites inferior/superior, nao apenas um ponto fixo.",
            ],
            [
                "B. Tessem, Interval probability propagation, IJAR 7 (1992) 95-120",
                "Probabilidades intervalares representam incerteza por limites.",
                "O sistema usa intervalos em tres casas para P(A), P(B) e P(A e B).",
            ],
            [
                "Laplace smoothing",
                "Evita probabilidade zero em estimativas por frequencia.",
                "Foi documentado como extensao cientifica para bases com eventos raros ou ausentes.",
            ],
            [
                "Material sobre verossimilhanca",
                "Probabilidades podem ser estimadas por frequencia/maximum likelihood.",
                "P(E)=count(E)/N e usado como estimador empirico das probabilidades.",
            ],
            [
                "SciPy linprog",
                "Define LP como minimizacao linear com restricoes de igualdade/desigualdade.",
                "Foi o solver usado no backend: scipy.optimize.linprog(method='highs').",
            ],
            [
                "Gurobi - Linear Programming in Python",
                "Fluxo geral: definir variaveis, funcao objetivo, restricoes, resolver e interpretar.",
                "O projeto segue esse fluxo, usando SciPy como solver aberto no deploy.",
            ],
            [
                "lp_solve",
                "Solver MILP/LP livre baseado em simplex revisado; reforca que restricoes devem ser lineares.",
                "Entrou como referencia alternativa e comparacao futura de solver.",
            ],
            [
                "COPT-Public/cuPDLP-C",
                "Referencia moderna para resolvedores de LP em larga escala.",
                "Entrou como base para futuro benchmark de tempo, tamanho e speedup.",
            ],
        ],
        [4.7 * cm, 5.6 * cm, 5.3 * cm],
        header_color=INK,
    )
    story.append(sources)

    add_heading(story, "5.1 Como a fundamentacao virou codigo", h2)
    story.append(
        Paragraph(
            "A implementacao foi organizada como uma cadeia de inferencia probabilistica. "
            "Primeiro, o dataset e carregado e categorizado. Depois, o sistema minera "
            "frequencias empiricas para calcular probabilidades marginais, conjuntas e "
            "condicionais. Em seguida, a consulta escolhida pelo usuario e interpretada "
            "como uma regra explicita B -> A, com suporte, confianca e lift.",
            body,
        )
    )
    story.append(
        Paragraph(
            "Na etapa de programacao linear, cada mundo possivel w recebe uma variavel x_w. "
            "A normalizacao soma(x_w)=1 garante uma distribuicao valida. As probabilidades "
            "mineradas sao transformadas em restricoes intervalares do tipo L <= soma(x_w "
            "onde o evento ocorre) <= U. Isso inclui marginais, conjuntas por pares, "
            "restricoes da consulta selecionada e, quando houver suporte positivo, a regra "
            "de associacao selecionada.",
            body,
        )
    )
    story.append(
        Paragraph(
            "A consulta P(A | B) e uma funcao fracionaria, pois depende de P(A e B)/P(B). "
            "Para manter o problema compativel com um solver de programacao linear, o projeto "
            "usa a transformacao de Charnes-Cooper. Depois da transformacao, o HiGHS, chamado "
            "por scipy.optimize.linprog, resolve duas otimizacoes: uma para o limite inferior "
            "e outra para o limite superior da probabilidade condicional.",
            body,
        )
    )
    story.append(
        Paragraph(
            "A validacao foi implementada com um solver separado. A interface envia a mesma "
            "consulta para esse script independente e compara suporte, confianca, lift, "
            "intervalo linear e tempo de execucao. Desse modo, o projeto mostra tanto o "
            "resultado principal quanto uma verificacao externa da montagem do modelo.",
            body,
        )
    )

    add_heading(story, "6. Justificativa das escolhas", h2)
    story.append(
        Paragraph(
            "A escolha de SciPy/HiGHS foi feita porque o trabalho precisava de um "
            "solver executavel em Python e adequado ao Render. A documentacao do SciPy "
            "define exatamente o formato usado: objetivo linear, restricoes A_ub x <= b_ub, "
            "restricoes A_eq x = b_eq e limites para variaveis. Como P(A | B) e uma razao, "
            "a consulta foi transformada por Charnes-Cooper para manter o problema linear.",
            body,
        )
    )
    story.append(
        styled_table(
            [
                ["Solver", "Papel no projeto", "Situacao da comparacao"],
                ["SciPy HiGHS", "Solver usado pela interface e pelo script separado.", "Executado e comparado numericamente."],
                ["HiGHS Dual Simplex", "Referencia tecnica dentro da familia HiGHS.", "Documentado para benchmark futuro."],
                ["HiGHS Interior Point", "Referencia tecnica dentro da familia HiGHS.", "Documentado para benchmark futuro."],
                ["Gurobi", "Solver comercial de referencia para LP/MILP.", "Nao executado por licenca/deploy."],
                ["lp_solve", "Solver livre tradicional para LP/MILP.", "Nao executado nesta versao."],
                ["cuPDLP-C", "Solver moderno para LP em larga escala.", "Nao executado nesta versao."],
            ],
            [3.8 * cm, 5.8 * cm, 6.0 * cm],
            header_color=INK,
        )
    )

    add_heading(story, "7. Saidas geradas", h2)
    outputs = styled_table(
        [
            ["Arquivo", "Conteudo"],
            ["reports/relatorio_saida_programacao_linear.pdf", "Explicacao, graficos, algoritmos matematicos, metricas e LP."],
            ["reports/saida_calculada_programacao_linear.json", "Resultados numericos, contagens, tempos e calculos por algoritmo."],
            ["reports/roadmap_base_cientifica.pdf", "Roadmap completo e base cientifica/técnica do projeto."],
            ["/api/query", "Saida dinamica com suporte, confianca, lift, intervalo linear e tempo de processamento."],
        ],
        [6.3 * cm, 9.3 * cm],
        header_color=TEAL_DARK,
    )
    story.append(outputs)

    add_heading(story, "8. Proximos passos sugeridos", h2)
    story.append(
        Paragraph(
            "Para evoluir o projeto, os proximos passos sao: adicionar Laplace smoothing "
            "como opcao na interface; permitir conjuntas de pares/trios como nivel de "
            "restricao; comparar SciPy, Gurobi, lp_solve e outros solvers; medir curva "
            "de crescimento de tempo por numero de variaveis/restricoes; e calcular "
            "acuracia quando o atributo escolhido for uma classe.",
            body,
        )
    )

    add_heading(story, "9. Links de referencia", h2)
    story.append(
        Paragraph(
            "ResearchGate: https://www.researchgate.net/publication/260908286_Mathematical_programming_models_for_reasoning_under_uncertainty<br/>"
            "ScienceDirect: https://www.sciencedirect.com/science/article/abs/pii/0004370294900795<br/>"
            "Tessem IJAR: Interval probability propagation, International Journal of Approximate Reasoning 7 (1992) 95-120<br/>"
            "ScienceDirect: https://www.sciencedirect.com/science/article/pii/S0165489625000241<br/>"
            "SciPy linprog: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html<br/>"
            "Gurobi LP Python: https://www.gurobi.com/resources/blog/introduction-to-linear-programming-in-python<br/>"
            "Gurobi docs: https://docs.gurobi.com/current/<br/>"
            "lp_solve: https://sourceforge.net/projects/lpsolve/<br/>"
            "cuPDLP-C: https://github.com/COPT-Public/cuPDLP-C<br/>"
            "Laplace smoothing: materiais indicados no Classroom e referencias complementares.",
            code,
        )
    )

    doc.build(story, onFirstPage=page_style, onLaterPages=page_style)


if __name__ == "__main__":
    build_report()
    print(OUTPUT)
