"""Auditoria reproduzivel entre o artigo-base e o projeto agricola.

O artigo de Ghosh e Ramakrishnan trabalha com observaveis em programas
ProbLog. Este projeto reproduz diretamente suas definicoes de VoI e o algoritmo
guloso da Figura 3, mas adapta a ideia para selecionar restricoes de um modelo
linear intervalar. O robo abaixo valida as duas camadas separadamente e trata a
ausencia de arredondamento como uma invariante auditavel do modelo agricola.
"""

from __future__ import annotations

import math
import runpy
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any, Callable

import app
from scripts import solve_query as independent_solver
from voi import ARTICLE, binary_entropy, build_conditional_plan, scenario_summary
from voi import subset_value_of_information


ROOT = Path(__file__).resolve().parent
ARTICLE_REFERENCE = {
    "title": "Value of Information in Probabilistic Logic Programs",
    "authors": ["Sarthak Ghosh", "C. R. Ramakrishnan"],
    "year": 2019,
    "doi": "10.4204/EPTCS.306.14",
    "arxiv": "1909.08234",
    "figure": 3,
}
EXPECTED_INTERVAL_RADIUS = 0.001
REFERENCE_DIGEST = "39359ee69c6cc7b1ec8c3d36c7a0b6fb85110caa2fc626de74660d75df55e39e"
REFERENCE_QUERY = {
    "target": {"attribute": "label", "value": "rice"},
    "conditions": [
        {"attribute": "ph", "value": "acido"},
        {"attribute": "rainfall", "value": "alto"},
    ],
}
REFERENCE_ACTIVE_SELECTION = {
    "target": {"attribute": "label", "value": "apple"},
    "conditions": [
        {"attribute": "ph", "value": "acido"},
        {"attribute": "rainfall", "value": "alto"},
    ],
    "budget": 25,
    "minimumLiteralOverlap": 2,
    "maxCandidates": 80,
}


class ConformityFailure(RuntimeError):
    """Falha de uma propriedade obrigatoria da auditoria."""


@dataclass
class ConformityCheck:
    name: str
    category: str
    status: str
    duration_seconds: float
    evidence: dict[str, Any]
    error: str | None = None


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ConformityFailure(message)


def close_to(actual: Any, expected: float, tolerance: float = 1e-9) -> bool:
    try:
        return math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=tolerance)
    except (TypeError, ValueError):
        return False


def expected_probability_interval(value: float) -> tuple[float, float]:
    """Especificacao independente usada para fiscalizar ``app``."""

    value = float(value)
    return (
        max(0.0, value - EXPECTED_INTERVAL_RADIUS),
        min(1.0, value + EXPECTED_INTERVAL_RADIUS),
    )


def run_check(
    name: str,
    category: str,
    action: Callable[[], dict[str, Any]],
) -> ConformityCheck:
    started = time.perf_counter()
    try:
        evidence = action()
        return ConformityCheck(
            name=name,
            category=category,
            status="aprovado",
            duration_seconds=time.perf_counter() - started,
            evidence=evidence,
        )
    except Exception as error:  # noqa: BLE001 - toda falha deve ir ao relatorio
        return ConformityCheck(
            name=name,
            category=category,
            status="reprovado",
            duration_seconds=time.perf_counter() - started,
            evidence={},
            error=str(error),
        )


def temperature_worlds() -> list[dict[str, Any]]:
    """Distribuicao do Exemplo 1 do artigo (monitoramento de temperatura)."""

    transition = {
        ("lo", "lo"): 0.7,
        ("lo", "hi"): 0.3,
        ("hi", "lo"): 0.3,
        ("hi", "hi"): 0.7,
    }
    worlds: list[dict[str, Any]] = []
    for t1, t2, t3 in product(("lo", "hi"), repeat=3):
        probability = 0.5 * transition[(t1, t2)] * transition[(t2, t3)]
        worlds.append(
            {
                "values": {
                    "T1": t1,
                    "T2": t2,
                    "T3": t3,
                    "heat_on": "sim" if "lo" in (t1, t2, t3) else "nao",
                },
                "count": probability,
            }
        )
    return worlds


def check_article_identity() -> dict[str, Any]:
    require(ARTICLE["title"] == ARTICLE_REFERENCE["title"], "Titulo do artigo divergente.")
    require(ARTICLE["authors"] == ARTICLE_REFERENCE["authors"], "Autores do artigo divergentes.")
    require(ARTICLE["year"] == ARTICLE_REFERENCE["year"], "Ano do artigo divergente.")
    require(ARTICLE["doi"] == ARTICLE_REFERENCE["doi"], "DOI do artigo divergente.")
    require(ARTICLE_REFERENCE["arxiv"] in ARTICLE["url"], "Identificador arXiv divergente.")
    return {**ARTICLE_REFERENCE, "url": ARTICLE["url"]}


def check_published_temperature_example() -> dict[str, Any]:
    worlds = temperature_worlds()
    target = {"attribute": "heat_on", "value": "sim"}
    baseline = scenario_summary(worlds, target)
    best_subset = subset_value_of_information(worlds, target, ["T1", "T3"])
    plan = build_conditional_plan(
        worlds,
        target,
        ["T1", "T2", "T3"],
        {"T1": 1, "T2": 1, "T3": 1},
        budget=2,
    )

    require(close_to(baseline["probability"], 0.755), "Pr(heat_on) nao reproduz 0,755.")
    require(close_to(baseline["entropy"], 0.8032566998), "Entropia inicial divergente.")
    require(close_to(best_subset["expectedEntropy"], 0.1805639517), "H(heat_on|T1,T3) divergente.")
    require(close_to(best_subset["voi"], 0.6226927481), "VoI de {T1,T3} divergente.")
    require(plan["algorithm"].startswith("Figura 3"), "Plano nao identifica a Figura 3.")
    require(plan["tree"]["choice"]["observable"] == "T1", "Primeira escolha gulosa deveria ser T1.")
    require(close_to(plan["planVoi"], best_subset["voi"]), "Plano guloso nao recupera o VoI publicado.")
    return {
        "queryProbability": baseline["probability"],
        "initialEntropyExact": baseline["entropy"],
        "initialEntropyArticleRounded": 0.8,
        "bestSubset": ["T1", "T3"],
        "expectedEntropyExact": best_subset["expectedEntropy"],
        "expectedEntropyArticleRounded": 0.18,
        "voiExact": best_subset["voi"],
        "voiArticleRounded": 0.62,
        "greedyFirstChoice": plan["tree"]["choice"]["observable"],
    }


def check_no_rounding_policy() -> dict[str, Any]:
    require(
        close_to(app.PROBABILITY_INTERVAL_RADIUS, EXPECTED_INTERVAL_RADIUS),
        "O raio intervalar deixou de ser 0,001.",
    )
    package_namespace = runpy.run_path(str(ROOT / "pacote_solver_colegas" / "solver.py"))
    interval_implementations = {
        "aplicacao": app.probability_interval,
        "solver_independente": independent_solver.probability_interval,
        "pacote_solver_colegas": package_namespace["probability_interval"],
    }
    cases = [0.0, 1.0, 0.976, 33 / 218, 1 / 22, 0.0004, 0.9996]
    checked: list[dict[str, Any]] = []
    for value in cases:
        expected = expected_probability_interval(value)
        implementations: dict[str, dict[str, float]] = {}
        for name, implementation in interval_implementations.items():
            actual = implementation(value)
            require(
                close_to(actual[0], expected[0]),
                f"Limite inferior incorreto em {name} para p={value}.",
            )
            require(
                close_to(actual[1], expected[1]),
                f"Limite superior incorreto em {name} para p={value}.",
            )
            require(
                actual[0] <= value + 1e-15 <= actual[1] + 1e-15,
                f"O intervalo de {name} nao contem p={value}.",
            )
            implementations[name] = {"lower": actual[0], "upper": actual[1]}
        checked.append({"probability": value, "implementations": implementations})
    return {
        "formula": "max(0, p-0.001) <= P(E) <= min(1, p+0.001)",
        "computationalRounding": False,
        "radius": EXPECTED_INTERVAL_RADIUS,
        "solversChecked": list(interval_implementations),
        "cases": checked,
        "importantDistinction": (
            "p completo e o centro; 0,001 e a tolerancia das evidencias; "
            "1e-8 e apenas a tolerancia "
            "numerica de comparacao/poda do solver"
        ),
    }


def sparse_close(left: dict[int, float], right: dict[int, float], tolerance: float = 1e-12) -> bool:
    indexes = set(left) | set(right)
    return all(close_to(left.get(index, 0.0), right.get(index, 0.0), tolerance) for index in indexes)


def check_exact_probabilities_in_all_constraints() -> dict[str, Any]:
    data = app.load_dataset()
    a_ub, b_ub, records = app.cached_linear_constraint_model()
    records_that_would_change_if_rounded = 0
    counts: dict[str, int] = {}

    for record in records:
        value = float(record["value"])
        expected_lower, expected_upper = expected_probability_interval(value)
        lower = float(record["lower"])
        upper = float(record["upper"])
        require(close_to(lower, expected_lower), f"Lower divergente em {record['kind']}.")
        require(close_to(upper, expected_upper), f"Upper divergente em {record['kind']}.")
        require(lower <= value + 1e-15 <= upper + 1e-15, f"Valor fora do intervalo em {record['kind']}.")
        require(0.0 <= lower <= upper <= 1.0, f"Intervalo fora de [0,1] em {record['kind']}.")
        require(upper - lower <= 0.002 + 1e-12, f"Intervalo largo demais em {record['kind']}.")
        rounded_lower = max(0.0, round(value, 3) - EXPECTED_INTERVAL_RADIUS)
        rounded_upper = min(1.0, round(value, 3) + EXPECTED_INTERVAL_RADIUS)
        if not close_to(lower, rounded_lower, 1e-12) or not close_to(upper, rounded_upper, 1e-12):
            records_that_would_change_if_rounded += 1
        counts[record["kind"]] = counts.get(record["kind"], 0) + 1

        row_indexes = record.get("rowIndexes") or []
        require(len(row_indexes) == 2, f"Restricao sem duas faces em {record['kind']}.")
        upper_index, lower_index = row_indexes
        if "conditions" in record:
            mask = app.sparse_world_mask(data["worlds"], record["conditions"])
            negative_mask = {index: -coefficient for index, coefficient in mask.items()}
            require(sparse_close(a_ub[upper_index], mask), "Mascara upper de evento divergente.")
            require(sparse_close(a_ub[lower_index], negative_mask), "Mascara lower de evento divergente.")
            require(close_to(b_ub[upper_index], upper), "b_ub upper nao usa a probabilidade completa.")
            require(close_to(b_ub[lower_index], -lower), "b_ub lower nao usa a probabilidade completa.")
        else:
            antecedent = app.sparse_world_mask(data["worlds"], record["antecedent"])
            both = app.sparse_world_mask(
                data["worlds"],
                [*record["antecedent"], *record["consequent"]],
            )
            expected_upper_row = app.add_sparse_vectors(both, antecedent, -upper)
            expected_lower_row = app.add_sparse_vectors(
                {index: -coefficient for index, coefficient in both.items()},
                antecedent,
                lower,
            )
            require(sparse_close(a_ub[upper_index], expected_upper_row), "Linha upper de confianca divergente.")
            require(sparse_close(a_ub[lower_index], expected_lower_row), "Linha lower de confianca divergente.")
            require(close_to(b_ub[upper_index], 0.0), "b_ub upper da confianca deveria ser zero.")
            require(close_to(b_ub[lower_index], 0.0), "b_ub lower da confianca deveria ser zero.")

    require(len(a_ub) == 6802, "O modelo deveria ter 6.802 desigualdades antes das duas igualdades.")
    require(counts.get("apriori_rule_confidence") == 1205, "Quantidade de confiancas fortes divergente.")
    require(records_that_would_change_if_rounded > 0, "A auditoria nao encontrou evidencias sensiveis a arredondamento.")
    return {
        "constraintRecordsChecked": len(records),
        "inequalityRowsChecked": len(a_ub),
        "counts": counts,
        "computationalRounding": False,
        "recordsThatWouldDifferUnderThreeDecimalRounding": records_that_would_change_if_rounded,
        "allOriginalProbabilitiesInsideIntervals": True,
        "allNumericRowsMatchDeclaredBounds": True,
    }


def check_interval_solver() -> dict[str, Any]:
    result = app.compute_query(REFERENCE_QUERY)
    linear = result.get("linear") or {}
    require(result.get("ok") is True and linear.get("ok") is True, "Consulta linear falhou.")
    require(close_to(linear["lower"], 0.13987284287011797, 1e-8), "Limite inferior divergente.")
    require(close_to(linear["upper"], 0.1631139944392957, 1e-8), "Limite superior divergente.")
    require(linear["constraints"] == 6804, "Quantidade total de restricoes divergente.")
    require(linear["modelDigest"] == REFERENCE_DIGEST, "Digest do modelo matematico divergente.")
    return {
        "query": "P(label=rice | ph=acido, rainfall=alto)",
        "lower": linear["lower"],
        "upper": linear["upper"],
        "constraints": linear["constraints"],
        "modelDigest": linear["modelDigest"],
    }


def check_zero_joint_semantics() -> dict[str, Any]:
    result = app.compute_query(
        {
            "target": {"attribute": "label", "value": "apple"},
            "conditions": [
                {"attribute": "N", "value": "alto"},
                {"attribute": "P", "value": "alto"},
            ],
        }
    )
    linear = result.get("linear") or {}
    require(result.get("pAB") == 0.0, "A auditoria empirica deveria registrar P(A e B)=0.")
    require(linear.get("ok") is True, "O LP da conjuncao nao observada falhou.")
    require(close_to(linear.get("lower"), 0.0), "O limite inferior deveria ser zero.")
    require(0.0 < float(linear.get("upper", 0.0)) <= 0.005, "O upper deveria ser pequeno e positivo.")
    require(linear.get("queryCompletionWorlds") == 243, "Completamento de mundos divergente.")
    return {
        "empiricalJoint": result["pAB"],
        "lower": linear["lower"],
        "upper": linear["upper"],
        "queryCompletionWorlds": linear["queryCompletionWorlds"],
        "interpretation": "ausencia na amostra nao foi convertida em impossibilidade logica",
    }


def check_agricultural_mapping() -> dict[str, Any]:
    plan = app.compute_voi_plan(
        {
            "target": {"attribute": "label", "value": "rice"},
            "budget": 2,
            "observables": [
                {"attribute": attribute, "cost": 1}
                for attribute in ("N", "P", "K", "temperature", "humidity", "ph", "rainfall")
            ],
        }
    )
    require(plan["tree"]["choice"]["observable"] == "rainfall", "Primeira medicao agricola divergente.")
    require(close_to(plan["planVoi"], 0.15818255058040318), "VoI agricola divergente.")
    require(plan["computation"]["linearSolverUsed"] is False, "VoI foi confundido com o solver linear.")
    return {
        "directArticleLayer": "observaveis, realizacoes, custos, cenarios, entropia e Figura 3",
        "agriculturalFirstObservable": plan["tree"]["choice"]["observable"],
        "agriculturalPlanVoi": plan["planVoi"],
        "linearSolverUsedForArticlePlan": plan["computation"]["linearSolverUsed"],
    }


def check_active_selection_adaptation() -> dict[str, Any]:
    result = app.compute_active_selection(REFERENCE_ACTIVE_SELECTION)
    active = result["activeSelection"]
    effort = result["solverEffort"]
    require(result["selectedCount"] == 23, "Quantidade selecionada divergente.")
    require(result["candidatePool"]["evaluated"] == 46, "Quantidade de candidatas divergente.")
    require(close_to(active["width"], 0.023055963110458988), "Largura ativa divergente.")
    require(close_to(active["relativeWidthReduction"], 0.8937585576919349), "Reducao ativa divergente.")
    require(effort["selectedEndpointLpSolves"] == 35, "Numero de reotimizacoes divergente.")
    require(effort["exactPruningRate"] > 0.70, "Taxa de poda exata abaixo da referencia.")
    require(effort["totalAvoidanceRate"] > 0.97, "Taxa total de economia abaixo da referencia.")
    return {
        "relationshipToArticle": "adaptacao inspirada em VoI; nao reproducao literal de ProbLog",
        "query": "P(label=apple | ph=acido, rainfall=alto)",
        "selectedConstraints": result["selectedCount"],
        "candidateConstraints": result["candidatePool"]["evaluated"],
        "activeWidth": active["width"],
        "relativeWidthReduction": active["relativeWidthReduction"],
        "exactPruningRate": effort["exactPruningRate"],
        "totalAvoidanceRate": effort["totalAvoidanceRate"],
    }


def check_documentation() -> dict[str, Any]:
    readme = (ROOT / "README_VOI.md").read_text(encoding="utf-8")
    required_text = [
        "p - 0,001",
        "0,001",
        "sem arredondamento",
        "reprodução do artigo-base",
        "adaptação",
        "poda é exata",
        "não garante o subconjunto globalmente ótimo",
    ]
    for item in required_text:
        require(item.casefold() in readme.casefold(), f"README_VOI nao explica: {item}")
    report_path = ROOT / "output" / "pdf" / "relatorio_selecao_ativa_agricultura.pdf"
    require(report_path.is_file(), "Relatorio cientifico final ausente.")
    require(report_path.stat().st_size > 10_000, "Relatorio cientifico parece incompleto.")
    return {
        "readme": "README_VOI.md",
        "scientificReport": str(report_path.relative_to(ROOT)),
        "scientificReportBytes": report_path.stat().st_size,
        "roundingExplained": True,
        "directReproductionSeparatedFromAdaptation": True,
    }


def run_article_conformity(*, include_active_selection: bool = True) -> dict[str, Any]:
    """Executa a matriz de conformidade e devolve evidencias serializaveis."""

    started = time.perf_counter()
    definitions: list[tuple[str, str, Callable[[], dict[str, Any]]]] = [
        ("identidade_bibliografica", "reproducao_artigo", check_article_identity),
        ("exemplo_temperatura_publicado", "reproducao_artigo", check_published_temperature_example),
        ("politica_sem_arredondamento", "modelo_intervalar", check_no_rounding_policy),
        ("probabilidades_exatas_em_todas_as_restricoes", "modelo_intervalar", check_exact_probabilities_in_all_constraints),
        ("consulta_intervalar_de_referencia", "modelo_intervalar", check_interval_solver),
        ("conjuncao_nao_observada", "modelo_intervalar", check_zero_joint_semantics),
        ("mapeamento_para_agricultura", "adaptacao_agricola", check_agricultural_mapping),
    ]
    if include_active_selection:
        definitions.append(
            ("selecao_ativa_de_restricoes", "adaptacao_agricola", check_active_selection_adaptation)
        )
    definitions.append(("documentacao_e_relatorio", "rastreabilidade", check_documentation))

    checks = [run_check(name, category, action) for name, category, action in definitions]
    failed = [check for check in checks if check.status != "aprovado"]
    categories: dict[str, dict[str, int]] = {}
    for check in checks:
        summary = categories.setdefault(check.category, {"approved": 0, "failed": 0})
        summary["approved" if check.status == "aprovado" else "failed"] += 1

    return {
        "ok": not failed,
        "status": "aprovado" if not failed else "reprovado",
        "classification": (
            "conforme_com_adaptacao_explicita" if not failed else "nao_conforme"
        ),
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "article": ARTICLE_REFERENCE,
        "scope": {
            "directReproduction": (
                "Definicao 7, utilidade por entropia, custos, cenarios e algoritmo "
                "guloso da Figura 3"
            ),
            "explicitAdaptation": (
                "selecao de restricoes para reduzir a largura de P(A|B), com poda "
                "exata dos extremos do programa linear"
            ),
            "notClaimed": "uma implementacao literal completa do sistema ProbLog do artigo",
        },
        "intervalPolicy": {
            "computationalRounding": False,
            "radius": EXPECTED_INTERVAL_RADIUS,
            "formula": "p +/- 0.001, limitado a [0,1]",
        },
        "summary": {
            "total": len(checks),
            "approved": len(checks) - len(failed),
            "failed": len(failed),
            "categories": categories,
            "durationSeconds": time.perf_counter() - started,
        },
        "checks": [asdict(check) for check in checks],
    }
