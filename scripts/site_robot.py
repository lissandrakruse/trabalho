#!/usr/bin/env python3
"""Robo de teste funcional do site Probabilidades do Solo.

O script usa somente a biblioteca padrao do Python. Ele testa a aplicacao como
um cliente HTTP real, registra evidencias em JSON e retorna codigo diferente de
zero quando qualquer verificacao falha.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen


DEFAULT_BASE_URL = "https://trabalho-bh30.onrender.com/"
REFERENCE_DIGEST = "6a366deaf3e50926268e8cb12c3ff30531185c3b65a7f78695bb2c8db093a651"
REFERENCE_QUERY = {
    "target": {"attribute": "label", "value": "rice"},
    "conditions": [
        {"attribute": "ph", "value": "acido"},
        {"attribute": "rainfall", "value": "alto"},
    ],
}
MISSING_RULE_QUERY = {
    "target": {"attribute": "label", "value": "apple"},
    "conditions": [
        {"attribute": "N", "value": "alto"},
        {"attribute": "P", "value": "alto"},
    ],
}


class RobotFailure(RuntimeError):
    """Falha de uma verificacao funcional do robo."""


@dataclass
class CheckResult:
    name: str
    status: str
    duration_seconds: float
    evidence: dict[str, Any]
    error: str | None = None


class SiteClient:
    def __init__(self, base_url: str, timeout: float) -> None:
        self.base_url = base_url.rstrip("/") + "/"
        self.timeout = timeout

    def request(
        self,
        path: str,
        *,
        method: str = "GET",
        payload: dict[str, Any] | None = None,
    ) -> tuple[bytes, dict[str, str]]:
        body = None
        headers = {
            "Accept": "application/json, text/plain, application/pdf, text/html",
            "User-Agent": "Probabilidades-do-Solo-Robot/1.0",
        }
        if payload is not None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"

        request = Request(
            urljoin(self.base_url, path.lstrip("/")),
            data=body,
            headers=headers,
            method=method,
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                return response.read(), dict(response.headers.items())
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[:1000]
            raise RobotFailure(f"HTTP {error.code} em {path}: {detail}") from error
        except (URLError, TimeoutError) as error:
            raise RobotFailure(f"Falha de conexao em {path}: {error}") from error

    def json(
        self,
        path: str,
        *,
        method: str = "GET",
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        body, _ = self.request(path, method=method, payload=payload)
        try:
            result = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RobotFailure(f"Resposta JSON invalida em {path}") from error
        if not isinstance(result, dict):
            raise RobotFailure(f"Resposta inesperada em {path}: objeto JSON esperado")
        return result


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RobotFailure(message)


def close_to(actual: Any, expected: float, tolerance: float = 1e-9) -> bool:
    try:
        return math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=tolerance)
    except (TypeError, ValueError):
        return False


def wait_until_healthy(
    client: SiteClient,
    wait_seconds: int,
    expected_commit: str | None = None,
) -> dict[str, Any]:
    deadline = time.monotonic() + wait_seconds
    last_error = "servico ainda nao consultado"
    while True:
        try:
            result = client.json("/healthz")
            deployed_commit = str(result.get("commit") or "")
            commit_matches = not expected_commit or deployed_commit.startswith(expected_commit)
            if result.get("ok") is True and commit_matches:
                return result
            if result.get("ok") is True and not commit_matches:
                last_error = (
                    f"deploy atual {deployed_commit or 'sem SHA'}; "
                    f"aguardando {expected_commit}"
                )
            else:
                last_error = f"resposta inesperada: {result}"
        except RobotFailure as error:
            last_error = str(error)
        if time.monotonic() >= deadline:
            raise RobotFailure(f"Servico nao ficou saudavel: {last_error}")
        time.sleep(min(10, max(1, int(deadline - time.monotonic()))))


def run_check(name: str, action: Callable[[], dict[str, Any]]) -> CheckResult:
    started = time.perf_counter()
    try:
        evidence = action()
        return CheckResult(name, "aprovado", time.perf_counter() - started, evidence)
    except Exception as error:  # noqa: BLE001 - o relatorio precisa registrar toda falha
        return CheckResult(name, "reprovado", time.perf_counter() - started, {}, str(error))


def robot_run(
    client: SiteClient,
    wait_seconds: int,
    include_artifacts: bool,
    expected_commit: str | None = None,
) -> dict[str, Any]:
    state: dict[str, Any] = {}
    checks: list[CheckResult] = []

    checks.append(
        run_check(
            "saude_do_servico",
            lambda: {
                "response": wait_until_healthy(client, wait_seconds, expected_commit),
                "expected_commit": expected_commit,
            },
        )
    )

    def check_home() -> dict[str, Any]:
        home, _ = client.request("/")
        script, _ = client.request("/script.js")
        home_text = html.unescape(home.decode("utf-8", errors="replace"))
        script_text = script.decode("utf-8", errors="replace")
        expected_labels = [
            "Resultado do programa linear: P(A | B)",
            "Resumo didático",
            "Gerar modelo auditável",
            "Baixar matrizes exatas em TXT",
        ]
        for label in expected_labels:
            require(label in home_text, f"Rotulo ausente na interface: {label}")
        require("não calculado" not in script_text.lower(), "A mensagem generica 'não calculado' reapareceu")
        require("nao calculado" not in script_text.lower(), "A mensagem generica 'nao calculado' reapareceu")
        return {"labels": expected_labels, "generic_not_calculated_absent": True}

    checks.append(run_check("interface_publica", check_home))

    def check_metadata() -> dict[str, Any]:
        metadata = client.json("/api/metadata")
        require(metadata.get("total") == 2200, "Quantidade de registros diferente de 2.200")
        require(metadata.get("omegaWorlds") == 466, "Quantidade de mundos diferente de 466")
        apriori = metadata.get("apriori") or {}
        require(apriori.get("ruleCount") == 5312, "Quantidade de regras Apriori diferente de 5.312")
        state["metadata"] = metadata
        return {
            "records": metadata["total"],
            "worlds": metadata["omegaWorlds"],
            "apriori_rules": apriori["ruleCount"],
        }

    checks.append(run_check("dataset_e_apriori", check_metadata))

    def check_query() -> dict[str, Any]:
        result = client.json("/api/query", method="POST", payload=REFERENCE_QUERY)
        require(result.get("ok") is True, f"Consulta recusada: {result.get('error')}")
        linear = result.get("linear") or {}
        require(linear.get("ok") is True, f"Solver falhou: {linear.get('error')}")
        require(result.get("countBoth") == 33, "Contagem A e B diferente de 33")
        require(result.get("countBase") == 218, "Contagem de B diferente de 218")
        require(close_to(result.get("support"), 0.015), "Suporte diferente de 0,015")
        require(close_to(result.get("confidence"), 33 / 218), "Confianca diferente de 33/218")
        require(close_to(linear.get("lower"), 0.15005815141370793, 1e-8), "Limite inferior inesperado")
        require(close_to(linear.get("upper"), 0.152, 1e-8), "Limite superior inesperado")
        require(linear.get("worldVariables") == 466, "Quantidade de variaveis de mundos incorreta")
        require(linear.get("solverVariables") == 467, "Quantidade de variaveis do solver incorreta")
        require(linear.get("constraints") == 15018, "Quantidade de restricoes incorreta")
        require(linear.get("modelDigest") == REFERENCE_DIGEST, "SHA-256 do modelo diferente da referencia")
        state["query"] = result
        return {
            "support": result["support"],
            "confidence": result["confidence"],
            "lift": result["lift"],
            "count_both": result["countBoth"],
            "count_base": result["countBase"],
            "lower": linear["lower"],
            "upper": linear["upper"],
            "world_variables": linear["worldVariables"],
            "solver_variables": linear["solverVariables"],
            "constraints": linear["constraints"],
            "model_digest": linear["modelDigest"],
        }

    checks.append(run_check("consulta_de_referencia", check_query))

    def check_missing_rule() -> dict[str, Any]:
        result = client.json("/api/query", method="POST", payload=MISSING_RULE_QUERY)
        require(result.get("ok") is True, f"Consulta sem regra recusada: {result.get('error')}")
        require(result.get("releasedAssociationRule") is None, "O sistema fabricou uma regra Apriori para a consulta")
        require(result.get("support") is None, "Suporte deveria ficar vazio quando a regra nao foi gerada")
        require(result.get("confidence") is None, "Confianca deveria ficar vazia quando a regra nao foi gerada")
        require(result.get("lift") is None, "Lift deveria ficar vazio quando a regra nao foi gerada")
        require(result.get("countBoth") == 0, "A consulta sem regra deveria ter zero ocorrencias conjuntas")
        require(result.get("countBase") == 103, "A consulta sem regra deveria ter 103 casos de B")
        linear = result.get("linear") or {}
        require(linear.get("ok") is True, f"Solver falhou na consulta sem regra: {linear.get('error')}")
        require(close_to(linear.get("lower"), 0.0), "Limite inferior deveria ser zero")
        require(0.0 < float(linear.get("upper", 0.0)) <= 0.005, "Limite superior deveria ser pequeno e maior que zero")
        require(linear.get("observedWorldVariables") == 466, "Quantidade de mundos observados incorreta")
        require(linear.get("queryCompletionWorlds") == 243, "Completamentos da consulta deveriam somar 243 mundos")
        require(linear.get("worldVariables") == 709, "Modelo da consulta deveria usar 709 mundos")
        require(linear.get("solverVariables") == 710, "Modelo da consulta deveria usar 710 variaveis")
        serialized = json.dumps(result, ensure_ascii=False).lower()
        require("não calculado" not in serialized, "A resposta voltou a usar 'não calculado'")
        require("nao calculado" not in serialized, "A resposta voltou a usar 'nao calculado'")
        queried_rule = result.get("queriedAssociationRule") or {}
        require(queried_rule.get("released") is False, "Status da regra ausente nao foi informado")
        require("nao" in str(queried_rule.get("reason", "")).lower(), "Motivo da regra ausente nao foi explicado")
        return {
            "rule_available": False,
            "support": result.get("support"),
            "confidence": result.get("confidence"),
            "lift": result.get("lift"),
            "count_both": result["countBoth"],
            "count_base": result["countBase"],
            "lower": linear["lower"],
            "upper": linear["upper"],
            "observed_worlds": linear["observedWorldVariables"],
            "query_completion_worlds": linear["queryCompletionWorlds"],
            "solver_variables": linear["solverVariables"],
            "reason": queried_rule.get("reason"),
        }

    checks.append(run_check("consulta_sem_regra_apriori", check_missing_rule))

    def check_solver_comparison() -> dict[str, Any]:
        prepared_methods = []
        for method in ("highs", "highs-ds"):
            prepared = client.json(
                "/api/solver/run",
                method="POST",
                payload={**REFERENCE_QUERY, "solverMethod": method},
            )
            require(prepared.get("ok") is True, f"Preparacao do metodo {method} falhou: {prepared.get('error')}")
            prepared_engine = prepared.get("solverEngineResult") or {}
            require(prepared_engine.get("status") == "ok", f"Metodo preparado {method} falhou")
            require(prepared_engine.get("method") == method, f"Metodo preparado incorreto: {method}")
            prepared_methods.append(method)
        result = client.json("/api/solver/compare", method="POST", payload=REFERENCE_QUERY)
        require(result.get("ok") is True, f"Comparacao dos solvers falhou: {result.get('error')}")
        comparison = result.get("comparison") or {}
        require(comparison.get("allMatch") is True, "Solver principal e solver independente divergiram")
        engines = result.get("solverEngineResults") or []
        require(len(engines) == 3, "A comparacao nao executou os tres metodos HiGHS")
        expected_methods = {"highs", "highs-ds", "highs-ipm"}
        require({engine.get("method") for engine in engines} == expected_methods, "Catalogo de metodos executados esta incompleto")
        for engine in engines:
            require(engine.get("status") == "ok", f"Metodo {engine.get('method')} falhou")
            require(engine.get("allMatch") is True, f"Metodo {engine.get('method')} divergiu")
            require(engine.get("variables") == 467, f"Metodo {engine.get('method')} nao usou 467 variaveis")
            require(engine.get("constraints") == 15018, f"Metodo {engine.get('method')} nao usou 15.018 restricoes")
            require(close_to(engine.get("lower"), 0.15005815141370793, 1e-8), f"Limite inferior divergente em {engine.get('method')}")
            require(close_to(engine.get("upper"), 0.152, 1e-8), f"Limite superior divergente em {engine.get('method')}")
        state["solver_comparison"] = result
        return {
            "all_match": True,
            "prepared_methods": prepared_methods,
            "engines": [
                {
                    "method": engine["method"],
                    "status": engine["status"],
                    "lower": engine["lower"],
                    "upper": engine["upper"],
                }
                for engine in engines
            ],
        }

    checks.append(run_check("comparacao_dos_tres_solvers", check_solver_comparison))

    if include_artifacts:
        def check_txt() -> dict[str, Any]:
            query = state.get("query")
            require(query is not None, "Consulta de referencia precisa ser aprovada antes do TXT")
            generated = client.json(
                "/api/linear-program/full",
                method="POST",
                payload=REFERENCE_QUERY,
            )
            require(generated.get("ok") is True, f"Geracao do TXT falhou: {generated.get('error')}")
            require(generated.get("modelDigest") == query["linear"]["modelDigest"], "TXT e solver possuem SHA-256 diferentes")
            require(generated.get("solverVariables") == 467, "TXT nao registra 467 variaveis")
            require(generated.get("constraints") == 15018, "TXT nao registra 15.018 restricoes")
            txt, _ = client.request(generated.get("downloadUrl") or generated["fileUrl"])
            text = txt.decode("utf-8", errors="strict")
            require(len(txt) > 1_000_000, "TXT auditavel esta pequeno demais")
            require(f"sha256_modelo={REFERENCE_DIGEST}" in text, "Digest ausente no TXT")
            for marker in ("c_lower=", "c_upper_as_min=", "A_ub[0]=", "A_eq[0]=", "bounds[0:y_0001]"):
                require(marker in text, f"Secao ausente no TXT: {marker}")
            require("soma(x_w)" not in text, "TXT exato contem a notacao resumida soma(x_w)")
            return {
                "bytes": len(txt),
                "model_digest": generated["modelDigest"],
                "solver_variables": generated["solverVariables"],
                "constraints": generated["constraints"],
            }

        checks.append(run_check("txt_auditavel", check_txt))

        def check_pdf() -> dict[str, Any]:
            generated = client.json("/api/report/query", method="POST", payload=REFERENCE_QUERY)
            require(generated.get("ok") is True, f"Geracao do PDF falhou: {generated.get('error')}")
            pdf, headers = client.request(generated["reportUrl"])
            require(pdf.startswith(b"%PDF-"), "Arquivo gerado nao possui cabecalho PDF")
            require(len(pdf) > 4_000, "PDF gerado esta pequeno demais")
            return {
                "bytes": len(pdf),
                "content_type": headers.get("Content-Type", ""),
                "report_url": generated["reportUrl"],
            }

        checks.append(run_check("relatorio_pdf", check_pdf))

        def check_solver_pdf() -> dict[str, Any]:
            generated = client.json(
                "/api/report/solver-comparison",
                method="POST",
                payload=REFERENCE_QUERY,
            )
            require(generated.get("ok") is True, f"PDF comparativo falhou: {generated.get('error')}")
            require((generated.get("comparison") or {}).get("allMatch") is True, "PDF comparativo foi gerado com divergencia")
            pdf, headers = client.request(generated["reportUrl"])
            require(pdf.startswith(b"%PDF-"), "Relatorio comparativo nao possui cabecalho PDF")
            require(len(pdf) > 4_000, "Relatorio comparativo esta pequeno demais")
            return {
                "bytes": len(pdf),
                "content_type": headers.get("Content-Type", ""),
                "report_url": generated["reportUrl"],
                "all_match": True,
            }

        checks.append(run_check("relatorio_pdf_dos_solvers", check_solver_pdf))

    approved = all(check.status == "aprovado" for check in checks)
    return {
        "robot": "Probabilidades do Solo - teste funcional",
        "base_url": client.base_url,
        "executed_at": datetime.now(timezone.utc).isoformat(),
        "reference_query": REFERENCE_QUERY,
        "status": "aprovado" if approved else "reprovado",
        "summary": {
            "approved": sum(check.status == "aprovado" for check in checks),
            "failed": sum(check.status == "reprovado" for check in checks),
            "total": len(checks),
        },
        "checks": [asdict(check) for check in checks],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Testa a aplicacao Probabilidades do Solo como um robo HTTP.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Endereco da aplicacao")
    parser.add_argument("--timeout", type=float, default=420.0, help="Timeout de cada requisicao em segundos")
    parser.add_argument("--wait-seconds", type=int, default=300, help="Tempo para aguardar o Render acordar")
    parser.add_argument("--expected-commit", help="SHA que precisa estar implantado antes dos testes")
    parser.add_argument("--quick", action="store_true", help="Nao gera o TXT grande nem o PDF")
    parser.add_argument("--output", type=Path, default=Path("robot-test-report.json"), help="Arquivo JSON de evidencias")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = robot_run(
        SiteClient(args.base_url, timeout=args.timeout),
        wait_seconds=max(0, args.wait_seconds),
        include_artifacts=not args.quick,
        expected_commit=args.expected_commit,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    for check in report["checks"]:
        marker = "OK" if check["status"] == "aprovado" else "FALHA"
        detail = f" - {check['error']}" if check.get("error") else ""
        print(f"[{marker}] {check['name']} ({check['duration_seconds']:.3f}s){detail}")
    print(f"Resultado: {report['status']} ({report['summary']['approved']}/{report['summary']['total']})")
    print(f"Evidencias: {args.output}")
    return 0 if report["status"] == "aprovado" else 1


if __name__ == "__main__":
    sys.exit(main())
