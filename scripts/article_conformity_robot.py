#!/usr/bin/env python3
"""Executa a auditoria artigo-projeto e grava o relatorio de evidencias."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from article_conformity import run_article_conformity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compara a implementacao com o artigo-base e reprova arredondamento no PL."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "article-conformity-report.json",
        help="Caminho do relatorio JSON.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Pula somente o experimento mais caro de selecao ativa.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run_article_conformity(include_active_selection=not args.quick)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    summary = report["summary"]
    print("ROBO DE CONFORMIDADE ARTIGO-PROJETO")
    print(f"Classificacao: {report['classification']}")
    print("Politica numerica: p completo +/- 0,001, sem arredondamento no PL")
    for check in report["checks"]:
        marker = "OK" if check["status"] == "aprovado" else "FALHA"
        print(f"[{marker}] {check['name']} ({check['duration_seconds']:.3f}s)")
        if check.get("error"):
            print(f"  {check['error']}")
    print(
        f"Resultado: {summary['approved']}/{summary['total']} verificacoes aprovadas; "
        f"relatorio: {args.output}"
    )
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
