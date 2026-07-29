from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import compute_query, load_dataset  # noqa: E402


def parse_condition(text: str) -> dict[str, str]:
    if "=" not in text:
        raise argparse.ArgumentTypeError("Use o formato atributo=valor. Exemplo: ph=acido")
    attribute, value = text.split("=", 1)
    attribute = attribute.strip()
    value = value.strip()
    if not attribute or not value:
        raise argparse.ArgumentTypeError("Atributo e valor nao podem ficar vazios.")
    return {"attribute": attribute, "value": value}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Resolve uma consulta probabilistica condicional usando o solver linear "
            "do projeto. Exemplo: python scripts/solve_query.py --target label=rice "
            "--condition ph=acido --condition rainfall=alto"
        )
    )
    parser.add_argument(
        "--target",
        type=parse_condition,
        help="Evento A no formato atributo=valor. Exemplo: label=rice",
    )
    parser.add_argument(
        "--condition",
        action="append",
        default=[],
        type=parse_condition,
        help="Condicao B no formato atributo=valor. Pode repetir. Exemplo: --condition ph=acido",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Caminho opcional para salvar o resultado JSON.",
    )
    parser.add_argument(
        "--show-domains",
        action="store_true",
        help="Mostra atributos e valores validos antes de resolver.",
    )
    return parser


def domains_payload() -> dict[str, Any]:
    data = load_dataset()
    return {
        "total": data["total"],
        "attributes": data["attributes"],
        "numericAttributes": data["numericAttributes"],
        "categoricalAttributes": data["categoricalAttributes"],
        "domains": data["domains"],
    }


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.show_domains:
        print(json.dumps(domains_payload(), ensure_ascii=False, indent=2))
        if not args.target:
            return 0

    if not args.target:
        parser.error("informe --target atributo=valor ou use --show-domains para listar os valores")

    payload = {
        "target": args.target,
        "conditions": args.condition,
    }

    try:
        result = compute_query(payload)
    except ValueError as error:
        print(f"Erro: {error}", file=sys.stderr)
        print("Use --show-domains para ver atributos e valores validos.", file=sys.stderr)
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
