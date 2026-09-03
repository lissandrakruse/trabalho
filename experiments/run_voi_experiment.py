#!/usr/bin/env python3
"""Executa a demonstracao controlada de VoI sobre todas as culturas."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app
from voi import ARTICLE, build_conditional_plan, subset_value_of_information, value_of_information


def confidence_score(candidate: dict[str, Any]) -> float:
    """Heuristica comparativa: maior mudanca posterior entre realizacoes."""

    baseline = float(candidate["currentQueryProbability"])
    return max(
        (abs(float(outcome["queryProbability"]) - baseline) for outcome in candidate["outcomes"]),
        default=0.0,
    )


def evaluate_crop(
    worlds: list[dict[str, Any]],
    crop: str,
    observables: list[str],
    budget: int,
) -> dict[str, Any]:
    target = {"attribute": "label", "value": crop}
    costs = {attribute: 1.0 for attribute in observables}
    conditional = build_conditional_plan(
        worlds,
        target,
        observables,
        costs,
        budget=budget,
    )

    subset_size = min(budget, len(observables))
    subset_results = [
        subset_value_of_information(worlds, target, subset)
        for subset in itertools.combinations(observables, subset_size)
    ]
    best_static = max(subset_results, key=lambda result: result["voi"])
    random_mean = statistics.fmean(float(result["voi"]) for result in subset_results)

    single_candidates = [
        value_of_information(worlds, target, observable)
        for observable in observables
    ]
    confidence_subset = [
        candidate["observable"]
        for candidate in sorted(
            single_candidates,
            key=lambda candidate: (-confidence_score(candidate), candidate["observable"]),
        )[:subset_size]
    ]
    confidence_result = subset_value_of_information(worlds, target, confidence_subset)

    first_choice = conditional["tree"].get("choice") or {}
    return {
        "crop": crop,
        "initialProbability": conditional["initialQueryProbability"],
        "initialEntropy": conditional["initialEntropy"],
        "greedyConditional": {
            "firstObservable": first_choice.get("observable"),
            "expectedFinalEntropy": conditional["expectedFinalEntropy"],
            "voi": conditional["planVoi"],
            "nodes": conditional["summary"]["nodes"],
        },
        "exhaustiveBestStatic": {
            "observables": best_static["observables"],
            "expectedFinalEntropy": best_static["expectedEntropy"],
            "voi": best_static["voi"],
        },
        "randomStaticMean": {
            "subsetCount": len(subset_results),
            "voi": random_mean,
        },
        "confidenceHeuristicStatic": {
            "observables": confidence_subset,
            "expectedFinalEntropy": confidence_result["expectedEntropy"],
            "voi": confidence_result["voi"],
        },
        "greedyGainOverRandom": conditional["planVoi"] - random_mean,
    }


def build_report(budget: int) -> dict[str, Any]:
    data = app.load_dataset()
    observables = list(data["numericAttributes"])
    crops = list(data["domains"]["label"])
    results = [
        evaluate_crop(data["worlds"], crop, observables, budget)
        for crop in crops
    ]

    def mean(path: tuple[str, ...]) -> float:
        values = []
        for result in results:
            current: Any = result
            for key in path:
                current = current[key]
            values.append(float(current))
        return statistics.fmean(values)

    wins = sum(
        result["greedyConditional"]["voi"] + 1e-12 >= result["randomStaticMean"]["voi"]
        for result in results
    )
    return {
        "experiment": "VoI agricola - demonstracao controlada",
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "article": ARTICLE,
        "dataset": {
            "name": "Crop Recommendation",
            "records": data["total"],
            "observedWorlds": len(data["worlds"]),
            "crops": len(crops),
            "observables": observables,
        },
        "design": {
            "query": "label=cultura (ground e binaria)",
            "utility": "negativo da entropia binaria",
            "budget": budget,
            "costPerObservable": 1.0,
            "methods": [
                "plano condicional guloso da Figura 3",
                "melhor subconjunto estatico por busca exaustiva",
                "media de todos os subconjuntos estaticos (expectativa aleatoria)",
                "subconjunto estatico por heuristica de confianca",
            ],
            "scope": "demonstracao in-sample; nao e validacao agronomica externa",
        },
        "aggregate": {
            "meanInitialEntropy": mean(("initialEntropy",)),
            "meanGreedyConditionalVoi": mean(("greedyConditional", "voi")),
            "meanBestStaticVoi": mean(("exhaustiveBestStatic", "voi")),
            "meanRandomStaticVoi": mean(("randomStaticMean", "voi")),
            "meanConfidenceStaticVoi": mean(("confidenceHeuristicStatic", "voi")),
            "meanGreedyGainOverRandom": mean(("greedyGainOverRandom",)),
            "greedyAtLeastRandomCrops": wins,
            "totalCrops": len(results),
        },
        "results": results,
    }


def write_csv(report: dict[str, Any], path: Path) -> None:
    fieldnames = [
        "crop",
        "initial_probability",
        "initial_entropy",
        "first_observable",
        "greedy_conditional_voi",
        "best_static_observables",
        "best_static_voi",
        "random_static_mean_voi",
        "confidence_observables",
        "confidence_static_voi",
        "greedy_gain_over_random",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for result in report["results"]:
            writer.writerow(
                {
                    "crop": result["crop"],
                    "initial_probability": result["initialProbability"],
                    "initial_entropy": result["initialEntropy"],
                    "first_observable": result["greedyConditional"]["firstObservable"],
                    "greedy_conditional_voi": result["greedyConditional"]["voi"],
                    "best_static_observables": "+".join(result["exhaustiveBestStatic"]["observables"]),
                    "best_static_voi": result["exhaustiveBestStatic"]["voi"],
                    "random_static_mean_voi": result["randomStaticMean"]["voi"],
                    "confidence_observables": "+".join(result["confidenceHeuristicStatic"]["observables"]),
                    "confidence_static_voi": result["confidenceHeuristicStatic"]["voi"],
                    "greedy_gain_over_random": result["greedyGainOverRandom"],
                }
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=int, default=2)
    parser.add_argument("--json", type=Path, default=ROOT / "experiments" / "results_voi.json")
    parser.add_argument("--csv", type=Path, default=ROOT / "experiments" / "results_voi.csv")
    args = parser.parse_args()
    if args.budget < 1 or args.budget > 3:
        parser.error("O experimento controlado aceita orcamento entre 1 e 3.")

    report = build_report(args.budget)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(report, args.csv)
    print(json.dumps(report["aggregate"], ensure_ascii=False, indent=2))
    print(f"JSON: {args.json}")
    print(f"CSV: {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
