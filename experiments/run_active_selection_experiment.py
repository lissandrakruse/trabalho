#!/usr/bin/env python3
"""Avalia a selecao ativa contra baselines em dez consultas agricolas."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app


DEFAULT_CROPS = [
    "apple",
    "banana",
    "blackgram",
    "coconut",
    "kidneybeans",
    "maize",
    "pigeonpeas",
    "pomegranate",
    "rice",
    "watermelon",
]
REFERENCE_CONDITIONS = [
    {"attribute": "ph", "value": "acido"},
    {"attribute": "rainfall", "value": "alto"},
]


def parser() -> argparse.ArgumentParser:
    created = argparse.ArgumentParser(description="Experimento da selecao ativa de restricoes.")
    created.add_argument("--budget", type=int, default=25)
    created.add_argument("--crop", action="append", dest="crops")
    created.add_argument(
        "--json-output",
        type=Path,
        default=ROOT / "experiments" / "results_active_selection.json",
    )
    created.add_argument(
        "--csv-output",
        type=Path,
        default=ROOT / "experiments" / "results_active_selection.csv",
    )
    return created


def main() -> int:
    args = parser().parse_args()
    crops = args.crops or DEFAULT_CROPS
    rows = []
    for index, crop in enumerate(crops, start=1):
        print(f"[{index}/{len(crops)}] {crop}", flush=True)
        result = app.compute_active_selection(
            {
                "target": {"attribute": "label", "value": crop},
                "conditions": REFERENCE_CONDITIONS,
                "budget": args.budget,
                "minimumLiteralOverlap": 2,
                "maxCandidates": 80,
            }
        )
        active = result["activeSelection"]
        heuristic = result["baselines"]["supportConfidence"]
        random_baseline = result["baselines"]["random"]
        rows.append(
            {
                "crop": crop,
                "empiricalJointZero": result["fullModel"]["queryCompletionWorlds"] > 0,
                "candidates": result["candidatePool"]["evaluated"],
                "selected": result["selectedCount"],
                "baseWidth": result["baseModel"]["width"],
                "activeWidth": active["width"],
                "activeWidthReduction": active["widthReduction"],
                "activeRelativeReduction": active["relativeWidthReduction"],
                "heuristicWidth": heuristic["width"],
                "randomMeanWidth": random_baseline["meanWidth"],
                "fullModelWidth": result["fullModel"]["width"],
                "recoveredFullModelReduction": result["recoveredFullModelReduction"],
                "exactPruningRate": result["solverEffort"]["exactPruningRate"],
                "totalAvoidanceRate": result["solverEffort"]["totalAvoidanceRate"],
                "selectedEndpointLpSolves": result["solverEffort"]["selectedEndpointLpSolves"],
                "durationSeconds": result["durationSeconds"],
                "activeBeatsHeuristic": active["width"] < heuristic["width"] - 1e-10,
                "activeBeatsRandomMean": active["width"] < random_baseline["meanWidth"] - 1e-10,
            }
        )

    aggregate = {
        "queries": len(rows),
        "budget": args.budget,
        "meanBaseWidth": statistics.fmean(row["baseWidth"] for row in rows),
        "meanActiveWidth": statistics.fmean(row["activeWidth"] for row in rows),
        "meanActiveRelativeReduction": statistics.fmean(row["activeRelativeReduction"] for row in rows),
        "meanHeuristicWidth": statistics.fmean(row["heuristicWidth"] for row in rows),
        "meanRandomWidth": statistics.fmean(row["randomMeanWidth"] for row in rows),
        "meanExactPruningRate": statistics.fmean(row["exactPruningRate"] for row in rows),
        "meanTotalAvoidanceRate": statistics.fmean(row["totalAvoidanceRate"] for row in rows),
        "meanRecoveredFullModelReduction": statistics.fmean(row["recoveredFullModelReduction"] for row in rows),
        "activeBeatsHeuristicQueries": sum(row["activeBeatsHeuristic"] for row in rows),
        "activeBeatsRandomMeanQueries": sum(row["activeBeatsRandomMean"] for row in rows),
        "zeroJointQueries": sum(row["empiricalJointZero"] for row in rows),
    }
    document = {
        "experiment": "active constraint selection for interval-valued PLP queries",
        "conditions": REFERENCE_CONDITIONS,
        "method": "greedy maximum endpoint violation with exact endpoint pruning",
        "baselines": ["support x confidence", "mean of five deterministic random samples"],
        "aggregate": aggregate,
        "queries": rows,
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with args.csv_output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(aggregate, ensure_ascii=False, indent=2), flush=True)
    print(f"JSON: {args.json_output.resolve()}", flush=True)
    print(f"CSV: {args.csv_output.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
