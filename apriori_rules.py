from __future__ import annotations

import math
from itertools import combinations
from typing import Any


Item = tuple[str, str]
Itemset = frozenset[Item]


def _valid_itemset(itemset: Itemset) -> bool:
    """A categorical world cannot contain two values of the same attribute."""
    attributes = [attribute for attribute, _ in itemset]
    return len(attributes) == len(set(attributes))


def _conditions(itemset: Itemset) -> list[dict[str, str]]:
    return [
        {"attribute": attribute, "value": value}
        for attribute, value in sorted(itemset)
    ]


def _weighted_transactions(
    worlds: list[dict[str, Any]],
) -> list[tuple[Itemset, int]]:
    transactions: list[tuple[Itemset, int]] = []
    for world in worlds:
        count = int(world.get("count", 1))
        if count <= 0:
            continue
        items = frozenset(
            (str(attribute), str(value))
            for attribute, value in world["values"].items()
        )
        transactions.append((items, count))
    return transactions


def mine_frequent_itemsets(
    worlds: list[dict[str, Any]],
    total: int,
    min_support: float,
    max_itemset_size: int,
) -> tuple[dict[Itemset, int], dict[str, int]]:
    """Mine weighted categorical transactions with the Apriori algorithm.

    The unique observed worlds form Ω. ``world["count"]`` preserves the
    multiplicity of each transaction, so support is still measured against all
    rows of the original dataset.
    """
    if total <= 0:
        raise ValueError("O total de transacoes deve ser positivo.")
    if not 0 < min_support <= 1:
        raise ValueError("min_support deve estar no intervalo (0, 1].")
    if max_itemset_size < 1:
        raise ValueError("max_itemset_size deve ser pelo menos 1.")

    transactions = _weighted_transactions(worlds)
    minimum_count = max(1, math.ceil((min_support * total) - 1e-12))
    singleton_counts: dict[Item, int] = {}
    for transaction, weight in transactions:
        for item in transaction:
            singleton_counts[item] = singleton_counts.get(item, 0) + weight

    current_level: dict[Itemset, int] = {
        frozenset([item]): count
        for item, count in singleton_counts.items()
        if count >= minimum_count
    }
    frequent: dict[Itemset, int] = dict(current_level)
    candidate_count = len(singleton_counts)
    largest_level = 1 if current_level else 0

    for size in range(2, max_itemset_size + 1):
        previous = set(current_level)
        candidates: set[Itemset] = set()
        previous_list = sorted(previous, key=lambda itemset: sorted(itemset))
        for left_index, left in enumerate(previous_list):
            for right in previous_list[left_index + 1 :]:
                candidate = left | right
                if len(candidate) != size or not _valid_itemset(candidate):
                    continue
                if all(
                    frozenset(subset) in previous
                    for subset in combinations(candidate, size - 1)
                ):
                    candidates.add(candidate)

        candidate_count += len(candidates)
        if not candidates:
            break

        counts = {candidate: 0 for candidate in candidates}
        for transaction, weight in transactions:
            for candidate in candidates:
                if candidate.issubset(transaction):
                    counts[candidate] += weight

        current_level = {
            candidate: count
            for candidate, count in counts.items()
            if count >= minimum_count
        }
        if not current_level:
            break
        frequent.update(current_level)
        largest_level = size

    return frequent, {
        "minimumCount": minimum_count,
        "candidateItemsets": candidate_count,
        "frequentItemsets": len(frequent),
        "largestFrequentItemset": largest_level,
    }


def generate_rules(
    frequent_itemsets: dict[Itemset, int],
    total: int,
    min_confidence: float,
) -> list[dict[str, Any]]:
    """Generate singleton-consequent rules from Apriori frequent itemsets.

    Support and confidence are probabilities used by the LP constraints. Lift
    remains descriptive metadata; it is deliberately not a filter, ranking
    score, classification metric, or LP coefficient.
    """
    if not 0 <= min_confidence <= 1:
        raise ValueError("min_confidence deve estar no intervalo [0, 1].")

    rules: list[dict[str, Any]] = []
    for itemset, itemset_count in frequent_itemsets.items():
        if len(itemset) < 2:
            continue
        for consequent_item in sorted(itemset):
            consequent = frozenset([consequent_item])
            antecedent = itemset - consequent
            antecedent_count = frequent_itemsets.get(antecedent)
            consequent_count = frequent_itemsets.get(consequent)
            if not antecedent_count or not consequent_count:
                continue

            support = itemset_count / total
            antecedent_support = antecedent_count / total
            consequent_support = consequent_count / total
            confidence = itemset_count / antecedent_count
            if confidence + 1e-12 < min_confidence:
                continue
            lift = confidence / consequent_support if consequent_support > 0 else None
            rules.append(
                {
                    "antecedent": _conditions(antecedent),
                    "consequent": _conditions(consequent),
                    "support": support,
                    "confidence": confidence,
                    "lift": lift,
                    "supportCount": itemset_count,
                    "antecedentSupport": antecedent_support,
                    "consequentSupport": consequent_support,
                    "itemsetSize": len(itemset),
                    "source": "regra gerada pelo algoritmo Apriori",
                }
            )

    # Stable presentation order only. No measure is used as an efficacy score.
    rules.sort(
        key=lambda rule: (
            tuple(
                (condition["attribute"], condition["value"])
                for condition in rule["antecedent"]
            ),
            tuple(
                (condition["attribute"], condition["value"])
                for condition in rule["consequent"]
            ),
        )
    )
    return rules


def mine_apriori_rules(
    worlds: list[dict[str, Any]],
    total: int,
    min_support: float = 0.01,
    min_confidence: float = 0.2,
    max_itemset_size: int = 3,
) -> dict[str, Any]:
    frequent_itemsets, stats = mine_frequent_itemsets(
        worlds,
        total,
        min_support,
        max_itemset_size,
    )
    rules = generate_rules(frequent_itemsets, total, min_confidence)
    return {
        "algorithm": "Apriori",
        "omegaWorlds": len(worlds),
        "transactions": total,
        "minSupport": min_support,
        "minConfidence": min_confidence,
        "maxItemsetSize": max_itemset_size,
        "rules": rules,
        "ruleCount": len(rules),
        **stats,
    }
