"""Agreement between two annotators: raw agreement and Cohen's kappa (proposal, Section 8.3).

Cohen's kappa corrects raw agreement for the agreement two people would reach by chance:
    kappa = (p_observed - p_chance) / (1 - p_chance)
where p_chance = sum over labels of P(A says label) * P(B says label).
kappa = 1 is perfect agreement, 0 is no better than chance. A common rough reading: above 0.8 is very
good, 0.6-0.8 good, 0.4-0.6 moderate.
"""

from __future__ import annotations

from collections import Counter


def cohen_kappa(a: list[str], b: list[str]) -> float:
    if len(a) != len(b) or not a:
        raise ValueError("need two label lists of the same, non-zero length")
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    chance = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    if chance == 1.0:
        return 1.0
    return (observed - chance) / (1 - chance)


def confusion(a: list[str], b: list[str]) -> dict[str, dict[str, int]]:
    labels = sorted(set(a) | set(b))
    table = {x: {y: 0 for y in labels} for x in labels}
    for x, y in zip(a, b):
        table[x][y] += 1
    return table
