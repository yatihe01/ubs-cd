"""Shared lexicographic ranking for the cumulative Ghost Chains phases.

The primary ordering is the number of distinct risky evidence occurrences.  The
existing weighted model is retained as a tie-breaker, compressed to ``[0, 0.5)``
so it can never move a transaction into the next occurrence tier.
"""

from __future__ import annotations


# Controls only the display/calibration of the occurrence tiers.
OCCURRENCE_SQUASH = 2.0

# Leave a strict half-tier gap so six-decimal output rounding cannot collapse the
# strongest possible N-occurrence score into the weakest N+1 score.
TIE_BREAK_WIDTH = 0.5


def occurrence_rank_score(
    occurrence_count: int,
    weighted_raw: float,
    *,
    weighted_squash: float,
) -> float:
    """Map ``(occurrence count, weighted severity)`` to a risk score in ``[0, 1)``.

    Any ``n + 1`` occurrence case outranks every ``n`` occurrence case regardless
    of severity.  Within one count tier, the original weighted raw score preserves
    the model's internal ordering.
    """
    if occurrence_count <= 0 or weighted_raw <= 0.0:
        return 0.0

    tie_break = TIE_BREAK_WIDTH * weighted_raw / (weighted_raw + weighted_squash)
    rank_raw = float(occurrence_count) + tie_break
    return round(rank_raw / (rank_raw + OCCURRENCE_SQUASH), 6)


def directed_cycle_rank(
    adjacency: dict[str, dict[str, None]],
    reverse_adjacency: dict[str, dict[str, None]],
    nodes: set[str],
) -> int:
    """Count independent directed cycles inside ``nodes`` via SCC cycle rank.

    Each strongly connected component contributes ``E - V + 1``.  This counts a
    pre-existing loop once without mistaking an acyclic convergence diamond for a
    cycle, as an undirected cycle count would.
    """
    if not nodes:
        return 0

    visited: set[str] = set()
    order: list[str] = []
    for root in nodes:
        if root in visited:
            continue
        visited.add(root)
        stack: list[tuple[str, bool]] = [(root, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                order.append(node)
                continue
            stack.append((node, True))
            for neighbour in adjacency.get(node) or ():
                if neighbour in nodes and neighbour not in visited:
                    visited.add(neighbour)
                    stack.append((neighbour, False))

    assigned: set[str] = set()
    rank = 0
    for root in reversed(order):
        if root in assigned:
            continue
        component: set[str] = {root}
        assigned.add(root)
        stack = [root]
        while stack:
            node = stack.pop()
            for neighbour in reverse_adjacency.get(node) or ():
                if neighbour in nodes and neighbour not in assigned:
                    assigned.add(neighbour)
                    component.add(neighbour)
                    stack.append(neighbour)

        internal_edges = sum(
            1
            for node in component
            for neighbour in adjacency.get(node) or ()
            if neighbour in component
        )
        if internal_edges:
            rank += max(0, internal_edges - len(component) + 1)
    return rank
