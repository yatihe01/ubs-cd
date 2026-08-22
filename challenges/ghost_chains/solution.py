from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any


LOOKBACK = timedelta(hours=24)


@dataclass(frozen=True)
class Transaction:
    tx_id: str
    from_user: str
    to_user: str
    amount: float
    created_at: datetime


class GhostChainsModel:
    def __init__(self) -> None:
        self.transactions: list[Transaction] = []
        self.graph: defaultdict[str, set[str]] = defaultdict(set)
        self.scores: dict[str, tuple[Transaction, float]] = {}
        self.latest_time: datetime | None = None

    def reset(self) -> None:
        self.__init__()

    def process(self, transaction: Transaction) -> float:
        previous = self.scores.get(transaction.tx_id)
        if previous is not None:
            if previous[0] != transaction:
                raise ValueError(f"txId {transaction.tx_id!r} was submitted with a different payload")
            return previous[1]

        current_time = max(self.latest_time or transaction.created_at, transaction.created_at)
        cutoff = current_time - LOOKBACK
        self.transactions = [
            item for item in self.transactions if item.created_at >= cutoff
        ]
        self._rebuild_graph()

        # A late arrival that is already outside the active event-time window
        # cannot change the rolling graph and therefore has no structural delta.
        if transaction.created_at < cutoff:
            score = 0.0
        else:
            score = self._score(transaction.from_user, transaction.to_user)
            self.transactions.append(transaction)
        self.scores[transaction.tx_id] = (transaction, score)
        self.latest_time = current_time
        return score

    def _rebuild_graph(self) -> None:
        self.graph = defaultdict(set)
        for transaction in self.transactions:
            self.graph[transaction.from_user].add(transaction.to_user)

    def _score(self, source: str, target: str) -> float:
        # A repeated edge does not change the Phase 1 topology. Transaction
        # frequency may become a separate signal in a later phase, but it is
        # deliberately not treated as a structural delta here.
        if target in self.graph.get(source, ()):
            return 0.0

        if source == target:
            return 0.3

        ancestors = _reverse_distances(self.graph, source)
        descendants = _shortest_distances(self.graph, target)
        distance_cache: dict[str, dict[str, int]] = {}

        new_reachability = 0.0
        shortened_paths = 0.0
        alternative_routes = 0.0

        # Any route which uses the candidate edge has the shape
        # ancestor -> source -> target -> descendant. Compare that candidate
        # route with the graph before insertion to measure the edge's marginal
        # structural effect rather than matching one named pattern.
        for ancestor, distance_to_source in ancestors.items():
            old_distances = distance_cache.setdefault(
                ancestor, _shortest_distances(self.graph, ancestor)
            )
            for descendant, distance_from_target in descendants.items():
                if ancestor == descendant:
                    # Recurring paths are scored explicitly below; shortest
                    # path distance to oneself is always zero.
                    continue

                candidate_distance = (
                    distance_to_source + 1 + distance_from_target
                )
                old_distance = old_distances.get(descendant)

                if old_distance is None:
                    # Ignore the candidate edge's inevitable direct path so
                    # an isolated transfer remains the lowest-risk baseline.
                    if ancestor != source or descendant != target:
                        new_reachability += 1.0 / candidate_distance
                elif candidate_distance < old_distance:
                    shortened_paths += (
                        old_distance - candidate_distance
                    ) / old_distance
                elif candidate_distance == old_distance:
                    # A second route of equal length is the convergence signal
                    # shown in the Phase 1 examples.
                    alternative_routes += 1.0 / candidate_distance

        return_paths = _path_count(self.graph, target, source)
        score = (
            0.14 * min(new_reachability, 2.5)
            + 0.22 * min(shortened_paths, 2.0)
            + 0.36 * min(alternative_routes, 1.5)
        )

        if return_paths:
            score += 0.42 + 0.1 * min(return_paths - 1, 3)
            # Closing another loop into an already cyclic destination is the
            # multi-loop case: two independent return routes converge there.
            if _path_count(self.graph, target, target):
                score += 0.22

        return round(min(score, 1.0), 6)


def parse_created_at(value: Any) -> datetime:
    if not isinstance(value, str):
        raise ValueError("createdAt must be an ISO 8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("createdAt must be an ISO 8601 timestamp") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def make_transaction(value: Any) -> Transaction:
    if not isinstance(value, dict):
        raise ValueError("each transaction must be an object")
    fields = ("txId", "fromUserId", "toUserId")
    if any(not isinstance(value.get(field), str) or not value[field] for field in fields):
        raise ValueError("txId, fromUserId, and toUserId must be non-empty strings")
    amount = value.get("amount")
    if isinstance(amount, bool) or not isinstance(amount, (int, float)):
        raise ValueError("amount must be a number")
    return Transaction(
        tx_id=value["txId"],
        from_user=value["fromUserId"],
        to_user=value["toUserId"],
        amount=float(amount),
        created_at=parse_created_at(value.get("createdAt")),
    )


def _path_count(graph: dict[str, set[str]], start: str, end: str) -> int:
    count = 0
    pending = [(start, {start})]
    while pending and count < 32:
        node, visited = pending.pop()
        for neighbour in graph.get(node, ()):
            if neighbour == end:
                count += 1
            elif neighbour not in visited:
                pending.append((neighbour, visited | {neighbour}))
    return count


def _shortest_distances(
    graph: dict[str, set[str]], start: str
) -> dict[str, int]:
    distances = {start: 0}
    pending = [start]
    for node in pending:
        next_distance = distances[node] + 1
        for neighbour in graph.get(node, ()):
            if neighbour not in distances:
                distances[neighbour] = next_distance
                pending.append(neighbour)
    return distances


def _reverse_distances(
    graph: dict[str, set[str]], node: str
) -> dict[str, int]:
    reverse: defaultdict[str, set[str]] = defaultdict(set)
    for source, targets in graph.items():
        for target in targets:
            reverse[target].add(source)
    return _shortest_distances(reverse, node)
