"""Stateful Phase 1 graph scoring for the Ghost Chains challenge."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
from typing import Any


# The exact boundary is active: created_at >= watermark - LOOKBACK.
LOOKBACK = timedelta(hours=24)


@dataclass(frozen=True)
class Transaction:
    tx_id: str
    from_user: str
    to_user: str
    amount: float
    created_at: datetime
    ip_address: Any
    device_id: Any
    payload_signature: str


class GhostChainsModel:
    def __init__(self) -> None:
        self.transactions: list[Transaction] = []
        self.graph: defaultdict[str, set[str]] = defaultdict(set)
        self.edge_counts: defaultdict[tuple[str, str], int] = defaultdict(int)
        self.scores: dict[str, tuple[Transaction, float]] = {}
        self.latest_time: datetime | None = None

    def reset(self) -> None:
        self.__init__()

    def process(self, transaction: Transaction) -> float:
        previous = self.scores.get(transaction.tx_id)
        if previous is not None:
            if previous[0] != transaction:
                raise ValueError(
                    f"txId {transaction.tx_id!r} was submitted with a different payload"
                )
            return previous[1]

        current_time = max(
            self.latest_time or transaction.created_at,
            transaction.created_at,
        )
        cutoff = current_time - LOOKBACK
        active_transactions = [
            item for item in self.transactions if item.created_at >= cutoff
        ]
        active_ids = {item.tx_id for item in active_transactions}
        self.scores = {
            tx_id: value
            for tx_id, value in self.scores.items()
            if tx_id in active_ids
        }
        self.transactions = active_transactions
        self._rebuild_graph()

        # A late arrival outside the event-time window cannot change the
        # active graph, so its structural delta is zero and it is not stored.
        if transaction.created_at < cutoff:
            score = 0.0
        else:
            score = self._score(transaction.from_user, transaction.to_user)
            self.transactions.append(transaction)
            edge = (transaction.from_user, transaction.to_user)
            self.edge_counts[edge] += 1
            self.graph[transaction.from_user].add(transaction.to_user)
            self.scores[transaction.tx_id] = (transaction, score)

        self.latest_time = current_time
        return score

    def _rebuild_graph(self) -> None:
        self.graph = defaultdict(set)
        self.edge_counts = defaultdict(int)
        for transaction in self.transactions:
            edge = (transaction.from_user, transaction.to_user)
            self.edge_counts[edge] += 1
            self.graph[transaction.from_user].add(transaction.to_user)

    def _score(self, source: str, target: str) -> float:
        # A parallel transaction leaves the Phase 1 simple-graph topology
        # unchanged. Frequency can become a separate later-phase signal.
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

        # Every path introduced by the candidate edge has the form
        # ancestor -> source -> target -> descendant. Compare that route with
        # the graph before insertion to measure the edge's marginal effect.
        for ancestor, distance_to_source in ancestors.items():
            old_distances = distance_cache.setdefault(
                ancestor,
                _shortest_distances(self.graph, ancestor),
            )
            for descendant, distance_from_target in descendants.items():
                if ancestor == descendant:
                    # Cycles are scored explicitly below; shortest distance to
                    # the same node is always zero.
                    continue

                candidate_distance = (
                    distance_to_source + 1 + distance_from_target
                )
                old_distance = old_distances.get(descendant)

                if old_distance is None:
                    # Exclude the inevitable direct path so an isolated edge
                    # remains the lowest-risk baseline.
                    if ancestor != source or descendant != target:
                        new_reachability += 1.0 / candidate_distance
                elif candidate_distance < old_distance:
                    shortened_paths += (
                        old_distance - candidate_distance
                    ) / old_distance
                elif candidate_distance == old_distance:
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
            # documented multi-loop signal.
            if _path_count(self.graph, target, target):
                score += 0.22

        return round(min(score, 1.0), 6)


def parse_created_at(value: Any) -> datetime:
    if not isinstance(value, str):
        raise ValueError("createdAt must be an ISO 8601 timestamp")
    text = value.strip()
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError("createdAt must be an ISO 8601 timestamp") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def make_transaction(value: Any) -> Transaction:
    if not isinstance(value, dict):
        raise ValueError("each transaction must be an object")
    fields = ("txId", "fromUserId", "toUserId")
    if any(
        not isinstance(value.get(field), str) or not value[field]
        for field in fields
    ):
        raise ValueError(
            "txId, fromUserId, and toUserId must be non-empty strings"
        )

    # Amount is not a Phase 1 signal. Keep malformed or missing amounts from
    # rejecting an otherwise processable transaction.
    amount = value.get("amount")
    amount = (
        float(amount)
        if not isinstance(amount, bool) and isinstance(amount, (int, float))
        else 0.0
    )
    return Transaction(
        tx_id=value["txId"],
        from_user=value["fromUserId"],
        to_user=value["toUserId"],
        amount=amount,
        created_at=parse_created_at(value.get("createdAt")),
        ip_address=value.get("ipAddress"),
        device_id=value.get("deviceId"),
        payload_signature=json.dumps(value, sort_keys=True, separators=(",", ":")),
    )


def _path_count(
    graph: dict[str, set[str]],
    start: str,
    end: str,
) -> int:
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
    graph: dict[str, set[str]],
    start: str,
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
    graph: dict[str, set[str]],
    node: str,
) -> dict[str, int]:
    reverse: defaultdict[str, set[str]] = defaultdict(set)
    for source, targets in graph.items():
        for target in targets:
            reverse[target].add(source)
    return _shortest_distances(reverse, node)
