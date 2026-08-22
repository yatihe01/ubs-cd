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
            item for item in self.transactions if item.created_at > cutoff
        ]
        self._rebuild_graph()
        score = self._score(transaction.from_user, transaction.to_user)

        if transaction.created_at > cutoff:
            self.transactions.append(transaction)
        self.scores[transaction.tx_id] = (transaction, score)
        self.latest_time = current_time
        return score

    def _rebuild_graph(self) -> None:
        self.graph = defaultdict(set)
        for transaction in self.transactions:
            self.graph[transaction.from_user].add(transaction.to_user)

    def _score(self, source: str, target: str) -> float:
        return_paths = _path_count(self.graph, target, source)
        convergence_paths = sum(
            _path_count(self.graph, ancestor, target)
            for ancestor in _ancestors(self.graph, source)
            if ancestor != source
        )
        score = 0.38 * min(return_paths, 2) + 0.16 * min(convergence_paths, 4)
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


def _ancestors(graph: dict[str, set[str]], node: str) -> set[str]:
    reverse: defaultdict[str, set[str]] = defaultdict(set)
    for source, targets in graph.items():
        for target in targets:
            reverse[target].add(source)
    found: set[str] = set()
    pending = list(reverse.get(node, ()))
    while pending:
        ancestor = pending.pop()
        if ancestor not in found:
            found.add(ancestor)
            pending.extend(reverse.get(ancestor, ()))
    return found


def _common_ancestor_count(graph: dict[str, set[str]], source: str, target: str) -> int:
    return len(_ancestors(graph, source) & _ancestors(graph, target))
