from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
from math import exp
from typing import Any


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
                raise ValueError(f"txId {transaction.tx_id!r} was submitted with a different payload")
            return previous[1]

        current_time = max(self.latest_time or transaction.created_at, transaction.created_at)
        cutoff = current_time - LOOKBACK
        active_transactions = [
            item for item in self.transactions if item.created_at > cutoff
        ]
        expired = len(active_transactions) != len(self.transactions)
        active_ids = {item.tx_id for item in active_transactions}
        self.scores = {
            tx_id: value for tx_id, value in self.scores.items() if tx_id in active_ids
        }
        self.transactions = active_transactions
        if expired:
            self._rebuild_graph()
        score = self._score(transaction.from_user, transaction.to_user)

        if transaction.created_at > cutoff:
            self.transactions.append(transaction)
            edge = (transaction.from_user, transaction.to_user)
            self.edge_counts[edge] += 1
            self.graph[transaction.from_user].add(transaction.to_user)
        if transaction.created_at > cutoff:
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
        new_pairs = _new_reachable_pairs(self.graph, source, target)
        convergence = len(_predecessors(self.graph, target)) + len(
            _ancestors(self.graph, source) & _ancestors(self.graph, target)
        )
        circulation_capacity = _scc_capacity(self.graph, source, target)
        repeated_edge = self.edge_counts[(source, target)]
        raw_signal = (
            0.2 * new_pairs
            + 0.2 * convergence
            + circulation_capacity
            + 0.05 * repeated_edge
        )
        return round(1.0 - exp(-raw_signal / 8.0), 6)


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
        ip_address=value.get("ipAddress"),
        device_id=value.get("deviceId"),
        payload_signature=json.dumps(value, sort_keys=True, separators=(",", ":")),
    )


def _reachable(graph: dict[str, set[str]], start: str) -> set[str]:
    found: set[str] = set()
    pending = list(graph.get(start, ()))
    while pending:
        node = pending.pop()
        if node not in found:
            found.add(node)
            pending.extend(graph.get(node, ()))
    return found


def _new_reachable_pairs(graph: dict[str, set[str]], source: str, target: str) -> int:
    ancestors = _predecessors(graph, source) | {source}
    descendants = _reachable(graph, target) | {target}
    return sum(
        1
        for ancestor in ancestors
        for descendant in descendants
        if descendant != ancestor and descendant not in _reachable(graph, ancestor)
    )


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


def _predecessors(graph: dict[str, set[str]], node: str) -> set[str]:
    return {
        source
        for source, targets in graph.items()
        if node in targets
    }


def _scc_capacity(graph: dict[str, set[str]], source: str, target: str) -> int:
    augmented = {node: set(targets) for node, targets in graph.items()}
    augmented.setdefault(source, set()).add(target)
    nodes = set(augmented)
    for neighbours in augmented.values():
        nodes.update(neighbours)
    index = 0
    indices: dict[str, int] = {}
    lowlinks: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    capacities: list[int] = []

    def visit(node: str) -> None:
        nonlocal index
        indices[node] = index
        lowlinks[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)
        for neighbour in augmented.get(node, ()):
            if neighbour not in indices:
                visit(neighbour)
                lowlinks[node] = min(lowlinks[node], lowlinks[neighbour])
            elif neighbour in on_stack:
                lowlinks[node] = min(lowlinks[node], indices[neighbour])
        if lowlinks[node] == indices[node]:
            component_size = 0
            while True:
                member = stack.pop()
                on_stack.remove(member)
                component_size += 1
                if member == node:
                    break
            if component_size > 1:
                capacities.append(component_size * (component_size - 1))

    for node in nodes:
        if node not in indices:
            visit(node)
    return sum(capacities)
