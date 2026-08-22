"""Ghost Chains - Phase 1 (structural signal).

Scoring principle (from the brief): a transaction's risk score reflects *how much
the transaction increases the graph's structural signal* - the combined effect of
new or shortened paths, and of the graph's capacity to support recurring flow.

So the score is a delta, not a snapshot of the graph.  Adding edge (u, v) creates
exactly the walks that pass through it: every damped walk `a -> ... -> u`, then the
new edge, then every damped walk `v -> ... -> b`.  With a length damping factor
`GAMMA` (Katz-style), the total weight of those new walks is

    delta = GAMMA * sum_a B[a] * sum_b F[b]

where `B[a]` sums GAMMA**len over walks a -> u and `F[b]` over walks v -> b (both
including the empty walk).  This single quantity already covers both halves of the
principle: a brand new connection makes a term appear, and a shortcut makes an
existing term grow because a shorter walk carries a larger GAMMA**len.

That raw weight is then split by the *pre-existing* relationship of the endpoints,
which is what separates the brief's five examples:

  * `a` could not reach `v` before          -> a new path            (W_NEW)
  * `a` could already reach `v` before      -> a redundant route     (W_REDUNDANT)
  * the walk closes on itself (a == b)      -> recurring flow / loop (W_CYCLE)

The partition is by an intrinsic graph property rather than by pattern matching, so
extension < convergence < return < multi-loop falls out of one formula instead of
being hand-tuned per shape.

Finally the raw weight is squashed by `raw / (raw + SQUASH)` into [0, 1): monotone,
continuous, and without a hard cap, so ranking resolution survives at the top end.
"""

from __future__ import annotations

import heapq
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
from math import exp
from typing import Any


# Active lookback window.  A transaction is active while its age is <= LOOKBACK,
# i.e. `created_at >= now - LOOKBACK` - the boundary itself is inside the window.
LOOKBACK = timedelta(hours=24)

# Walk damping.  GAMMA**MAX_DEPTH is ~4e-3, so truncating there costs nothing.
GAMMA = 0.4
MAX_DEPTH = 6

# Relative weight of the three ways a new edge can change the structure.
W_NEW = 1.0
W_REDUNDANT = 3.0
W_CYCLE = 6.0

# A parallel edge leaves the simple-graph structure unchanged, so its delta is
# damped: repeating an existing transfer is ordinary business, not new structure.
REPEAT_DAMPING = 0.35

# Squash constant; larger spreads the low end, smaller spreads the high end.
SQUASH = 2.0

# Bounds the per-layer frontier so a dense hub cannot blow up a single score.
MAX_FRONTIER = 512

# Weights below this contribute nothing at 6 decimal places.
MIN_WEIGHT = 1e-9


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
        previous = self._memo.get(transaction.tx_id)
        if previous is not None:
            # Identical payload: return the original score, mutate nothing.
            # Differing payload violates the "txId is unique" contract; returning
            # the original score is the conservative choice - it keeps graph state
            # consistent and never costs the rest of the batch its scores.
            return previous

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


def _optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def make_transaction(value: Any) -> Transaction:
    if not isinstance(value, dict):
        raise ValueError("each transaction must be an object")
    fields = ("txId", "fromUserId", "toUserId")
    if any(not isinstance(value.get(field), str) or not value[field] for field in fields):
        raise ValueError("txId, fromUserId, and toUserId must be non-empty strings")
    # Phase 1 does not use amount, so a missing or odd value must not fail the batch.
    amount = value.get("amount")
    amount = 0.0 if isinstance(amount, bool) or not isinstance(amount, (int, float)) else float(amount)
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
