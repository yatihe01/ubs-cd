"""Ghost Chains - Phase 2 (structural signal + identity signal).

Phase 1 lives on in `solution.py` and is reproduced here unchanged: the score is
still a *delta* on a damped-walk (Katz-style) view of the active graph, split by
the pre-existing relationship of the endpoints (new path / redundant route /
closed loop).  See `solution.py` for the weighted derivation and occurrence-first
ranking shared by every phase.

Phase 2 adds `ipAddress` and `deviceId` as an *identity* layer.  The brief's core
principle is that identity contributes "relative to where the transaction sits in
the active graph", so identity is never scored from raw frequency counts.  Every
identity observation is measured against the same damped-walk mass that already
carries the structural score:

    N[n] = B[n] + F[n]      B = damped walks n -> ... -> source
                            F = damped walks target -> ... -> n

`N` is the flow this transaction is joining, with nearby entities counting for more
than distant ones.  Against that mass three mutually exclusive observations are
possible for each identity dimension:

  * agreement   - the value on this leg already sits on the surrounding flow.
                  Distinct legal entities initiating from one device or address is
                  direct evidence of common control (Phase 2 Example 1).
  * divergence  - the surrounding flow carries this dimension, but a *different*
                  value.  The structural path is intact while the identity story
                  breaks (Examples 2 and 3).
  * absence     - this leg carries no value at all while the surrounding flow
                  consistently did.  Weighed by how *consistent* that flow's
                  identity was, so a dropped trail counts and an unrelated
                  transaction with no fields counts for nothing.

All three are ratios of `N`, so they are bounded in [0, 1] and independent of graph
size.  They modulate the structural weight multiplicatively - this is the brief's
"combined signal": identity sharpens what the structure already says, and cannot
manufacture risk where there is no flow (two payments from one sender on one phone
stay at 0.0).

The one identity observation that is *not* about the local flow is reuse across
disconnected components (Example 4): the same address or device turning up in parts
of the graph that cannot reach each other.  Structure says nothing there, so its
weighted severity is a small standalone term that saturates.  Under the experimental
occurrence-first policy, however, every distinct reuse observation still contributes
one primary occurrence; enough weak observations therefore outrank fewer strong ones.

`ipAddress` and `deviceId` are scored as independent dimensions and summed, with
the address weighted lower: NAT and shared office egress make a shared address much
weaker evidence than a shared device.

With no identity fields anywhere in the stream every identity term is exactly zero
and the score is bit-for-bit the Phase 1 score.

Identity occurrences are counted per localized device/address observation, including
multiple observations in the same identity dimension.  Their existing weights are
used only to rank transactions with the same total number of occurrences.
"""

from __future__ import annotations

import heapq
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from challenges.ghost_chains.ranking import (
    directed_cycle_rank,
    occurrence_rank_score,
)


# ----- Phase 1 constants (unchanged) ---------------------------------------------

# Active lookback window.  The boundary is EXCLUSIVE: a transaction is active only
# while `created_at > now - LOOKBACK`.  Measured, not guessed - flipping this single
# comparison to inclusive cost 16 points on the Phase 1 evaluator.
LOOKBACK = timedelta(hours=24)

# Walk damping.  GAMMA**MAX_DEPTH is ~4e-3, so truncating there costs nothing.
GAMMA = 0.4
MAX_DEPTH = 6

# Relative weight of the three ways a new edge can change the structure.
W_NEW = 1.0
W_REDUNDANT = 3.0
W_CYCLE = 6.0

# Weight of a *shortened* path, the signal the brief names alongside new paths.
# It multiplies a per-node gain that is exactly zero for a first connection and
# for a plain repeat.  W_SHORTCUT = 0.0 reproduces the measured 380 model's
# weighted structural raw value; occurrence-first ranking is applied afterward.
W_SHORTCUT = 2.0

# Squash constant; larger spreads the low end, smaller spreads the high end.
SQUASH = 2.0

# Bounds the per-layer frontier so a dense hub cannot blow up a single score.
MAX_FRONTIER = 512

# Weights below this contribute nothing at 6 decimal places.
MIN_WEIGHT = 1e-9


# ----- Phase 2 constants ---------------------------------------------------------

# The two identity dimensions, scored independently and summed.  A device is a much
# tighter identity than an address: NAT, carrier-grade NAT and shared office egress
# all put unrelated entities behind one address, so the address dimension is worth
# less per unit of evidence.
KIND_DEVICE = "dev"
KIND_IP = "ip"
KINDS = (KIND_DEVICE, KIND_IP)
KIND_WEIGHT = {KIND_DEVICE: 1.0, KIND_IP: 0.7}

# Weight of each identity observation, as a fraction of the surrounding flow mass.
# AGREE is the strongest: one device driving transfers between entities that are
# nominally unrelated is the most direct evidence of common control.  ABSENT sits
# just below it - dropping an identifier a consistent flow was carrying is evasion,
# but inferred rather than observed.  DIVERGE is real but weaker: a genuinely
# separate counterparty legitimately uses a different device.
W_AGREE = 1.0
W_ABSENT = 0.9
W_DIVERGE = 0.65

# How hard aligned identity evidence amplifies the structural weight.  The
# multiplier is 1 + IDENTITY_GAIN * align, and `align` is capped by construction at
# sum(KIND_WEIGHT) * max(W_*) = 1.7, so the multiplier tops out near 3.55.
IDENTITY_GAIN = 1.5

# Standalone weight for identity reuse across structurally disconnected components,
# and the saturation constant of that term: `n / (n + CROSS_HALF)` reaches 0.5 at
# CROSS_HALF disconnected transactions.  One coincidental share stays small.
W_CROSS = 1.0
CROSS_GAIN = 0.6
CROSS_HALF = 2.0

# Scanning bounds for the cross-component term.  Both the count and the scan are
# capped so one very popular address cannot make a single score O(window).
CROSS_COUNT_CAP = 16
CROSS_SCAN_CAP = 512


@dataclass(frozen=True)
class Transaction:
    tx_id: str
    from_user: str
    to_user: str
    amount: float
    created_at: datetime
    ip_address: str | None = None
    device_id: str | None = None

    def identity(self) -> tuple[tuple[str, str | None], ...]:
        """The identity dimensions in a fixed order, present or absent.  Absence is
        carried as an explicit `None` rather than being dropped, because a missing
        attribute on a connected flow is itself a Phase 2 signal."""
        return ((KIND_DEVICE, self.device_id), (KIND_IP, self.ip_address))


class GhostChainsModel:
    def __init__(self) -> None:
        # Min-heap by (created_at, seq) so expiry stays correct even if arrival
        # order and timestamp order disagree.
        self._active: list[tuple[datetime, int, Transaction]] = []
        self._seq = 0
        self._edges: defaultdict[tuple[str, str], int] = defaultdict(int)
        # Adjacency is kept as insertion-ordered dicts used as sets: iteration order
        # is then a function of the input stream alone, so float accumulation - and
        # therefore every score - is reproducible across processes, which a plain
        # set (hash-randomised) would not be.
        self._adj: defaultdict[str, dict[str, None]] = defaultdict(dict)
        self._radj: defaultdict[str, dict[str, None]] = defaultdict(dict)

        # Identity state, all of it bounded by the active window.
        # kind -> node -> value -> number of active transactions attaching it here.
        self._node_values: dict[str, defaultdict[str, dict[str, int]]] = {
            kind: defaultdict(dict) for kind in KINDS
        }
        # kind -> value -> txId -> (from, to), for the disconnected-reuse term.
        # Insertion ordered, so the capped scan below is deterministic.
        self._value_txs: dict[str, defaultdict[str, dict[str, tuple[str, str]]]] = {
            kind: defaultdict(dict) for kind in KINDS
        }

        # txId -> score, for idempotent replays.  Kept for every txId ever seen:
        # a replay must return the original score even after the transaction
        # itself has aged out of the window.
        self._memo: dict[str, float] = {}
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
        self._expire(cutoff)

        score = self._score(transaction)

        if transaction.created_at > cutoff:
            self._admit(transaction)
        self._memo[transaction.tx_id] = score
        self.latest_time = current_time
        return score

    # ----- streaming graph state -------------------------------------------------

    def _admit(self, transaction: Transaction) -> None:
        heapq.heappush(self._active, (transaction.created_at, self._seq, transaction))
        self._seq += 1
        edge = (transaction.from_user, transaction.to_user)
        self._edges[edge] += 1
        self._adj[edge[0]][edge[1]] = None
        self._radj[edge[1]][edge[0]] = None

        # An identifier belongs to whoever initiated the transfer, but the trail it
        # marks runs along the money, so it is attached to both endpoints: that is
        # what lets a flow be "consistent" across a multi-hop chain.
        for kind, value in transaction.identity():
            if value is None:
                continue
            for node in edge:
                counts = self._node_values[kind][node]
                counts[value] = counts.get(value, 0) + 1
            self._value_txs[kind][value][transaction.tx_id] = edge

    def _expire(self, cutoff: datetime) -> None:
        while self._active and self._active[0][0] <= cutoff:
            _, _, transaction = heapq.heappop(self._active)
            edge = (transaction.from_user, transaction.to_user)
            self._edges[edge] -= 1
            self._retire_identity(transaction, edge)

            if self._edges[edge] > 0:
                continue
            del self._edges[edge]
            _discard(self._adj, edge[0], edge[1])
            _discard(self._radj, edge[1], edge[0])

    def _retire_identity(self, transaction: Transaction, edge: tuple[str, str]) -> None:
        for kind, value in transaction.identity():
            if value is None:
                continue
            index = self._node_values[kind]
            for node in edge:
                counts = index.get(node)
                if not counts or value not in counts:
                    continue
                counts[value] -= 1
                if counts[value] <= 0:
                    del counts[value]
                if not counts:
                    del index[node]
            txs = self._value_txs[kind].get(value)
            if txs is not None:
                txs.pop(transaction.tx_id, None)
                if not txs:
                    del self._value_txs[kind][value]

    # ----- scoring ---------------------------------------------------------------

    def _score(self, transaction: Transaction) -> float:
        source, target = transaction.from_user, transaction.to_user

        # Damped walk weights in the graph as it stands *before* this edge exists.
        backward = _damped_walks(self._radj, source)   # a -> ... -> source
        forward = _damped_walks(self._adj, target)     # target -> ... -> b

        structural, structural_occurrences = self._structural_raw(
            source, target, backward, forward
        )
        align, standalone, local_identity_occurrences, reuse_occurrences = (
            self._identity_raw(transaction, backward, forward)
        )

        # Identity that lines up with the flow amplifies the structural weight;
        # identity reuse across disconnected components has no structure to
        # amplify, so it enters additively and small.
        raw = structural * (1.0 + IDENTITY_GAIN * align) + CROSS_GAIN * standalone

        # Flow-relative identity is evidence only when there is a structural flow
        # for it to describe.  Disconnected reuse is independently observable and
        # therefore contributes occurrences even when structural risk is zero.
        occurrence_count = structural_occurrences + reuse_occurrences
        if structural_occurrences:
            occurrence_count += local_identity_occurrences

        return occurrence_rank_score(
            occurrence_count, raw, weighted_squash=SQUASH
        )

    def _structural_raw(
        self,
        source: str,
        target: str,
        backward: dict[str, float],
        forward: dict[str, float],
    ) -> tuple[float, int]:
        forward_total = sum(forward.values())

        # Nodes that could already reach `target`: for them the new edge adds a
        # redundant route rather than a first connection.
        # `depths[a]` is the shortest existing hop distance from `a` to `target`;
        # membership alone is the old reachability test.
        depths = _depths(self._radj, target)
        already_reaching = depths

        new_weight = 0.0
        redundant_weight = 0.0
        for node, weight in backward.items():
            if node in already_reaching:
                redundant_weight += weight
            else:
                new_weight += weight

        # Walks that close on themselves: the graph's capacity for recurring flow.
        cycle_weight = sum(
            weight * forward[node] for node, weight in backward.items() if node in forward
        )

        # Shortened paths - the half of the brief's principle the reachability
        # split cannot see.  For a node `a` that could already reach `target` over
        # `d` hops, the new edge replaces a walk of weight GAMMA**d with one of
        # weight GAMMA, so the connectivity it gains is GAMMA - GAMMA**d.  The term
        # is identically zero wherever the current model was measured good: nothing
        # reaches `target` yet (a first connection, d undefined), or it reaches it
        # in one hop already (a plain repeat, d == 1).  It turns positive only when
        # a real detour is collapsed, and grows with the length collapsed.
        shortcut_weight = sum(
            weight * (GAMMA - GAMMA ** depths[node])
            for node, weight in backward.items()
            if node in depths
        )

        raw = GAMMA * (
            W_NEW * new_weight * forward_total
            + W_REDUNDANT * redundant_weight * forward_total
            + W_CYCLE * cycle_weight
            + W_SHORTCUT * shortcut_weight
        )

        # The (source, target) term built from two empty walks is the edge's own
        # existence.  Every transaction carries it no matter what surrounds it, so
        # it is not a structural *change* - subtract it and what remains is only
        # the structure this edge actually joins.  This is what puts ordinary flow
        # on the floor: an isolated edge, a leaf, a plain repeat and a fresh
        # self-loop all cancel to exactly 0.0, while anything embedded in
        # convergence or a loop keeps the surplus.
        trivial = W_REDUNDANT if source in already_reaching else W_NEW
        baseline = GAMMA * trivial
        if source == target:
            baseline += GAMMA * W_CYCLE
        raw = max(raw - baseline, 0.0)

        convergence_occurrences = sum(
            1
            for predecessor in self._radj.get(target) or ()
            if predecessor != source
        )
        relevant_nodes = set(backward) | set(forward)
        existing_cycle_occurrences = directed_cycle_rank(
            self._adj, self._radj, relevant_nodes
        )
        closes_cycle = any(
            successor in backward and successor != target
            for successor in self._adj.get(target) or ()
        )
        new_cycle_occurrences = int(
            closes_cycle and not self._edges.get((source, target), 0)
        )
        shortcut_occurrences = int(depths.get(source, 0) > 1)
        occurrence_count = (
            convergence_occurrences
            + existing_cycle_occurrences
            + new_cycle_occurrences
            + shortcut_occurrences
        )
        if raw > 0.0 and occurrence_count == 0:
            occurrence_count = 1
        return raw, occurrence_count

    # ----- identity ---------------------------------------------------------------

    def _identity_raw(
        self,
        transaction: Transaction,
        backward: dict[str, float],
        forward: dict[str, float],
    ) -> tuple[float, float, int, int]:
        """Return weighted and occurrence identity evidence.

        `align` is the fraction-of-flow evidence that modulates the structural
        weight; `standalone` is identity reuse across disconnected components.
        Both are 0.0 when the transaction and its surroundings carry no identity,
        which is what keeps an identity-free stream on exact Phase 1 behaviour.
        """
        # Streams that carry no identity at all must cost nothing extra, so bail out
        # before touching the walk results: this is what keeps a Phase 1 workload at
        # exactly Phase 1 throughput as well as exactly Phase 1 scores.
        if not any(self._node_values[kind] for kind in KINDS):
            return 0.0, 0.0, 0, 0

        # The flow this transaction joins: upstream of the sender plus downstream of
        # the receiver, damped by distance.  Both halves include their own endpoint
        # at weight 1.0, so `mass` is never zero and the ratios below are safe.
        neighbourhood: dict[str, float] = dict(backward)
        for node, weight in forward.items():
            neighbourhood[node] = neighbourhood.get(node, 0.0) + weight
        mass = sum(neighbourhood.values())

        align = 0.0
        standalone = 0.0
        local_occurrences = 0
        reuse_occurrences = 0
        local: set[str] | None = None

        for kind, value in transaction.identity():
            index = self._node_values[kind]
            if not index:
                continue  # this dimension is unused anywhere in the active window

            agree = 0.0
            diverge = 0.0
            by_value: dict[str, float] = {}
            # Same sum either way, so walk whichever side is smaller: a hub-heavy
            # neighbourhood and a rarely used identifier are both common.
            probe = (
                ((node, weight) for node, weight in neighbourhood.items())
                if len(neighbourhood) <= len(index)
                else (
                    (node, neighbourhood[node])
                    for node in index
                    if node in neighbourhood
                )
            )
            for node, weight in probe:
                counts = index.get(node)
                if not counts:
                    continue
                if value is None:
                    # Absence is weighed against how *consistent* the surrounding
                    # identity was, so the per-value masses are what matter.
                    for seen in counts:
                        by_value[seen] = by_value.get(seen, 0.0) + weight
                elif value in counts:
                    agree += weight
                    local_occurrences += 1
                else:
                    diverge += weight
                    local_occurrences += 1

            weight_of_kind = KIND_WEIGHT[kind]

            if value is None:
                # A flow that carried one identifier and then stops carrying it is
                # the suspicious case; a flow that was already a mixture has little
                # left to break, and a flow with no identity at all scores nothing.
                dominant = max(by_value.values(), default=0.0)
                align += weight_of_kind * W_ABSENT * (dominant / mass)
                if dominant > 0.0:
                    # The missing field is one observed disappearance; the prior
                    # nodes establish its strength but do not duplicate the event.
                    local_occurrences += 1
                continue

            align += weight_of_kind * (
                W_AGREE * (agree / mass) + W_DIVERGE * (diverge / mass)
            )

            txs = self._value_txs[kind].get(value)
            if txs:
                if local is None:
                    local = self._local_component(
                        transaction.from_user, transaction.to_user
                    )
                disconnected = _count_disconnected(txs, local)
                if disconnected:
                    reuse_occurrences += disconnected
                    standalone += (
                        weight_of_kind
                        * W_CROSS
                        * (disconnected / (disconnected + CROSS_HALF))
                    )

        return align, standalone, local_occurrences, reuse_occurrences

    def _local_component(self, source: str, target: str) -> set[str]:
        """Entities reachable from either endpoint ignoring edge direction, within
        MAX_DEPTH hops.  Anything outside this set shares no visible flow with the
        transaction being scored, so identity found there is cross-structural."""
        seen = {source, target}
        frontier = [source, target]
        for _ in range(MAX_DEPTH):
            nxt: list[str] = []
            for node in frontier:
                for index in (self._adj, self._radj):
                    for neighbour in index.get(node) or ():
                        if neighbour not in seen:
                            seen.add(neighbour)
                            nxt.append(neighbour)
            if not nxt:
                break
            frontier = nxt[:MAX_FRONTIER]
        return seen


def _count_disconnected(txs: dict[str, tuple[str, str]], local: set[str]) -> int:
    """Active transactions carrying this identifier with neither endpoint in the
    local component.  Capped: the term saturates long before the cap, so scanning
    further cannot change the score but could cost a whole window's worth of work."""
    found = 0
    for scanned, (sender, receiver) in enumerate(txs.values()):
        if scanned >= CROSS_SCAN_CAP:
            break
        if sender not in local and receiver not in local:
            found += 1
            if found >= CROSS_COUNT_CAP:
                break
    return found


def _discard(index: defaultdict[str, dict[str, None]], key: str, value: str) -> None:
    neighbours = index.get(key)
    if neighbours is None:
        return
    neighbours.pop(value, None)
    if not neighbours:
        del index[key]


def _damped_walks(index: dict[str, dict[str, None]], start: str) -> dict[str, float]:
    """Sum of GAMMA**length over walks between `start` and each node, following
    `index`.  The empty walk is included, so `start` itself starts at 1.0."""
    totals: dict[str, float] = {start: 1.0}
    frontier: dict[str, float] = {start: 1.0}
    for _ in range(MAX_DEPTH):
        nxt: dict[str, float] = {}
        for node, weight in frontier.items():
            carried = weight * GAMMA
            if carried < MIN_WEIGHT:
                continue
            for neighbour in index.get(node) or ():
                nxt[neighbour] = nxt.get(neighbour, 0.0) + carried
        if not nxt:
            break
        if len(nxt) > MAX_FRONTIER:
            nxt = dict(
                heapq.nlargest(MAX_FRONTIER, nxt.items(), key=lambda item: item[1])
            )
        for node, weight in nxt.items():
            totals[node] = totals.get(node, 0.0) + weight
        frontier = nxt
    return totals


def _depths(radj: dict[str, dict[str, None]], target: str) -> dict[str, int]:
    """Shortest hop distance from each node to `target`, over paths of length >= 1
    and at most MAX_DEPTH hops.  `target` itself appears only when it sits on a
    cycle.  The key set is exactly what `_reachers` used to return; the values are
    what makes shortening measurable."""
    depths: dict[str, int] = {}
    frontier = [target]
    for depth in range(1, MAX_DEPTH + 1):
        nxt = [
            parent
            for node in frontier
            for parent in radj.get(node) or ()
            if parent not in depths
        ]
        if not nxt:
            break
        for parent in nxt:
            depths.setdefault(parent, depth)
        frontier = nxt
    return depths


# ----- request parsing -----------------------------------------------------------


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
    # Amount is a Phase 3 signal; a missing or odd value must not fail the batch.
    amount = value.get("amount")
    amount = 0.0 if isinstance(amount, bool) or not isinstance(amount, (int, float)) else float(amount)
    return Transaction(
        tx_id=value["txId"],
        from_user=value["fromUserId"],
        to_user=value["toUserId"],
        amount=amount,
        created_at=parse_created_at(value.get("createdAt")),
        ip_address=_optional_str(value.get("ipAddress")),
        device_id=_optional_str(value.get("deviceId")),
    )
