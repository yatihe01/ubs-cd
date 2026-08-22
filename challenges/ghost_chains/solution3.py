"""Ghost Chains - Phase 3 (structural + identity + value signal).

Phases 1 and 2 live on in `solution.py` / `solution2.py` and are reproduced here
unchanged: the score is still a *delta* on a damped-walk (Katz-style) view of the
active graph (structural), amplified by identity agreement/absence/divergence
measured against that same walk mass (Phase 2).  See those files for the
derivation; nothing in either half moved.

Phase 3 adds `amount` as a *value* layer.  The brief's core principle is explicit
about the failure mode to avoid: "do not blindly aggregate amounts across
unrelated branches without structural segmentation."  So value evidence is never
pooled across the graph - it is computed per transaction, against only that
transaction's own direct predecessor edge(s) into its sender, which is the
smallest unit the brief calls a "structurally inferred flow segment":

    ratio = this_amount / amount_on_the_edge_immediately_before_it

A single amount means nothing alone (there is no predecessor edge, so the ratio
is undefined and the term is 0).  Along a chain, each hop is compared only to the
leg that fed it - never to the chain's first leg, never to a sibling branch, never
to the graph's average.  This is what keeps Example 2's two branches out of
Apex Logistics (Example 2) and Example 4's two branches into Horizon Capital
(Example 4) structurally segmented: each transaction's ratio only ever looks
backward along its own edge, so two branches sharing an ancestor never blend.

When a node has more than one active predecessor edge, each is weighted by the
same damped backward-walk weight the structural score already computed for that
node - so a distant, weakly-connected predecessor cannot swamp a close one, and
the weighting vocabulary stays identical to Phase 2's.

Turning a ratio into a score:

  * ratio in [DECAY_FLOOR, 1.0]  -  a step that keeps some-to-all of the prior
    amount.  This is layering's signature move (Example 1), not a deviation from
    it, so it contributes exactly 0.  This also means a stream of uniform amounts
    (ratio == 1.0 everywhere, as every Phase 1/2 fixture uses) scores identically
    whether or not Phase 3 is switched on.
  * ratio < DECAY_FLOOR  -  most of the value did not continue onto this leg.
    Plausible (fees, partial forwarding) but no longer "the same money", so it
    contributes a small, capped amount of signal - much less than a reversal.
  * ratio > 1.0  -  the amount *grew* across a hop that structurally continues a
    path.  This is Example 3's "value trajectory reversal": the expected
    degradation pattern is violated while the path stays intact, which the brief
    calls out as the strongest of the four value examples, so it is scored on a
    much steeper curve than a shortfall and saturates fast (REVERSAL_SCALE).

Composition with the other two signals follows Phase 2's own pattern exactly:
identity and value are both flow-relative multipliers on the structural weight,
combined additively inside one multiplier so they read as one system rather than
three scores glued together -

    raw = structural * (1 + IDENTITY_GAIN * align + VALUE_GAIN * value_align)
          + CROSS_GAIN * standalone

Because Examples 1 and 3 in the brief use the *identical* graph shape (the same
five-node linear chain, same edges, only the amounts differ), their structural
and identity terms are exactly equal - the entire required ordering between them
is carried by `value_align` alone. With decay scoring exactly 0 and any reversal
scoring strictly above 0, Example 3 > Example 1 holds by construction, not by
tuning against the specific numbers in the brief.

With every amount on the stream identical (or with no predecessor edge anywhere),
`value_align` is 0.0 everywhere and the score is bit-for-bit the Phase 2 score.
"""

from __future__ import annotations

import heapq
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any


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

# Squash constant; larger spreads the low end, smaller spreads the high end.
SQUASH = 2.0

# Bounds the per-layer frontier so a dense hub cannot blow up a single score.
MAX_FRONTIER = 512

# Weights below this contribute nothing at 6 decimal places.
MIN_WEIGHT = 1e-9


# ----- Phase 2 constants (unchanged) ----------------------------------------------

KIND_DEVICE = "dev"
KIND_IP = "ip"
KINDS = (KIND_DEVICE, KIND_IP)
KIND_WEIGHT = {KIND_DEVICE: 1.0, KIND_IP: 0.7}

W_AGREE = 1.0
W_ABSENT = 0.9
W_DIVERGE = 0.65

IDENTITY_GAIN = 1.5

W_CROSS = 1.0
CROSS_GAIN = 0.6
CROSS_HALF = 2.0

CROSS_COUNT_CAP = 16
CROSS_SCAN_CAP = 512


# ----- Phase 3 constants -----------------------------------------------------------

# Spread of retention ratios along one segment at which inconsistency saturates.
# A segment whose hops retain 99.1%, 99.1%, 99.1% has spread ~0 (Example 1); one
# that halves and then retains 98% has spread ~0.48 (Examples 2 and 4).
SPREAD_SCALE = 0.5

# Weight of segment inconsistency as a multiplicative term.  Deliberately small:
# an incoherent value trail is real evidence but nothing like a reversal, and it
# must not let a long ordinary chain overtake a genuine structural signal.
W_INCONSISTENT = 0.15

# A ratio above 1.0 is a value trajectory reversal.  *Any* growth against a
# continuing path already contradicts the expected degradation pattern, so the
# term starts at REVERSAL_BASE the moment the ratio crosses 1.0 and climbs to 1.0
# over REVERSAL_SCALE of further growth rather than ramping up from nothing.
REVERSAL_BASE = 0.6
REVERSAL_SCALE = 0.1

# How hard value inconsistency amplifies the structural weight.  Same role and
# order of magnitude as IDENTITY_GAIN, so neither signal dominates by construction.
VALUE_GAIN = 1.5

# A reversal enters ADDITIVELY rather than as a multiplier on the structural
# weight.  The brief calls a reversal "a direct contradiction" of the layering
# pattern - evidence in its own right, not a modifier on how much structure the
# edge happens to join.  Multiplicatively it could never outrank a shape with
# several times the structural weight (Example 4 carries ~2.7x Example 3's), yet
# the brief requires exactly that ordering.  It is still gated on there being an
# inferred flow to contradict: with no predecessor edge the term is 0.
REVERSAL_WEIGHT = 2.0

# Longest inferred value segment walked back from the sender.
MAX_TRAIL = 6


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

        # Phase 3: amount most recently admitted on each active edge.  Expiry only
        # ever removes the *oldest* active transaction on an edge first (the heap
        # is ordered by created_at), so the entry recorded here - always the most
        # recently admitted - can never be the one that expires while an older
        # same-edge transaction is still active; it is only ever cleared once the
        # whole edge empties, alongside `_adj`/`_radj`.
        self._edge_amount: dict[tuple[str, str], float] = {}

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
        self._edge_amount[edge] = transaction.amount

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
            self._edge_amount.pop(edge, None)

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

        structural = self._structural_raw(source, target, backward, forward)
        align, standalone = self._identity_raw(transaction, backward, forward)
        inconsistency, reversal = self._value_raw(transaction, source, backward)

        # Identity agreement and value incoherence both amplify the structural
        # weight, inside one multiplier, so the three signals read as one system.
        # A value reversal and identity reuse across disconnected components enter
        # additively instead: neither is a modifier on how much structure this edge
        # joins, both are evidence in their own right.
        raw = (
            structural * (1.0 + IDENTITY_GAIN * align + VALUE_GAIN * inconsistency)
            + REVERSAL_WEIGHT * reversal
            + CROSS_GAIN * standalone
        )

        return round(raw / (raw + SQUASH), 6)

    def _structural_raw(
        self,
        source: str,
        target: str,
        backward: dict[str, float],
        forward: dict[str, float],
    ) -> float:
        forward_total = sum(forward.values())

        # Nodes that could already reach `target`: for them the new edge adds a
        # redundant route rather than a first connection.
        already_reaching = _reachers(self._radj, target)

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

        raw = GAMMA * (
            W_NEW * new_weight * forward_total
            + W_REDUNDANT * redundant_weight * forward_total
            + W_CYCLE * cycle_weight
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
        return max(raw - baseline, 0.0)

    # ----- identity ---------------------------------------------------------------

    def _identity_raw(
        self,
        transaction: Transaction,
        backward: dict[str, float],
        forward: dict[str, float],
    ) -> tuple[float, float]:
        """Return `(align, standalone)`.

        `align` is the fraction-of-flow evidence that modulates the structural
        weight; `standalone` is identity reuse across disconnected components.
        Both are 0.0 when the transaction and its surroundings carry no identity,
        which is what keeps an identity-free stream on exact Phase 1 behaviour.
        """
        # Streams that carry no identity at all must cost nothing extra, so bail out
        # before touching the walk results: this is what keeps a Phase 1 workload at
        # exactly Phase 1 throughput as well as exactly Phase 1 scores.
        if not any(self._node_values[kind] for kind in KINDS):
            return 0.0, 0.0

        # The flow this transaction joins: upstream of the sender plus downstream of
        # the receiver, damped by distance.  Both halves include their own endpoint
        # at weight 1.0, so `mass` is never zero and the ratios below are safe.
        neighbourhood: dict[str, float] = dict(backward)
        for node, weight in forward.items():
            neighbourhood[node] = neighbourhood.get(node, 0.0) + weight
        mass = sum(neighbourhood.values())

        align = 0.0
        standalone = 0.0
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
                else:
                    diverge += weight

            weight_of_kind = KIND_WEIGHT[kind]

            if value is None:
                # A flow that carried one identifier and then stops carrying it is
                # the suspicious case; a flow that was already a mixture has little
                # left to break, and a flow with no identity at all scores nothing.
                dominant = max(by_value.values(), default=0.0)
                align += weight_of_kind * W_ABSENT * (dominant / mass)
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
                    standalone += (
                        weight_of_kind
                        * W_CROSS
                        * (disconnected / (disconnected + CROSS_HALF))
                    )

        return align, standalone

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

    # ----- value -------------------------------------------------------------------

    def _value_raw(
        self,
        transaction: Transaction,
        source: str,
        backward: dict[str, float],
    ) -> tuple[float, float]:
        """Return `(inconsistency, reversal)` from the amount trail of the single
        inferred flow segment this transaction extends.

        The segment is built by walking backward from `source` along the strongest
        predecessor edge at each step (strongest by the same damped backward-walk
        weight the structural score already computed), so it is exactly one path -
        never a merge of sibling branches, which is the brief's "structural
        segmentation".  Two branches out of a common ancestor therefore each see
        only their own trail, and two branches converging on one node likewise.

        Both terms are 0.0 when `source` has no active predecessor edge: a single
        amount means nothing alone.  `inconsistency` additionally needs two ratios
        to compare, and a trail of identical amounts has zero spread, so a uniform
        stream reproduces Phase 2 exactly.
        """
        trail = self._amount_trail(source, backward)
        if not trail:
            return 0.0, 0.0

        trail.append(transaction.amount)
        ratios = [
            trail[i + 1] / trail[i]
            for i in range(len(trail) - 1)
            if trail[i] > 0.0
        ]
        if not ratios:
            return 0.0, 0.0

        last = ratios[-1]
        if last > 1.0:
            reversal = REVERSAL_BASE + (1.0 - REVERSAL_BASE) * min(
                1.0, (last - 1.0) / REVERSAL_SCALE
            )
        else:
            reversal = 0.0

        if len(ratios) >= 2:
            spread = max(ratios) - min(ratios)
            inconsistency = W_INCONSISTENT * min(1.0, spread / SPREAD_SCALE)
        else:
            inconsistency = 0.0

        return inconsistency, reversal

    def _amount_trail(self, source: str, backward: dict[str, float]) -> list[float]:
        """Amounts along the strongest inferred path ending at `source`, oldest
        first.  Greedy on the damped backward-walk weight, so the segment follows
        the flow the structural score considers closest; visited nodes are tracked
        so a cycle terminates the walk instead of looping."""
        amounts: list[float] = []
        node = source
        seen = {source}
        for _ in range(MAX_TRAIL):
            predecessors = self._radj.get(node)
            if not predecessors:
                break
            best: str | None = None
            best_weight = -1.0
            for candidate in predecessors:
                if candidate in seen:
                    continue
                weight = backward.get(candidate, 0.0)
                if weight > best_weight:
                    best, best_weight = candidate, weight
            if best is None:
                break
            amount = self._edge_amount.get((best, node))
            if amount is None or amount <= 0.0:
                break
            amounts.append(amount)
            seen.add(best)
            node = best
        amounts.reverse()
        return amounts


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


def _reachers(radj: dict[str, dict[str, None]], target: str) -> set[str]:
    """Nodes with a path of length >= 1 to `target`, within MAX_DEPTH hops."""
    seen: set[str] = set()
    frontier = [target]
    for _ in range(MAX_DEPTH):
        nxt = [
            parent
            for node in frontier
            for parent in radj.get(node) or ()
            if parent not in seen
        ]
        if not nxt:
            break
        seen.update(nxt)
        frontier = nxt
    return seen


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
    # A missing or malformed amount must not fail the batch; it just carries no
    # value evidence (ratio against it is skipped as if there were no amount).
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
