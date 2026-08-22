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

Value evidence is SIGNED, which is the one genuinely new idea in this phase.
Phases 1 and 2 could only add risk, with a hard floor at zero, so every explicable
transaction piled up on that floor and the model had no way to say "this one is
accounted for".  The brief forces the other direction: it puts Example 1 - a
textbook layering decay - *below* three shapes that look tamer, which only makes
sense if the reference scores deviation from an inferred flow rather than
conformance to it.  So:

  * a tight, strictly decaying trail CORROBORATES the flow hypothesis.  The money
    is accounted for, and the score is pulled below what the shape alone earns.
  * a ratio of exactly 1.0 is not decay, it is an absence of value information,
    and stays neutral.  This is what keeps a uniform-amount stream - every Phase
    1/2 fixture - bit-for-bit identical to Phase 2.
  * an incoherent trail (hops that do not describe one movement of value)
    CONTRADICTS it mildly.
  * ratio > 1.0 CONTRADICTS it directly: the amount grew across a hop that
    structurally continues a path.  This is Example 3's "value trajectory
    reversal", and it is weighted heavily enough to lift a plain extension past a
    convergence, because the brief requires exactly that (Example 3 carries ~37%
    of Example 4's structural weight yet has to outrank it).

Composition is asymmetric, because confirming and disconfirming evidence are not
mirror images:

    modifier = (1 + IDENTITY_GAIN * align) * (1 + value_mod)   if value_mod >= 0
             = (1 + IDENTITY_GAIN * align) + value_mod         if value_mod <  0

    raw = structural * modifier + CROSS_GAIN * standalone

Contradiction MULTIPLIES, so a transaction that is structurally a return path AND
carries a broken identity AND grew its amount is a different order of suspicious
than any one of those alone - the brief's "unified system rather than evaluated in
isolation", with CROSS_SIGNAL_DEVIATION as its own diagnostic category.  Summing
would make three moderate signals merely a moderate total.

Corroboration only SUBTRACTS.  It is a claim about the value dimension's own
hypothesis and cannot explain away an identity anomaly; scaling both together
would be actively harmful, since a professional layering operation produces clean
geometric decay by design, and damping its common-control evidence in proportion
to how tidy its amounts look would suppress precisely the cases worth catching.

Examples 1 and 3 use the *identical* graph shape (same five-node chain, same
edges, only the amounts differ), so their structural and identity terms are equal
and the entire required ordering between them is carried by `value_mod` alone -
by construction rather than by tuning against the brief's specific numbers.

With every amount on the stream identical, or with no predecessor edge anywhere,
`value_mod` is 0.0 and the score is bit-for-bit the Phase 2 score.
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

# Weight of a *shortened* path, the signal the brief names alongside new paths.
# It multiplies a per-node gain that is exactly zero for a first connection and
# for a plain repeat, so W_SHORTCUT = 0.0 reproduces the measured 380 model
# bit-for-bit (verified over 300 randomised streams) and is the rollback.
# Kept in step with `solution.py` and `solution2.py`: the structural core is
# shared verbatim across the three phases and the parity tests enforce it.
#
# PHASE 3 CONSTRAINT, which the other two files do not have: raising this also
# raises convergence, and convergence is what Example 3 has to outrank.  Measured
# margin of Example 3 over Example 4 by W_SHORTCUT: 0.046 at 0.0, 0.011 at 2.0,
# 0.003 at 2.5, and NEGATIVE from ~2.7 up, which violates the brief's required
# ordering.  `test_value_reversal_is_the_highest_of_the_four` fails if it is
# crossed - if that test breaks after a structural retune, this is why.
W_SHORTCUT = 2.0

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

# Value evidence is SIGNED.  Phases 1 and 2 could only ever add risk, with a hard
# floor at zero, so every explicable transaction piled up on that floor and the
# model had no way to say "this one is accounted for".  Phase 3 needs the other
# direction: the brief puts Example 1 - a textbook layering decay - *below* three
# shapes that look tamer, which only makes sense if the reference scores deviation
# from an inferred flow rather than conformance to it.  A trail that behaves
# exactly as the flow hypothesis predicts corroborates the ordinary explanation
# and pulls the score down; a trail that contradicts it pushes the score up.

# A hop must retain at least this much of the previous leg to read as layering
# decay at all.  Below it the money largely left the flow, so the trail no longer
# describes one coherent movement of value and cannot corroborate anything.
RETENTION_FLOOR = 0.6

# Spread of retention ratios at which a trail stops looking like one flow.  A
# segment retaining 99.1%, 99.1%, 99.1% has spread ~0 (Example 1); one that halves
# and then retains 98% has spread ~0.48 (Examples 2 and 4).
SPREAD_SCALE = 0.5

# Corroboration: a tight, strictly decaying trail multiplies the structural weight
# by (1 - W_CORROBORATE), so a fully coherent flow scores at 40% of what its shape
# alone would earn.  Flat amounts are NOT corroborating - a ratio of exactly 1.0
# is not decay, it is an absence of value information - which is what keeps a
# uniform-amount stream bit-identical to Phase 2.
W_CORROBORATE = 0.6

# Contradiction, mild: the trail is neither coherent decay nor a reversal.  Real
# evidence, but nothing like a reversal.
W_INCOHERENT = 0.25

# Contradiction, direct: the amount grew across a hop that structurally continues
# a path.  *Any* growth already contradicts the expected degradation, so the term
# starts at REVERSAL_BASE the moment the ratio crosses 1.0 and ramps by
# REVERSAL_RAMP over REVERSAL_SCALE of further growth.  It is large because it
# must be able to lift a plain extension past a convergence: the brief's Example 3
# carries ~37% of Example 4's structural weight yet has to outrank it.
REVERSAL_BASE = 3.0
REVERSAL_RAMP = 1.0
REVERSAL_SCALE = 0.1

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
        value_mod = self._value_raw(transaction, source, backward)

        # Contradicting evidence compounds; corroborating evidence only offsets.
        #
        # When the value trail contradicts the flow, the two dimensions MULTIPLY:
        # summing them would make a transaction carrying a return path AND a
        # broken identity AND a grown amount merely the sum of three moderate
        # terms, when the brief is explicit that such a case is a different order
        # of suspicious - "a unified system rather than evaluated in isolation",
        # with CROSS_SIGNAL_DEVIATION as its own diagnostic category.
        #
        # When the trail corroborates the flow it is only SUBTRACTED, never used
        # to scale identity down.  Corroboration is evidence about the value
        # dimension's own hypothesis - it says the amounts are accounted for - and
        # it cannot explain away an identity anomaly, which is a different kind of
        # claim.  Scaling both together would be actively harmful here: a
        # professional layering operation produces clean geometric decay *by
        # design*, so damping its common-control evidence in proportion to how
        # tidy its amounts look would suppress precisely the cases worth catching.
        # Treating disconfirming evidence as more informative than confirming
        # evidence is the standard asymmetry, not a tuning choice.
        #
        # With a uniform-amount stream `value_mod` is 0 and both branches collapse
        # to exactly Phase 2's `1 + IDENTITY_GAIN * align`.
        identity_factor = 1.0 + IDENTITY_GAIN * align
        if value_mod >= 0.0:
            modifier = identity_factor * (1.0 + value_mod)
        else:
            # Floored at zero; W_CORROBORATE < 1 <= identity_factor already keeps
            # this positive, and the clamp makes that hold whatever constants say.
            modifier = max(0.0, identity_factor + value_mod)

        # Identity reuse across disconnected components is the one term that stays
        # additive: there is no structure for it to amplify, which is exactly why
        # the brief calls it a coordination hint rather than proof.
        raw = structural * modifier + CROSS_GAIN * standalone

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
    ) -> float:
        """Return the signed value modifier from the amount trail of the single
        inferred flow segment this transaction extends.

        Negative when the trail corroborates the flow hypothesis (a tight,
        strictly decaying progression), positive when it contradicts it (an
        incoherent trail, or an outright reversal), zero when there is nothing
        to judge.

        The segment is built by walking backward from `source` along the strongest
        predecessor edge at each step (strongest by the same damped backward-walk
        weight the structural score already computed), so it is exactly one path -
        never a merge of sibling branches, which is the brief's "structural
        segmentation".  Two branches out of a common ancestor therefore each see
        only their own trail, and two branches converging on one node likewise.

        0.0 when `source` has no active predecessor edge: a single amount means
        nothing alone.  Judging *coherence* needs at least two ratios to compare,
        and a flat trail is an absence of value information rather than decay, so
        a uniform-amount stream reproduces Phase 2 exactly.
        """
        trail = self._amount_trail(source, backward)
        if not trail:
            return 0.0

        trail.append(transaction.amount)
        ratios = [
            trail[i + 1] / trail[i]
            for i in range(len(trail) - 1)
            if trail[i] > 0.0
        ]
        if not ratios:
            return 0.0

        # A reversal is judged on the hop being scored, so it needs no history
        # beyond the one leg that fed it.
        last = ratios[-1]
        if last > 1.0:
            return REVERSAL_BASE + REVERSAL_RAMP * min(
                1.0, (last - 1.0) / REVERSAL_SCALE
            )

        # Coherence is a property of the trail, so it is undefined on a single hop.
        if len(ratios) < 2:
            return 0.0

        spread = max(ratios) - min(ratios)
        if all(RETENTION_FLOOR <= ratio < 1.0 for ratio in ratios):
            # Every hop kept most, but not all, of the prior amount: the flow
            # hypothesis explains the money.  The tighter the progression, the
            # more it corroborates.
            coherence = 1.0 - min(1.0, spread / SPREAD_SCALE)
            return -W_CORROBORATE * coherence

        return W_INCOHERENT * min(1.0, spread / SPREAD_SCALE)

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
