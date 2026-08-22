"""SHOWDOWN - Phase 3: A Crowded Table.

Phase 3 seats six players for four 60-hand legs, one opaque ``table_rule`` per
leg, and only pays out when we finish a leg with *strictly* the highest chip
delta.  Two things therefore dominate everything else:

  1. knowing the leg's showdown rule, because equity is meaningless without it,
     and
  2. not busting, because a busted seat cannot top the table.

Rule inference is a soft Bayesian filter over a generated family of ordering
rules, plus an ordering fitted straight to the data for rules the family cannot
express, all backed by the transitive closure of every comparison a showdown
has proven outright.  Nothing is eliminated outright: one mis-attributed
side-pot winner must not be able to destroy the true hypothesis, which is
exactly what a hard candidate set allows.

Betting then works on exact multiway equity - the joint probability that our
number survives *every* live range - never on a heads-up number.  Thresholds
are expressed per opponent (``p ** live``) so that one constant stays correct
whether five seats are in the pot or one.

The strategy layer is deliberately thin.  Everything measured here says the
leverage is in knowing the rule and in not busting, and that each extra layer
of cleverness on top - modelling opponent ranges, searching for a bet size by
EV, capping the strong hands - cost more than it earned.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass
from threading import RLock
from typing import Any

from challenges.showdown import phase_1, phase_2


DECK = 13
NUMBERS = tuple(range(1, DECK + 1))
TARGET_DELTA = 10

# One size for every value bet. Sizing scaled to the edge, and sizing searched
# for by an EV model, both measured worse than a flat fraction of the pot: five
# opponents mean five guesses about how each of them responds, and those errors
# compound into the chosen size faster than any real information does.
BET_FRACTION = 0.60

# Thresholds are per *opponent*: an equity of ``p ** live`` means we beat each
# individual live range with probability p.  Expressed this way one constant
# stays correct whether five seats are in the pot or one.
VALUE_BET_P = 0.72
VALUE_RAISE_P = 0.82
STACK_OFF_P = 0.82

# While the rule is still genuinely unknown every number has the same equity,
# so there is nothing to bet.  Buy a few cheap showdowns instead of donating
# the stack to a coin flip we cannot price.
EXPLORE_MIN_CONFIDENCE = 0.45
EXPLORE_CALL_CAP = 6
EXPLORE_HANDS = 14
# Where a read stops being provisional. Bet sizing rides this continuously, so
# the first hand past the exploring threshold is not played at full size.
CONFIDENT_AT = 0.85

# Chasing the leader is a closing-stretch adjustment. Applied earlier it turns
# every hand into a race that the leader is already winning.
RACE_STARTS_AT = 0.70
MAX_RACE_PRESSURE = 0.12

# Surcharges on the pot odds a call has to beat: for the share of the stack it
# risks, and for how much of the rule is still guesswork.  Both sit on the call
# and only on the call.  That is where a marginal number quietly bleeds a leg
# away; capping the *strong* hands instead measured far worse, because a hand
# that cannot raise properly ends up calling off the same chips with worse odds.
CALL_RISK_PREMIUM = 0.20
CALL_DOUBT_PREMIUM = 0.30

MAX_RULES = 64

# How many surviving rule shapes are carried into an equity calculation. The
# posterior collapses onto a handful within an orbit or two, and whatever mass
# is left over is priced at the symmetric "we know nothing" share instead.
POSTERIOR_LIMIT = 24

# Per-observation likelihood that a hypothesis which mispredicts is still the
# true rule.  Small enough to identify the rule fast, large enough that a
# handful of side-pot artefacts cannot bury it.
EPS_ERROR = 0.05
_LOG_ODDS = math.log(EPS_ERROR / (1.0 - EPS_ERROR))

_PRIMES = frozenset({2, 3, 5, 7, 11, 13})


def _cmp(left: Any, right: Any) -> int:
    return (left > right) - (left < right)


# --------------------------------------------------------------------------
# The hypothesis family
# --------------------------------------------------------------------------
#
# Every rule the guide describes has the same shape: an ordering feature of
# (number, community), optionally overridden by whether the number pairs the
# community, and settled by high card, low card, or a split.  Rather than
# hand-writing the combinations, generate them and drop the duplicates -
# "highest wins" and "furthest below the community, high card" are different
# sentences but the same relation.

_FEATURES = {
    "flat": lambda n, c: 0,
    "near": lambda n, c: -abs(n - c),
    "wrap_near": lambda n, c: -min(abs(n - c), DECK - abs(n - c)),
    "even": lambda n, c: int(n % 2 == 0),
    "parity_match": lambda n, c: int((n - c) % 2 == 0),
    "above": lambda n, c: int(n > c),
    "offset": lambda n, c: (n - c) % DECK,
    "sum_mod": lambda n, c: (n + c) % DECK,
    "prime": lambda n, c: int(n in _PRIMES),
    "mod_three": lambda n, c: n % 3,
    "high_half": lambda n, c: int(2 * n > DECK + 1),
    "central": lambda n, c: -abs(2 * n - DECK - 1),
}

_TIEBREAKS = (("high", 1), ("low", -1), ("split", 0))


def _build_hypotheses() -> "OrderedDict[str, list[list[int]]]":
    """Score tables ``scores[community][number]``, one per distinct relation."""
    built: "OrderedDict[str, list[list[int]]]" = OrderedDict()
    seen: set[tuple] = set()
    for feature_name, feature in _FEATURES.items():
        for direction in (1, -1):
            for pair_mod in (0, 1, -1):
                for tie_name, tie in _TIEBREAKS:
                    scores = [[0] * (DECK + 1) for _ in range(DECK + 1)]
                    for community in NUMBERS:
                        row = scores[community]
                        for number in NUMBERS:
                            row[number] = (
                                pair_mod * 10_000 * int(number == community)
                                + direction * feature(number, community) * 100
                                + tie * number
                            )
                    # Two hypotheses are the same rule when they induce the
                    # same ordering - ties included - for every community.
                    signature = tuple(
                        tuple(
                            sum(
                                1
                                for other in NUMBERS
                                if scores[community][other]
                                < scores[community][number]
                            )
                            for number in NUMBERS
                        )
                        for community in NUMBERS
                    )
                    if signature in seen:
                        continue
                    seen.add(signature)
                    sign = "+" if direction > 0 else "-"
                    name = f"{feature_name}{sign}|pair{pair_mod:+d}|{tie_name}"
                    built[name] = scores
    return built


HYPOTHESIS_SCORES = _build_hypotheses()
HYPOTHESES = tuple(HYPOTHESIS_SCORES)


@dataclass(frozen=True)
class Observation:
    community: int
    left: int
    right: int
    result: int


class RuleModel:
    """Everything one opaque codename has taught us, across legs and matches.

    Codenames outlive the leg they appear in - "the mapping has not changed" -
    so this is deliberately process-global state, not per-match state.
    """

    def __init__(self, codename: str) -> None:
        self.codename = codename
        self.observations: list[Observation] = []
        self.conflicts = 0

        self._exact: dict[tuple[int, int, int], int] = {}
        self._violations: dict[str, int] = dict.fromkeys(HYPOTHESES, 0)
        self._global_edges: dict[tuple[int, int], int] = {}
        self._informative = 0

        self._dirty = True
        self._win: dict[tuple[int, int, int], float] = {}
        self._tie: dict[tuple[int, int, int], float] = {}
        self._confidence = 0.0
        self._sharpness = 0.0
        self._posterior: list[tuple[list[list[int]], float]] = []
        self._posterior_mass = 0.0

    # -- evidence ---------------------------------------------------------

    def observe(self, community: int, left: int, right: int, result: int) -> None:
        if community not in NUMBERS or left not in NUMBERS or right not in NUMBERS:
            return
        result = _cmp(result, 0)
        key = (community, left, right)
        previous = self._exact.get(key)
        if previous is not None:
            if previous != result:
                self.conflicts += 1
            return

        self._exact[key] = result
        self._exact[(community, right, left)] = -result
        self.observations.append(Observation(community, left, right, result))
        self._dirty = True
        if left == right:
            # Equal numbers tie under every deterministic rule: no information.
            return

        self._informative += 1
        for name, scores in HYPOTHESIS_SCORES.items():
            row = scores[community]
            if _cmp(row[left], row[right]) != result:
                self._violations[name] += 1

        if left != community and right != community:
            # Also evidence for a community-independent ordering, which is what
            # rescues us when the true rule falls outside the generated family.
            canonical = (min(left, right), max(left, right))
            self._global_edges[canonical] = self._global_edges.get(canonical, 0) + (
                result if left < right else -result
            )

    # -- inference --------------------------------------------------------

    def _ensure(self) -> None:
        if self._dirty:
            self._rebuild()

    def _closure(self) -> list[list[list[int | None]]]:
        """Per community, the transitive closure of every proven comparison.

        ``a > b`` and ``b >= c`` prove ``a > c`` without another showdown, which
        roughly doubles what a sparse 60-hand leg actually knows.
        """
        closures: list[list[list[int | None]]] = [
            [[None] * (DECK + 1) for _ in range(DECK + 1)] for _ in range(DECK + 1)
        ]
        for community in NUMBERS:
            matrix = closures[community]
            for number in NUMBERS:
                matrix[number][number] = 0
        for (community, left, right), result in self._exact.items():
            closures[community][left][right] = result
        for community in NUMBERS:
            matrix = closures[community]
            for middle in NUMBERS:
                row_middle = matrix[middle]
                for left in NUMBERS:
                    left_middle = matrix[left][middle]
                    if left_middle is None or left_middle < 0:
                        continue
                    row_left = matrix[left]
                    for right in NUMBERS:
                        middle_right = row_middle[right]
                        if middle_right is None or middle_right < 0:
                            continue
                        implied = 1 if (left_middle > 0 or middle_right > 0) else 0
                        current = row_left[right]
                        if current is None or (current == 0 and implied == 1):
                            row_left[right] = implied
                            matrix[right][left] = -implied
        return closures

    def _global_signal(self, left: int, right: int) -> int:
        canonical = (min(left, right), max(left, right))
        signed = self._global_edges.get(canonical, 0)
        result = _cmp(signed, 0)
        return result if left < right else -result

    def _learned_tables(self) -> list[tuple[list[list[int]], int, float]]:
        """Rules the generated family cannot express, read straight off the data.

        Any rule that ranks the thirteen numbers by a fixed permutation - "5
        beats 12 beats 6 ..." with no arithmetic behind it - is invisible to a
        feature-based family but perfectly learnable: order the numbers by how
        often each has been seen to beat the others.

        The ordering is fitted to the observations, so it always fits them.
        Its weight therefore comes from *coverage* - how much of the 78-pair
        ordering the showdowns have actually pinned down - not from its fit.
        """
        matrix = [[None] * (DECK + 1) for _ in range(DECK + 1)]
        for number in NUMBERS:
            matrix[number][number] = 0
        for (left, right), signed in self._global_edges.items():
            result = _cmp(signed, 0)
            matrix[left][right] = result
            matrix[right][left] = -result
        for middle in NUMBERS:
            for left in NUMBERS:
                left_middle = matrix[left][middle]
                if left_middle is None or left_middle < 0:
                    continue
                for right in NUMBERS:
                    middle_right = matrix[middle][right]
                    if middle_right is None or middle_right < 0:
                        continue
                    implied = 1 if (left_middle > 0 or middle_right > 0) else 0
                    current = matrix[left][right]
                    if current is None or (current == 0 and implied == 1):
                        matrix[left][right] = implied
                        matrix[right][left] = -implied

        known = sum(
            1
            for left in NUMBERS
            for right in NUMBERS
            if left < right and matrix[left][right] is not None
        )
        coverage = known / (DECK * (DECK - 1) / 2.0)
        copeland = {
            number: sum(
                matrix[number][other] or 0
                for other in NUMBERS
                if other != number and matrix[number][other] is not None
            )
            for number in NUMBERS
        }
        order = sorted(NUMBERS, key=lambda number: (copeland[number], number))
        rank = {number: index for index, number in enumerate(order)}

        tables: list[tuple[list[list[int]], int, float]] = []
        for pair_mod in (0, 1, -1):
            scores = [[0] * (DECK + 1) for _ in range(DECK + 1)]
            for community in NUMBERS:
                row = scores[community]
                for number in NUMBERS:
                    row[number] = (
                        pair_mod * 10_000 * int(number == community) + rank[number]
                    )
            violations = sum(
                1
                for item in self.observations
                if item.left != item.right
                and _cmp(
                    scores[item.community][item.left],
                    scores[item.community][item.right],
                )
                != item.result
            )
            # An unfalsifiable fit deserves no belief until the data covers it.
            tables.append((scores, violations, coverage ** 2))
        return tables

    def _candidate_weights(self) -> tuple[list[tuple[list[list[int]], float]], float]:
        """Every rule shape still worth carrying, with its posterior weight.

        A shape is scored by how many showdowns it mispredicts, plus - for the
        fitted orderings, which mispredict nothing by construction - a penalty
        in the same units for the belief its prior does not earn.  The best
        penalised score anchors the whole posterior.
        """
        scored: list[tuple[list[list[int]], float]] = []
        for name, violations in self._violations.items():
            scored.append((HYPOTHESIS_SCORES[name], float(violations)))
        for scores, violations, prior in self._learned_tables():
            penalty = -math.log(max(prior, 1e-12)) / -_LOG_ODDS
            scored.append((scores, violations + penalty))

        best = min(cost for _, cost in scored)
        weighted = [
            (scores, math.exp(_LOG_ODDS * (cost - best)))
            for scores, cost in scored
            if math.exp(_LOG_ODDS * (cost - best)) >= 1e-9
        ]
        weighted.sort(key=lambda item: item[1], reverse=True)
        return weighted, best

    def _rebuild(self) -> None:
        closures = self._closure()
        weighted, best = self._candidate_weights()
        total = sum(weight for _, weight in weighted)
        top = max((weight for _, weight in weighted), default=0.0)

        # How much any rule shape deserves to be believed at all.  A best fit
        # that still mispredicts a tenth of the showdowns is describing some
        # other rule, so lean on raw observation instead.
        informative = self._informative
        trust = max(0.0, min(1.0, 1.0 - 3.0 * best / max(informative, 8)))

        win: dict[tuple[int, int, int], float] = {}
        tie: dict[tuple[int, int, int], float] = {}
        resolved = 0
        sharpness = 0.0

        for community in NUMBERS:
            matrix = closures[community]
            copeland = [0] * (DECK + 1)
            known_edges = 0
            for left in NUMBERS:
                row = matrix[left]
                for right in NUMBERS:
                    if left != right and row[right] is not None:
                        copeland[left] += row[right]
                        known_edges += 1
            reliability = min(known_edges / 60.0, 0.70)
            rows = [(scores[community], weight) for scores, weight in weighted]

            for left in NUMBERS:
                win[(community, left, left)] = 0.0
                tie[(community, left, left)] = 1.0
                for right in range(left + 1, DECK + 1):
                    proven = matrix[left][right]
                    if proven is not None:
                        p_win = 1.0 if proven > 0 else 0.0
                        p_tie = 1.0 if proven == 0 else 0.0
                        resolved += 1
                    else:
                        vote_win = vote_tie = 0.0
                        for row, weight in rows:
                            gap = row[left] - row[right]
                            if gap > 0:
                                vote_win += weight
                            elif gap == 0:
                                vote_tie += weight
                        if total > 0:
                            p_win, p_tie = vote_win / total, vote_tie / total
                        else:
                            p_win, p_tie = 0.5, 0.0
                        if trust < 1.0:
                            empirical = self._empirical(
                                left, right, copeland, reliability
                            )
                            p_win = trust * p_win + (1.0 - trust) * empirical
                            p_tie *= trust
                    p_lose = max(0.0, 1.0 - p_win - p_tie)
                    win[(community, left, right)] = p_win
                    tie[(community, left, right)] = p_tie
                    win[(community, right, left)] = p_lose
                    tie[(community, right, left)] = p_tie
                    sharpness += abs(p_win - p_lose)

        pairs = DECK * DECK * (DECK - 1) / 2.0
        self._sharpness = sharpness / pairs
        self._win, self._tie = win, tie
        self._posterior, self._posterior_mass = self._rank_posterior(
            weighted, total, trust
        )
        self._confidence = self._score_confidence(
            self._sharpness,
            top / total if total else 0.0,
            trust,
            resolved / pairs,
        )
        self._dirty = False

    @staticmethod
    def _rank_posterior(
        weighted: list[tuple[list[list[int]], float]],
        total: float,
        trust: float,
    ) -> tuple[list[tuple[list[list[int]], float]], float]:
        """The best-fitting rule shapes, plus how much belief they carry.

        Equity has to be averaged over *whole rules*, not over each comparison
        separately: our uncertainty is about which rule is in force, and that
        one draw applies to every opponent at the table at once.  Averaging the
        comparisons first and multiplying afterwards treats five correlated
        unknowns as independent and understates a good number badly.
        """
        kept_shapes = weighted[:POSTERIOR_LIMIT]
        kept = sum(weight for _, weight in kept_shapes)
        if kept <= 0 or total <= 0:
            return [], 0.0
        posterior = [(scores, weight / kept) for scores, weight in kept_shapes]
        return posterior, min(1.0, kept / total) * trust

    def posterior(self) -> tuple[list[tuple[list[list[int]], float]], float]:
        self._ensure()
        return self._posterior, self._posterior_mass

    def _empirical(
        self, left: int, right: int, copeland: list[int], reliability: float
    ) -> float:
        """Observation-only read for a pair no proof and no hypothesis covers."""
        signal = _cmp(copeland[left], copeland[right])
        if signal == 0:
            signal = self._global_signal(left, right)
            reliability = min(reliability, 0.45)
        if signal == 0:
            return 0.5
        return 0.5 + 0.5 * reliability * signal

    def _score_confidence(
        self, sharpness: float, top: float, trust: float, coverage: float
    ) -> float:
        if self._informative == 0:
            return 0.0
        evidence = min(1.0, 0.30 + self._informative / 16.0)
        family = sharpness * evidence * trust * (0.55 + 0.45 * top)
        # Even with no recognizable rule shape, enough proven comparisons is
        # itself a workable model of the table.
        return max(0.0, min(1.0, max(family, min(0.72, 1.5 * coverage))))

    # -- queries ----------------------------------------------------------

    @property
    def confidence(self) -> float:
        self._ensure()
        return self._confidence

    @property
    def sharpness(self) -> float:
        """How decisive the per-comparison model is, averaged over every pair.

        1.0 once every comparison is settled, 0.0 while they are all coin
        flips.  Equity needs this to know how much of its own answer to trust.
        """
        self._ensure()
        return self._sharpness

    def outcome_probs(
        self, left: int, right: int, community: int | None
    ) -> tuple[float, float]:
        """``(P(left wins), P(tie))`` for one comparison."""
        if left not in NUMBERS or right not in NUMBERS:
            return 0.5, 0.0
        if community not in NUMBERS:
            wins = ties = 0.0
            for board in NUMBERS:
                board_win, board_tie = self.outcome_probs(left, right, board)
                wins += board_win
                ties += board_tie
            return wins / DECK, ties / DECK
        self._ensure()
        key = (community, left, right)
        return self._win.get(key, 0.5), self._tie.get(key, 0.0)

    def win_share(self, left: int, right: int, community: int | None) -> float:
        p_win, p_tie = self.outcome_probs(left, right, community)
        return p_win + 0.5 * p_tie

    def strength(self, number: int, community: int | None) -> float:
        communities = (community,) if community in NUMBERS else NUMBERS
        return sum(
            self.win_share(number, other, board)
            for board in communities
            for other in NUMBERS
        ) / (len(communities) * DECK)


# --------------------------------------------------------------------------
# Opponent reads
# --------------------------------------------------------------------------

class TurnState(phase_1.TurnState):
    """Defensive parsing plus the full table, not one arbitrary opponent."""

    def __init__(self, body: dict) -> None:
        super().__init__(body)
        rule = body.get("table_rule")
        self.table_rule = rule if isinstance(rule, str) and rule else "_unknown"
        self.leg_number = phase_1._as_int(body.get("leg_number"))
        self.total_legs = phase_1._as_int(body.get("total_legs"))
        self.button_seat = body.get("button_seat")
        self.small_blind = max(0, phase_1._as_int(body.get("small_blind"), 1))
        self.big_blind = max(0, phase_1._as_int(body.get("big_blind"), 2))
        self.starting_stack = max(
            1, phase_1._as_int(body.get("starting_stack"), 200)
        )

        self.players = [
            player
            for player in phase_1._as_list(body.get("players"))
            if isinstance(player, dict) and isinstance(player.get("seat"), int)
        ]
        self.me = next(
            (player for player in self.players if player.get("seat") == self.seat),
            self.me,
        )
        self.active_players = [p for p in self.players if not p.get("busted")]
        self.live_opponents = [
            p
            for p in self.active_players
            if p.get("seat") != self.seat and not p.get("folded")
        ]
        self.opponents = [
            p for p in self.players if p.get("seat") != self.seat
        ]

        # Restore the inherited convenience fields using the largest live bet;
        # choosing players[0] is incorrect once action can arrive from any seat.
        self.opponent = self.live_opponents[0] if self.live_opponents else {}
        self.opponent_seat = self.opponent.get("seat")
        self.my_bet = phase_1._as_int(self.me.get("bet_this_round"))
        self.their_bet = max(
            (phase_1._as_int(p.get("bet_this_round")) for p in self.live_opponents),
            default=0,
        )


_RULES: "OrderedDict[str, RuleModel]" = OrderedDict()
_SEEN_HANDS: set[tuple[str, str, int, int]] = set()
_STATE_LOCK = RLock()


def _seed_from_phase_2(model: RuleModel) -> None:
    """Reuse anything Phase 2 already proved about this codename.

    The codename/rule mapping is shared across phases, so a leg whose rule we
    already met starts with a read instead of from scratch.
    """
    try:
        with phase_2._STATE_LOCK:
            earlier = phase_2._RULES.get(model.codename)
            observations = list(earlier.observations) if earlier else []
    except Exception:  # noqa: BLE001 - seeding is an optimisation, not a dependency
        return
    for item in observations:
        model.observe(item.community, item.left, item.right, item.result)


def _rule_for(codename: str) -> RuleModel:
    with _STATE_LOCK:
        model = _RULES.get(codename)
        if model is None:
            model = RuleModel(codename)
            _seed_from_phase_2(model)
            _RULES[codename] = model
            while len(_RULES) > MAX_RULES:
                _RULES.popitem(last=False)
        _RULES.move_to_end(codename)
        return model


def reset_models() -> None:
    """Clear Phase 3 reads and the shared table-rule knowledge (for tests)."""
    with _STATE_LOCK:
        _RULES.clear()
        _SEEN_HANDS.clear()
    phase_2.reset_models()


def rule_snapshot() -> dict[str, dict[str, Any]]:
    """A test/debug-friendly copy; never expose mutable live model state."""
    with _STATE_LOCK:
        return {
            name: {
                "observations": len(model.observations),
                "confidence": model.confidence,
                "conflicts": model.conflicts,
                "misfits": min(model._violations.values()),
                "posterior": len(model._posterior),
            }
            for name, model in _RULES.items()
        }


def _shown_number(shown: dict, seat: Any) -> Any:
    return shown.get(str(seat), shown.get(seat))


def _observe_showdowns(rule: RuleModel, state: TurnState) -> None:
    """Turn a multiway result into every comparison the result proves.

    A winner beat every other shown number outright.  Several winners holding
    *different* numbers means side pots, where "winner" no longer implies "best
    hand", so that hand is dropped rather than guessed at.
    """
    match_key = state.match_id if isinstance(state.match_id, str) else "_default"
    for hand in phase_1._as_list(state.recent_hands):
        if not isinstance(hand, dict):
            continue
        hand_number = hand.get("hand_number")
        community = hand.get("community_number")
        shown = hand.get("shown_numbers")
        winners = {
            seat for seat in phase_1._as_list(hand.get("winners"))
            if isinstance(seat, int)
        }
        if (
            not isinstance(hand_number, int)
            or community not in NUMBERS
            or not isinstance(shown, dict)
            or not winners
        ):
            continue
        identity = (rule.codename, match_key, state.leg_number, hand_number)
        if identity in _SEEN_HANDS:
            continue
        _SEEN_HANDS.add(identity)

        seats = [
            player.get("seat") for player in state.players
            if _shown_number(shown, player.get("seat")) in NUMBERS
        ]
        winning_seats = [seat for seat in seats if seat in winners]
        losing_seats = [seat for seat in seats if seat not in winners]
        if len({_shown_number(shown, seat) for seat in winning_seats}) > 1:
            continue  # side pots: the winners list is not a ranking
        for winner_seat in winning_seats:
            winner = _shown_number(shown, winner_seat)
            for loser_seat in losing_seats:
                rule.observe(community, winner, _shown_number(shown, loser_seat), 1)


# --------------------------------------------------------------------------
# Equity
# --------------------------------------------------------------------------

def _outcome_probs(
    rule: Any, number: int, other: int, board: int
) -> tuple[float, float]:
    """``(P(win), P(tie))``, tolerating a model that only exposes win shares."""
    probs = getattr(rule, "outcome_probs", None)
    if probs is not None:
        return probs(number, other, board)
    share = rule.win_share(number, other, board)
    if number == other or share == 0.5:
        return 0.0, 1.0
    return share, 0.0


def _uniform_range() -> dict[int, float]:
    return {number: 1.0 for number in NUMBERS}


def _range_profile(weights: dict[int, float]) -> dict[int, float]:
    """A range as a probability distribution over the thirteen numbers."""
    usable = {
        number: weight
        for number, weight in weights.items()
        if number in NUMBERS and weight > 0
    }
    mass = sum(usable.values())
    if mass <= 0:
        return {number: 1.0 / DECK for number in NUMBERS}
    return {number: weight / mass for number, weight in usable.items()}


def _pot_share(profiles: list[tuple[float, float]], indices: tuple[int, ...]) -> float:
    """Expected share of the pot given per-opponent (ahead, tie) probabilities.

    A small polynomial tracks the probability that no opponent beats us and
    exactly ``k`` of them tie us, so a split pot is priced at 1/(k+1) instead of
    being rounded to a win or a loss.  O(n) rather than 13**n.
    """
    distribution = [1.0]
    for index in indices:
        ahead, tie = profiles[index]
        updated = [0.0] * (len(distribution) + 1)
        for ties, probability in enumerate(distribution):
            updated[ties] += probability * ahead
            updated[ties + 1] += probability * tie
            # The losing mass simply drops out: we cannot win that branch.
        distribution = updated
    return sum(
        probability / (ties + 1) for ties, probability in enumerate(distribution)
    )


class _EquityEngine:
    """Joint equity of one number against a fixed set of opponent ranges.

    Built once per candidate bet size, then queried for any subset of the
    opponents - which is what makes evaluating all 32 call/fold combinations of
    a five-handed pot affordable.
    """

    def __init__(
        self,
        rule: Any,
        number: int,
        community: int | None,
        ranges: list[dict[int, float]],
    ) -> None:
        self.number = number
        self.count = len(ranges)
        self.boards = (community,) if community in NUMBERS else NUMBERS
        profiles = [_range_profile(weights) for weights in ranges]

        posterior_of = getattr(rule, "posterior", None)
        posterior, mass = posterior_of() if posterior_of else ([], 0.0)
        self.mass = mass if posterior else 0.0
        self.sharpness = getattr(rule, "sharpness", 1.0)
        self.hypothesis_weights = [weight for _, weight in posterior]

        # Per board, per surviving rule shape, per opponent: how often that
        # opponent's range is behind us and how often it is level with us.
        self.by_rule: list[list[list[tuple[float, float]]]] = []
        if self.mass > 0:
            for board in self.boards:
                per_rule = []
                for scores, _ in posterior:
                    row = scores[board]
                    ours = row[number]
                    per_opponent = []
                    for profile in profiles:
                        ahead = tie = 0.0
                        for other, share in profile.items():
                            gap = ours - row[other]
                            if gap > 0:
                                ahead += share
                            elif gap == 0:
                                tie += share
                        per_opponent.append((ahead, tie))
                    per_rule.append(per_opponent)
                self.by_rule.append(per_rule)

        # The same thing under the marginal (comparison-by-comparison) model,
        # which is what carries proven showdowns and off-family rules.
        self.marginal: list[list[tuple[float, float]]] = []
        for board in self.boards:
            per_opponent = []
            for profile in profiles:
                ahead = tie = 0.0
                for other, share in profile.items():
                    p_win, p_tie = _outcome_probs(rule, number, other, board)
                    ahead += share * p_win
                    tie += share * p_tie
                per_opponent.append((ahead, tie))
            self.marginal.append(per_opponent)

    def equity(self, indices: tuple[int, ...]) -> float:
        if self.number not in NUMBERS:
            return 0.0
        if not indices:
            return 1.0
        boards = len(self.boards)
        marginal = sum(
            _pot_share(self.marginal[index], indices) for index in range(boards)
        ) / boards
        # Multiplying undecided comparisons together drives the product towards
        # zero, but an unknown rule does not make our number bad - it makes
        # every number equally likely to be the best one. Fall back on that
        # symmetry in exactly the proportion the model is still undecided.
        fair_share = 1.0 / (len(indices) + 1)
        fallback = self.sharpness * marginal + (1.0 - self.sharpness) * fair_share
        if self.mass <= 0:
            return fallback
        family = 0.0
        for board_index in range(boards):
            per_rule = self.by_rule[board_index]
            for rule_index, weight in enumerate(self.hypothesis_weights):
                family += weight * _pot_share(per_rule[rule_index], indices)
        family /= boards
        return self.mass * family + (1.0 - self.mass) * fallback


def multiway_equity(
    rule: Any,
    number: int,
    community: int | None,
    ranges: list[dict[int, float]],
) -> float:
    """Expected pot share of ``number`` against every live range at once.

    A partially-known rule yields fractional per-comparison probabilities, and
    they must be carried through as such: rounding anything short of proof down
    to "we lose" collapses every equity to zero, which folds every hand for a
    whole leg.
    """
    if number not in NUMBERS:
        return 0.0
    if not ranges:
        return 1.0
    engine = _EquityEngine(rule, number, community, ranges)
    return engine.equity(tuple(range(len(ranges))))


def _per_opponent(equity: float, live: int) -> float:
    """Equity re-expressed as "how often do we beat one of them"."""
    if live <= 0:
        return 1.0
    return max(0.0, min(1.0, equity)) ** (1.0 / live)


# --------------------------------------------------------------------------
# Position, blinds and the endgame
# --------------------------------------------------------------------------

def _next_active(seat: int, active_seats: list[int]) -> int:
    larger = [candidate for candidate in active_seats if candidate > seat]
    return min(larger) if larger else min(active_seats)


def _future_blind_cost(state: TurnState) -> int:
    """Mandatory future bets, rotating the button past busted seats."""
    hands_left = max(state.total_hands - state.hand_number, 0)
    active = sorted(p.get("seat") for p in state.active_players)
    if hands_left <= 0 or state.seat not in active or len(active) <= 1:
        return 0
    if state.button_seat not in active:
        # A malformed button cannot safely be simulated; use a conservative cap.
        return state.big_blind * hands_left
    button = state.button_seat
    cost = 0
    for _ in range(hands_left):
        button = _next_active(button, active)
        small = _next_active(button, active)
        big = _next_active(small, active)
        if state.seat == small:
            cost += state.small_blind
        if state.seat == big:
            cost += state.big_blind
    return cost


def _required_delta(state: TurnState) -> int:
    # Stack is live during the hand whereas chip_delta is frozen at its start.
    leader_stack = max(
        (phase_1._as_int(player.get("stack")) for player in state.opponents),
        default=0,
    )
    required_stack = max(state.starting_stack + TARGET_DELTA, leader_stack + 1)
    return required_stack - state.starting_stack


def _endgame_locked(state: TurnState) -> bool:
    """Whether no opponent can catch us without receiving another voluntary chip.

    With hands left, the worst case is that every opposing stack consolidates
    into one seat.  Therefore owning strictly more than the rest of the table
    plus the current pot is a mathematical rank-one lock.  On the final hand
    opponents no longer have time to consolidate, so only the largest current
    stack plus the pot can catch us.
    """
    blind_cost = _future_blind_cost(state)
    guaranteed_stack = max(0, state.stack - blind_cost)
    guaranteed_delta = guaranteed_stack - state.starting_stack
    opponent_stacks = [
        max(0, phase_1._as_int(player.get("stack")))
        for player in state.opponents
    ]
    hands_left = max(state.total_hands - state.hand_number, 0)
    if hands_left == 0:
        opponent_ceiling = max(opponent_stacks, default=0) + state.pot
    else:
        opponent_ceiling = sum(opponent_stacks) + state.pot + blind_cost
    return (
        guaranteed_delta >= TARGET_DELTA
        and guaranteed_stack > opponent_ceiling
    )


def _protect_late_lead(state: TurnState) -> bool:
    """Practical rank protection in the last orbit.

    Full consolidation is possible in theory but increasingly unlikely with
    only a few deals left.  Once our fold-out stack is already above the live
    leader and clears +10, taking another voluntary confrontation has much more
    downside than upside for the actual scoring rule.
    """
    hands_left = max(state.total_hands - state.hand_number, 0)
    if hands_left > max(3, len(state.active_players)):
        return False
    blind_cost = _future_blind_cost(state)
    guaranteed_stack = max(0, state.stack - blind_cost)
    leader = max(
        (max(0, phase_1._as_int(player.get("stack"))) for player in state.opponents),
        default=0,
    )
    return (
        guaranteed_stack - state.starting_stack >= TARGET_DELTA
        and guaranteed_stack > leader + state.pot + blind_cost
    )


def _race_pressure(state: TurnState) -> float:
    """Top-the-table urgency: surviving in second place scores nothing.

    This is a last-quarter adjustment and nothing more.  Being behind on hand
    13 of 60 is not a reason to gamble - there are forty-odd hands left to win
    the chips back, and the leader has just as long to give them back.  Urgency
    is therefore cubed and clipped to the closing stretch, because a linear
    ramp reaches a third of its value with two thirds of the leg still to play.
    """
    if state.total_hands <= 0:
        return 0.0
    progress = min(1.0, state.hand_number / state.total_hands)
    if progress < RACE_STARTS_AT:
        return 0.0
    urgency = ((progress - RACE_STARTS_AT) / (1.0 - RACE_STARTS_AT)) ** 3
    live_delta = state.stack - state.starting_stack
    deficit = max(0, _required_delta(state) - live_delta)
    return min(
        MAX_RACE_PRESSURE, deficit / max(state.starting_stack, 1) * urgency
    )


# --------------------------------------------------------------------------
# Sizing, commitment and the price of a call
# --------------------------------------------------------------------------

def _belief(confidence: float) -> float:
    """Confidence rescaled to 0 at the exploring threshold and 1 when settled.

    Betting size has to ride this continuously.  A step change at the threshold
    means the first hand that clears it is played with a full-size stack behind
    a read that is barely half formed, which is exactly how a leg gets lost.
    """
    span = CONFIDENT_AT - EXPLORE_MIN_CONFIDENCE
    return max(0.0, min(1.0, (confidence - EXPLORE_MIN_CONFIDENCE) / span))


def _commit_cap(
    state: TurnState,
    equity: float,
    pressure: float,
    live: int,
    confidence: float,
) -> int:
    """Chips we will volunteer this hand, on top of what is already in."""
    if confidence < EXPLORE_MIN_CONFIDENCE:
        # An unpriced pot is a coin flip. Keep the stack for the hands we will
        # be able to price once the leg has taught us the rule.
        return max(0, min(state.stack, EXPLORE_CALL_CAP))
    # A real edge gets the whole stack behind it. Race pressure may widen a
    # value bet; it may never manufacture a shove out of a coin flip, so the
    # all-in test reads the raw equity and never the inflated one.
    if _per_opponent(equity, live) >= STACK_OFF_P:
        return state.stack
    strategic = _per_opponent(min(1.0, equity + pressure), live)
    fraction = 0.40 if strategic >= 0.74 else 0.25
    fraction *= 0.25 + 0.75 * _belief(confidence)
    budget = int(state.stack_at_hand_start * fraction) - state.committed_this_hand
    return max(0, min(budget, state.stack))


def _call_threshold(
    state: TurnState, live: int, confidence: float
) -> float:
    """Equity a call needs, not merely the pot odds.

    Pot odds price the chips.  They do not price the leg: busting forfeits
    every remaining hand and, with it, any chance of topping the table, so
    chips that represent most of the stack are worth more than their face
    value.  The premium grows with the share of the stack at risk and with how
    much of the rule is still guesswork.
    """
    price = state.to_call / max(1, state.pot + state.to_call)
    risk = min(1.0, state.to_call / max(1, state.stack + state.to_call))
    premium = (
        CALL_RISK_PREMIUM * risk * risk
        + CALL_DOUBT_PREMIUM * (1.0 - _belief(confidence)) * risk
    )
    return price + premium


def _bet_size(state: TurnState, max_add: int) -> int | None:
    """What to raise to: one fraction of the pot, bounded by the commitment cap.

    Deliberately not searched for by an EV model.  Such a model has to guess how
    five separate opponents each respond to a number it cannot see, and those
    guesses compound into the chosen size far faster than any real information
    does - it reliably talks itself into the wrong size out of its own error
    term.  A flat fraction cannot.
    """
    bounds = phase_1._raise_range(state)
    if bounds is None or max_add <= 0:
        return None
    low, high = bounds
    reference = max(state.pot, state.big_blind)
    target = state.my_bet + max(1, round(BET_FRACTION * reference))
    amount = min(max(target, low), high)
    if amount - state.my_bet > max_add:
        # Trim to the budget, but never below the legal minimum: a raise we
        # cannot afford to make properly is a raise we should not make.
        amount = min(state.my_bet + max_add, high)
    if amount < low or amount - state.my_bet > max_add:
        return None
    return amount


def _explore(state: TurnState, equity: float, live: int) -> dict:
    """Cheap, information-buying play while the rule is still unpriced.

    Showdowns between the other five seats teach us for free, so this never
    needs to buy information at a bad price - only at a trivial one.
    """
    if state.to_call <= 0:
        if "check" in state.legal:
            return {"action": "check"}
        return state.fallback()
    fair_share = 1.0 / (live + 1)
    price = state.to_call / max(1, state.pot + state.to_call)
    affordable = (
        state.hand_number <= EXPLORE_HANDS
        and state.to_call <= min(EXPLORE_CALL_CAP, state.big_blind)
    )
    if "call" in state.legal and (affordable or price <= max(equity, fair_share)):
        return {"action": "call"}
    if "fold" in state.legal:
        return {"action": "fold"}
    return state.fallback()


# --------------------------------------------------------------------------
# The decision itself
# --------------------------------------------------------------------------

def decide(state: TurnState) -> dict:
    rule = _rule_for(state.table_rule)
    with _STATE_LOCK:
        _observe_showdowns(rule, state)

    if state.number not in NUMBERS or not state.legal:
        return state.fallback()

    if _endgame_locked(state) or _protect_late_lead(state):
        # Rank, not chip EV, is the scoring objective. Once first place is
        # protected, never reopen the door with a "value" bet.
        if state.to_call <= 0 and "check" in state.legal:
            return {"action": "check"}
        if "fold" in state.legal:
            return {"action": "fold"}
        return state.fallback()

    confidence = rule.confidence
    # Every live seat is priced on the full thirteen numbers. Narrowing those
    # ranges from the betting measured much worse than leaving them uniform:
    # the narrowing compounds across a hand, and what it really moves is our
    # own equity, downwards, until every marginal spot folds.
    live = len(state.live_opponents)
    equity = multiway_equity(
        rule, state.number, state.community, [_uniform_range()] * live
    )

    if confidence < EXPLORE_MIN_CONFIDENCE:
        return _explore(state, equity, live)

    pressure = _race_pressure(state)
    strategic_equity = min(1.0, equity + pressure)
    edge = _per_opponent(equity, live)
    strategic_edge = _per_opponent(strategic_equity, live)
    max_add = _commit_cap(state, equity, pressure, live, confidence)

    if state.to_call <= 0:
        action = "bet" if "bet" in state.legal else (
            "raise" if "raise" in state.legal else None
        )
        if action and live and strategic_edge >= VALUE_BET_P:
            amount = _bet_size(state, max_add)
            if amount is not None:
                return {"action": action, "amount": amount}
        if "check" in state.legal:
            return {"action": "check"}
        return state.fallback()

    if "raise" in state.legal and live and strategic_edge >= VALUE_RAISE_P:
        amount = _bet_size(state, max_add)
        if amount is not None:
            return {"action": "raise", "amount": amount}

    thin_stack_off = (
        edge < STACK_OFF_P
        and state.stack > 0
        and state.to_call >= 0.9 * state.stack
        and state.hand_number < state.total_hands
    )
    if (
        strategic_equity >= _call_threshold(state, live, confidence)
        and not thin_stack_off
        and "call" in state.legal
    ):
        return {"action": "call"}
    if "fold" in state.legal:
        return {"action": "fold"}
    return state.fallback()


def move_from_body(body: dict) -> dict:
    """Fail-safe Phase 3 entry point used by the shared Flask route."""
    legal = phase_1._legal_actions(body)
    try:
        state = TurnState(body)
    except Exception:  # noqa: BLE001 - an illegal reply is more costly than a fold
        return phase_1._fallback(
            legal, body.get("min_raise_to"), body.get("max_raise_to")
        )
    try:
        move = decide(state)
    except Exception:  # noqa: BLE001
        move = state.fallback()
    return phase_1.legalise(move, state)
