"""SHOWDOWN - Phase 3: A Crowded Table.

Phase 3 keeps Phase 2's learnable table rules, but seats six players.  The
important consequence is that heads-up equity cannot be reused: our private
number must survive every live range, and ties split the pot among every tied
winner.  This module computes that joint win share exactly (conditional on the
community number) without enumerating all 13**5 opponent holdings.

Rule knowledge is deliberately shared with :mod:`phase_2`, since the codenames
and their meanings do not change.  Betting reads are kept per opponent name and
also survive legs because the same five personalities return in every leg.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass
from itertools import product
from threading import RLock
from typing import Any, Callable

from challenges.showdown import phase_1, phase_2


#: Identifies the deployed build. Bumped whenever behaviour changes, so that a
#: disappointing replay can be attributed to a version rather than guessed at.
BUILD = "p3-2026-08-22-explore-capped"

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
EXPLORE_STACK_SHARE = 0.10
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

# Thresholds are stated as a fraction of the way from an even split to
# certainty, then converted by ``_relative_floor`` for the live table size.
VALUE_BET_BASE = 0.34
VALUE_RAISE_BASE = 0.44
STACK_OFF_BASE = 0.62
BLUFF_BASE = 0.55          # bluff below this fraction of our fair share

# While the rule is unrecognised, cap what one hand may cost and buy the
# showdowns that identify it.
EXPLORE_MIN_CONFIDENCE = 0.55
EXPLORE_COMMIT_CAP = 10

MAX_OPPONENTS = 32


class OpponentModel(phase_1.OpponentModel):
    """A persistent read on one of the five fixed Phase 3 personalities."""

    def __init__(self) -> None:
        super().__init__()
        self._counted_phase3_hands: set[tuple[str, int, int]] = set()

    def observe(
        self,
        recent_hands: Any,
        seat: Any,
        match_id: Any,
        leg_number: int,
    ) -> None:
        if not isinstance(seat, int):
            return
        match_key = match_id if isinstance(match_id, str) else "_default"
        for hand in phase_1._as_list(recent_hands):
            if not isinstance(hand, dict):
                continue
            hand_number = hand.get("hand_number")
            if not isinstance(hand_number, int):
                continue
            identity = (match_key, leg_number, hand_number)
            if identity in self._counted_phase3_hands:
                continue
            self._counted_phase3_hands.add(identity)
            for entry in phase_1._as_list(hand.get("actions")):
                if isinstance(entry, dict) and entry.get("seat") == seat:
                    self._count(entry.get("action"))


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


_OPPONENTS: "OrderedDict[str, OpponentModel]" = OrderedDict()
_PHASE3_SEEN: set[tuple[str, str, int, int]] = set()
_STATE_LOCK = RLock()


def _opponent_key(player: dict) -> str:
    name = player.get("name")
    return name if isinstance(name, str) and name else f"_seat_{player.get('seat')}"


def _opponent_for(player: dict) -> OpponentModel:
    key = _opponent_key(player)
    with _STATE_LOCK:
        model = _OPPONENTS.get(key)
        if model is None:
            model = OpponentModel()
            _OPPONENTS[key] = model
            while len(_OPPONENTS) > MAX_OPPONENTS:
                _OPPONENTS.popitem(last=False)
        _OPPONENTS.move_to_end(key)
        return model


def reset_models() -> None:
    """Clear Phase 3 reads and the shared table-rule knowledge (for tests)."""
    with _STATE_LOCK:
        _OPPONENTS.clear()
        _PHASE3_SEEN.clear()
    phase_2.reset_models()


def _shown_number(shown: dict, seat: int) -> Any:
    return shown.get(str(seat), shown.get(seat))


def _observe_showdowns(rule: phase_2.RuleModel, state: TurnState) -> None:
    """Turn a multiway result into every comparison the result proves.

    Winners tie one another and beat every shown non-winner.  No ordering can
    be inferred between two losing hands, so those pairs are intentionally
    omitted.
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
        if identity in _PHASE3_SEEN:
            continue
        _PHASE3_SEEN.add(identity)

        seats = [
            p.get("seat") for p in state.players
            if _shown_number(shown, p.get("seat")) in NUMBERS
        ]
        for left_index, left_seat in enumerate(seats):
            left = _shown_number(shown, left_seat)
            for right_seat in seats[left_index + 1:]:
                right = _shown_number(shown, right_seat)
                left_won, right_won = left_seat in winners, right_seat in winners
                if left_won and right_won:
                    result = 0
                elif left_won:
                    result = 1
                elif right_won:
                    result = -1
                else:
                    continue
                rule.observe(community, left, right, result)


def _infer_range(
    state: TurnState,
    player: dict,
    rule: phase_2.RuleModel,
    model: OpponentModel,
) -> dict[int, float]:
    weights = {number: 1.0 for number in NUMBERS}
    open_width = min(max(model.aggression, 0.16), 0.82)
    seat = player.get("seat")
    for entry in state.hand_actions:
        if entry.get("seat") != seat:
            continue
        community = None if entry.get("round") == "pre_reveal" else state.community
        action = entry.get("action")
        if action in phase_1.AGGRESSIVE_ACTIONS:
            keep, softness = open_width, 0.34
        elif action == "call":
            keep, softness = 0.82, 0.52
        else:
            continue
        ranked = sorted(
            NUMBERS,
            key=lambda number: rule.strength(number, community),
            reverse=True,
        )
        strong = set(ranked[:max(1, round(keep * DECK))])
        weights = {
            number: weight * (1.0 if number in strong else softness)
            for number, weight in weights.items()
        }
    return weights


#: Equity is averaged over surviving rule hypotheses.  Capping the count keeps
#: the pre-reveal case (13 boards x N hypotheses x 32 call subsets) inside the
#: 5 second reply budget; the cap only binds very early in a leg, where the
#: exploration clamp is holding the pots small anyway.
EQUITY_MAX_HYPOTHESES = 8


def _scenarios(rule: phase_2.RuleModel, community: int | None) -> list[tuple]:
    """``(weight, hypothesis_or_None, board)`` - the joint we average over.

    The table rule is one unknown truth, not a fresh coin flip per opponent.
    Averaging *equities* across hypotheses keeps that correlation; averaging
    per-pair win probabilities first, as the previous implementation did,
    throws it away and - because it compared floats for equality - scored any
    comparison the hypotheses disagreed on as a certain loss.
    """
    boards = (community,) if community in NUMBERS else NUMBERS
    names: list[str | None] = sorted(rule.candidates)[:EQUITY_MAX_HYPOTHESES]
    if not names:
        # No recognised shape survives.  One probabilistic scenario driven by
        # the empirical ranking beats pretending every hand is a loser.
        names = [None]
    weight = 1.0 / (len(names) * len(boards))
    return [(weight, name, board) for name in names for board in boards]


def _pair_terms(
    rule: phase_2.RuleModel,
    name: str | None,
    board: int,
    number: int,
    weights: dict[int, float],
) -> tuple[float, float]:
    """``(P(we are ahead), P(we tie))`` against one range in one scenario."""
    ahead = tie = mass = 0.0
    if name is None:
        for other, weight in weights.items():
            if other not in NUMBERS or weight <= 0:
                continue
            mass += weight
            win, draw = rule.outcome_probs(number, other, board)
            ahead += weight * win
            tie += weight * draw
    else:
        table = phase_2.RANKS[name][board]
        mine = table[number]
        for other, weight in weights.items():
            if other not in NUMBERS or weight <= 0:
                continue
            mass += weight
            theirs = table[other]
            if mine > theirs:
                ahead += weight
            elif mine == theirs:
                tie += weight
    if mass <= 0:
        return 1.0 / DECK, 1.0 / DECK
    return ahead / mass, tie / mass


def _convolve(terms: list[tuple[float, float]]) -> float:
    """Expected pot share given per-opponent ahead/tie probabilities.

    Tracks P(nobody beats us and exactly k tie us) in O(13*n) rather than
    enumerating 13**n opponent holdings.
    """
    distribution = [1.0]
    for ahead, tie in terms:
        updated = [0.0] * (len(distribution) + 1)
        for ties, probability in enumerate(distribution):
            if probability <= 0.0:
                continue
            updated[ties] += probability * ahead
            updated[ties + 1] += probability * tie
        distribution = updated
    return sum(
        probability / (ties + 1)
        for ties, probability in enumerate(distribution)
    )


def equity_table(
    rule: phase_2.RuleModel,
    number: int,
    community: int | None,
    ranges: list[dict[int, float]],
) -> list[tuple[float, list[tuple[float, float]]]]:
    """Per-scenario terms, built once so call subsets are cheap to score."""
    return [
        (weight, [_pair_terms(rule, name, board, number, w) for w in ranges])
        for weight, name, board in _scenarios(rule, community)
    ]


def subset_equity(table: list, chosen: list[int]) -> float:
    if not chosen:
        return 1.0
    return sum(
        weight * _convolve([terms[index] for index in chosen])
        for weight, terms in table
    )


def multiway_equity(
    rule: phase_2.RuleModel,
    number: int,
    community: int | None,
    ranges: list[dict[int, float]],
) -> float:
    """Expected share of the pot against independent weighted ranges."""
    if number not in NUMBERS:
        return 0.0
    if not ranges:
        return 1.0
    return sum(
        weight * _convolve(terms)
        for weight, terms in equity_table(rule, number, community, ranges)
    )


def fair_share(opponents: int) -> float:
    return 1.0 / (opponents + 1)


def _relative_floor(opponents: int, base: float) -> float:
    """Turn a heads-up style threshold into a multiway one.

    A 0.52 equity share is average heads-up and enormous six-handed, so every
    threshold is expressed as a fraction of the way from an even split to
    certainty instead of as an absolute number.
    """
    fair = fair_share(opponents)
    return fair + base * (1.0 - fair)


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
    plus the current pot is a mathematical rank-one lock.  This is exactly the
    situation the failed replay reached at roughly +450 before giving it back.
    On the final hand opponents no longer have time to consolidate, so only the
    largest current stack plus the pot can catch us.
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
    leader and clears +10, taking another voluntary confrontation has much
    more downside than upside for the actual scoring rule.
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
    """Small, late adjustment when merely surviving cannot top the table."""
    live_delta = state.stack - state.starting_stack
    deficit = max(0, _required_delta(state) - live_delta)
    hands_left = max(1, state.total_hands - state.hand_number + 1)
    urgency = 1.0 - min(hands_left / max(state.total_hands, 1), 1.0)
    return min(0.22, deficit / max(state.starting_stack, 1) * urgency)


def _commit_cap(
    state: TurnState,
    equity: float,
    pressure: float,
    confidence: float = 1.0,
    opponents: int = 5,
    profile: "Profile | None" = None,
) -> int:
    """How many more chips this hand may cost us.

    While the rule is genuinely unknown, showdowns are worth more than the
    pot: every completed hand at a six-seat table proves up to C(k,2) pairwise
    comparisons, whoever won it.  So buy information cheaply rather than
    stacking off on an equity number the model cannot yet justify.
    """
    stack_off = STACK_OFF_BASE if profile is None else profile.stack_off
    cap = EXPLORE_COMMIT_CAP if profile is None else profile.explore_cap
    if confidence < EXPLORE_MIN_CONFIDENCE:
        return max(0, min(state.stack, cap))
    strategic_equity = min(1.0, equity + pressure)
    if strategic_equity >= _relative_floor(opponents, stack_off):
        return state.stack
    if strategic_equity >= _relative_floor(opponents, 0.46):
        fraction = 0.68
    elif strategic_equity >= _relative_floor(opponents, 0.30):
        fraction = 0.42
    else:
        fraction = 0.24
    budget = int(state.stack_at_hand_start * fraction) - state.committed_this_hand
    return max(0, min(budget, state.stack))


def _sizing_menu(state: TurnState, max_add: int) -> list[int]:
    bounds = phase_1._raise_range(state)
    if bounds is None:
        return []
    low, high = bounds
    targets = [
        state.my_bet + max(1, round(fraction * state.pot))
        for fraction in POT_FRACTIONS
    ]
    if max_add >= state.stack:
        targets.append(high)
    menu: list[int] = []
    for target in sorted(targets):
        amount = min(max(target, low), high)
        if amount - state.my_bet <= max_add and amount not in menu:
            menu.append(amount)
    return menu


def _calling_range(
    rule: phase_2.RuleModel,
    state: TurnState,
    weights: dict[int, float],
    price: float,
    live: int = 1,
) -> dict[int, float]:
    """The part of a range that can profitably continue for ``price``.

    ``strength`` is a win share against one opponent.  Someone who still has
    ``live`` players to get through needs roughly ``price ** (1 / live)`` of
    it, so a multiway bet folds out far more than the heads-up bar suggests.
    """
    bar = price ** (1.0 / max(1, live)) if 0.0 < price < 1.0 else price
    return {
        number: weight
        for number, weight in weights.items()
        if rule.strength(number, state.community) >= bar
    }


def _raise_ev(
    state: TurnState,
    amount: int,
    rule: phase_2.RuleModel,
    entries: list[tuple[dict, OpponentModel, dict[int, float]]],
) -> float:
    our_add = max(0, amount - state.my_bet)
    responses: list[tuple[float, int, dict[int, float]]] = []
    for player, model, weights in entries:
        their_bet = phase_1._as_int(player.get("bet_this_round"))
        their_stack = max(0, phase_1._as_int(player.get("stack")))
        their_add = min(max(0, amount - their_bet), their_stack)
        final_if_call = state.pot + our_add + their_add
        price = their_add / final_if_call if final_if_call > 0 else 0.0
        rational_calling = _calling_range(
            rule, state, weights, price, len(entries)
        )
        mass = sum(weights.values()) or 1.0
        rational_fold = 1.0 - sum(rational_calling.values()) / mass
        if player.get("all_in") or their_add <= 0:
            p_fold = 0.0
        else:
            p_fold = min(
                max(0.55 * rational_fold + 0.45 * model.fold_to_bet, 0.02),
                0.92,
            )
        # Even if the rational calling set is empty, the empirical fold model
        # deliberately leaves a little call probability. Use the original
        # range for that noisy/irrational branch instead of an empty range.
        calling = rational_calling or dict(weights)
        responses.append((p_fold, their_add, calling))

    # Scoring every call/fold subset avoids the serious multiway bias from
    # treating an average caller count as though every opponent were live at
    # showdown.  The scenario table is built once and indexed per subset, so
    # the 2**n loop stays cheap even with the hypothesis mixture on top.
    table = equity_table(
        rule, state.number, state.community, [c for _, _, c in responses]
    )
    ev = 0.0
    for calls in product((False, True), repeat=len(responses)):
        probability = 1.0
        chosen: list[int] = []
        called_chips = 0
        for index, (called, (p_fold, their_add, _)) in enumerate(
            zip(calls, responses)
        ):
            probability *= (1.0 - p_fold) if called else p_fold
            if called:
                chosen.append(index)
                called_chips += their_add
        if probability <= 0:
            continue
        if not chosen:
            ev += probability * state.pot
            continue
        final_pot = state.pot + our_add + called_chips
        equity = subset_equity(table, chosen)
        ev += probability * (equity * final_pot - our_add)
    return ev


def _best_raise(
    state: TurnState,
    rule: phase_2.RuleModel,
    entries: list[tuple[dict, OpponentModel, dict[int, float]]],
    max_add: int,
) -> tuple[float, int] | None:
    choices = [
        (_raise_ev(state, amount, rule, entries), amount)
        for amount in _sizing_menu(state, max_add)
    ]
    return max(choices, default=None)


@dataclass(frozen=True)
class Profile:
    """The tunables that differ between "top the table" and "survive the cut"."""

    value_bet: float = VALUE_BET_BASE
    value_raise: float = VALUE_RAISE_BASE
    stack_off: float = STACK_OFF_BASE
    bluff: float = BLUFF_BASE
    explore_cap: int = EXPLORE_COMMIT_CAP
    max_bluff_opponents: int = 2
    pressure: Callable[["TurnState"], float] | None = None
    locked: Callable[["TurnState"], bool] | None = None

    def pressure_for(self, state: "TurnState") -> float:
        return _race_pressure(state) if self.pressure is None else self.pressure(state)

    def locked_for(self, state: "TurnState") -> bool:
        if self.locked is not None:
            return self.locked(state)
        return _endgame_locked(state) or _protect_late_lead(state)


PHASE3 = Profile()


def decide(state: TurnState, profile: Profile = PHASE3) -> dict:
    rule = phase_2._rule_for(state.table_rule)
    # RuleModel is shared with Phase 2, so use its lock while adding evidence.
    with phase_2._STATE_LOCK:
        with _STATE_LOCK:
            _observe_showdowns(rule, state)
    with _STATE_LOCK:
        for player in state.opponents:
            _opponent_for(player).observe(
                state.recent_hands,
                player.get("seat"),
                state.match_id,
                state.leg_number,
            )

    if state.number not in NUMBERS or not state.legal:
        return state.fallback()
    locked = profile.locked_for(state)
    if locked:
        # Rank, not chip EV, is the scoring objective. Once first place is
        # protected, never reopen the door with a "value" bet.
        if state.to_call <= 0 and "check" in state.legal:
            return {"action": "check"}
        if "fold" in state.legal:
            return {"action": "fold"}
        return state.fallback()

    entries = [
        (player, _opponent_for(player), _infer_range(
            state, player, rule, _opponent_for(player)
        ))
        for player in state.live_opponents
    ]
    ranges = [weights for _, _, weights in entries]
    equity = multiway_equity(rule, state.number, state.community, ranges)
    pressure = profile.pressure_for(state)
    strategic_equity = min(1.0, equity + pressure)
    live = len(entries)
    max_add = _commit_cap(
        state, equity, pressure, rule.confidence, live, profile
    )

    if state.to_call <= 0:
        check_ev = equity * state.pot
        value = equity >= _relative_floor(live, profile.value_bet) - pressure
        # Bluffing through several independent opponents is rarely profitable.
        bluff = (
            live <= profile.max_bluff_opponents
            and equity <= profile.bluff * fair_share(live) + pressure
            and all(model.bluff_budget_left for _, model, _ in entries)
        )
        action = "bet" if "bet" in state.legal else (
            "raise" if "raise" in state.legal else None
        )
        if action and entries and (value or bluff):
            best = _best_raise(state, rule, entries, max_add)
            if best and best[0] > check_ev:
                for _, model, _ in entries:
                    model.our_aggressive += 1
                    if bluff and not value:
                        model.our_bluffs += 1
                return {"action": action, "amount": best[1]}
        if "check" in state.legal:
            return {"action": "check"}
        return state.fallback()

    raw_call_ev = equity * (state.pot + state.to_call) - state.to_call
    call_ev = strategic_equity * (state.pot + state.to_call) - state.to_call
    value = equity >= _relative_floor(live, profile.value_raise) - pressure
    if "raise" in state.legal and value and entries:
        best = _best_raise(state, rule, entries, max_add)
        if best and best[0] > max(raw_call_ev, 0.0):
            for _, model, _ in entries:
                model.our_aggressive += 1
            return {"action": "raise", "amount": best[1]}

    risk_premium = 0.015 * max(0, live - 1)
    thin_stack_off = (
        strategic_equity < _relative_floor(live, profile.stack_off)
        and state.stack > 0
        and state.to_call >= 0.9 * state.stack
        and state.hand_number < state.total_hands
    )
    if call_ev > risk_premium * (state.pot + state.to_call) and not thin_stack_off:
        if "call" in state.legal:
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


# Debug/test aliases make it explicit that these are the shared Phase 2 models.
RuleModel = phase_2.RuleModel
HYPOTHESES = phase_2.HYPOTHESES
_rule_for = phase_2._rule_for
rule_snapshot = phase_2.rule_snapshot
