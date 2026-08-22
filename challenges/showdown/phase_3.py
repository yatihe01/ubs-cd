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

from collections import OrderedDict
from itertools import product
from threading import RLock
from typing import Any, Callable

from challenges.showdown import phase_1, phase_2


DECK = phase_2.DECK
NUMBERS = phase_2.NUMBERS
TARGET_DELTA = 10

EXPLORE_MIN_CONFIDENCE = phase_2.EXPLORE_MIN_CONFIDENCE
EXPLORE_CALL_CAP = 5
POT_FRACTIONS = (0.40, 0.65, 0.90)

VALUE_BET_MIN = 0.52
VALUE_RAISE_MIN = 0.58
NUTS_EQUITY = 0.78
STACK_OFF_EQUITY = 0.70
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


def multiway_equity(
    rule: phase_2.RuleModel,
    number: int,
    community: int | None,
    ranges: list[dict[int, float]],
) -> float:
    """Exact expected pot share against independent weighted ranges.

    For each common board a small polynomial tracks the probability that no
    opponent beats us and exactly ``k`` opponents tie us.  This is O(13*n),
    rather than enumerating 13**n private-number combinations.
    """
    if number not in NUMBERS:
        return 0.0
    if not ranges:
        return 1.0
    communities = (community,) if community in NUMBERS else NUMBERS
    board_total = 0.0
    for board in communities:
        tie_distribution = [1.0]
        for weights in ranges:
            mass = sum(
                max(weight, 0.0)
                for other, weight in weights.items()
                if other in NUMBERS
            )
            if mass <= 0:
                weights = {other: 1.0 for other in NUMBERS}
                mass = float(DECK)
            ahead = tie = 0.0
            for other, weight in weights.items():
                if other not in NUMBERS or weight <= 0:
                    continue
                share = rule.win_share(number, other, board)
                if share == 1.0:
                    ahead += weight / mass
                elif share == 0.5:
                    tie += weight / mass
                # Losing probability simply drops out: we cannot win that branch.
            updated = [0.0] * (len(tie_distribution) + 1)
            for ties, probability in enumerate(tie_distribution):
                updated[ties] += probability * ahead
                updated[ties + 1] += probability * tie
            tie_distribution = updated
        board_total += sum(
            probability / (ties + 1)
            for ties, probability in enumerate(tie_distribution)
        )
    return board_total / len(communities)


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
    other_deltas = [
        phase_1._as_int(player.get("chip_delta")) for player in state.opponents
    ]
    return max(TARGET_DELTA, max(other_deltas, default=-10**9) + 1)


def _endgame_locked(state: TurnState) -> bool:
    """Whether folding out preserves +10 and a strict table lead.

    Current committed chips are treated pessimistically twice: they leave our
    delta and may all land with the current leader.  Future blinds are handled
    the same way.  This does not pretend other players stop playing each other,
    but avoids the common error of banking +10 while another seat is already
    further ahead.
    """
    blind_cost = _future_blind_cost(state)
    at_risk = state.committed_this_hand + blind_cost
    guaranteed = state.chip_delta - at_risk
    leader = max(
        (phase_1._as_int(player.get("chip_delta")) for player in state.opponents),
        default=-10**9,
    )
    return guaranteed >= TARGET_DELTA and guaranteed > leader + at_risk


def _race_pressure(state: TurnState) -> float:
    """Small, late adjustment when merely surviving cannot top the table."""
    deficit = max(0, _required_delta(state) - state.chip_delta)
    hands_left = max(1, state.total_hands - state.hand_number + 1)
    urgency = 1.0 - min(hands_left / max(state.total_hands, 1), 1.0)
    return min(0.12, deficit / max(state.stack_at_hand_start, 1) * urgency)


def _commit_cap(state: TurnState, equity: float, pressure: float) -> int:
    if equity >= NUTS_EQUITY:
        return state.stack
    fraction = 0.48 if equity + pressure >= 0.65 else 0.25
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
) -> dict[int, float]:
    return {
        number: weight
        for number, weight in weights.items()
        if rule.strength(number, state.community) >= price
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
        rational_calling = _calling_range(rule, state, weights, price)
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

    # Five opponents create only 32 call/fold subsets. Evaluating each subset
    # avoids the serious multiway bias from treating an average caller count as
    # though every opponent were still live at showdown.
    ev = 0.0
    for calls in product((False, True), repeat=len(responses)):
        probability = 1.0
        called_ranges: list[dict[int, float]] = []
        called_chips = 0
        for called, (p_fold, their_add, calling) in zip(calls, responses):
            probability *= (1.0 - p_fold) if called else p_fold
            if called:
                called_ranges.append(calling)
                called_chips += their_add
        if probability <= 0:
            continue
        if not called_ranges:
            ev += probability * state.pot
            continue
        final_pot = state.pot + our_add + called_chips
        equity = multiway_equity(
            rule, state.number, state.community, called_ranges
        )
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


def _explore(state: TurnState) -> dict:
    if state.to_call <= 0 and "check" in state.legal:
        return {"action": "check"}
    # A six-way showdown teaches many pairwise constraints, but do not pay a
    # heads-up-sized tuition when several live ranges can still beat us.
    cap = max(2, EXPLORE_CALL_CAP - max(0, len(state.live_opponents) - 2))
    if state.to_call <= cap and "call" in state.legal:
        return {"action": "call"}
    if "fold" in state.legal:
        return {"action": "fold"}
    return state.fallback()


def decide(state: TurnState) -> dict:
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
    if _endgame_locked(state) and state.to_call > 0:
        return {"action": "fold"} if "fold" in state.legal else state.fallback()
    if rule.confidence < EXPLORE_MIN_CONFIDENCE:
        return _explore(state)

    entries = [
        (player, _opponent_for(player), _infer_range(
            state, player, rule, _opponent_for(player)
        ))
        for player in state.live_opponents
    ]
    ranges = [weights for _, _, weights in entries]
    equity = multiway_equity(rule, state.number, state.community, ranges)
    pressure = _race_pressure(state)
    max_add = _commit_cap(state, equity, pressure)
    locked = _endgame_locked(state)

    if state.to_call <= 0:
        check_ev = equity * state.pot
        value_floor = NUTS_EQUITY if locked else VALUE_BET_MIN - pressure
        value = equity >= value_floor
        # Bluffing through several independent opponents is rarely profitable.
        bluff = (
            not locked
            and len(entries) <= 2
            and equity <= 0.10 + pressure
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

    call_ev = equity * (state.pot + state.to_call) - state.to_call
    value = equity >= VALUE_RAISE_MIN - pressure
    if "raise" in state.legal and value and entries:
        best = _best_raise(state, rule, entries, max_add)
        if best and best[0] > max(call_ev, 0.0):
            for _, model, _ in entries:
                model.our_aggressive += 1
            return {"action": "raise", "amount": best[1]}

    risk_premium = 0.015 * max(0, len(entries) - 1)
    thin_stack_off = (
        equity + pressure < STACK_OFF_EQUITY
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
