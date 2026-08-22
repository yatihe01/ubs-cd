"""SHOWDOWN - Phase 2: Reading the Table.

Phase 2 consists of four independent 40-hand legs.  Each opaque
``table_rule`` names a deterministic showdown rule which is stable across
retries.  This module therefore keeps two deliberately separate kinds of
state:

* rule knowledge is keyed by ``table_rule`` and survives legs/matches;
* betting-style statistics are keyed by the opponent name and are used only
  as a soft range read.

Completed showdowns are hard evidence.  A small ensemble recognizes common
simple rule shapes quickly; if none fits, observed comparisons and a global
pairwise ranking provide a conservative, non-crashing fallback.  Early hands
under an unknown rule play cheaply for information instead of pretending that
the Phase 1 (standard) ordering still applies.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from threading import RLock
from typing import Any, Callable

from challenges.showdown import phase_1


DECK = 13
NUMBERS = tuple(range(1, DECK + 1))
TARGET_DELTA = 25

POT_FRACTIONS = (0.45, 0.70, 1.0)
VALUE_BET_MIN = 0.62
VALUE_RAISE_MIN = 0.68
BLUFF_BET_MAX = 0.20
BLUFF_RAISE_MAX = 0.15
NUTS_EQUITY = 0.91
STACK_OFF_EQUITY = 0.84

# While the rule is genuinely unknown, buy cheap showdowns but preserve enough
# stack to observe the whole leg.  Repeated attempts then turn this exploration
# into permanent codename knowledge.
EXPLORE_MIN_CONFIDENCE = 0.58
EXPLORE_CALL_CAP = 8
EXPLORE_MAX_POT_FRACTION = 0.50

MAX_RULES = 64
MAX_OPPONENTS = 32

Compare = Callable[[int, int, int], int]


def _cmp(left: Any, right: Any) -> int:
    return (left > right) - (left < right)


def _key_standard(number: int, community: int) -> tuple[int, int]:
    return (int(number == community), number)


def _key_pair_low(number: int, community: int) -> tuple[int, int]:
    return (int(number == community), -number)


def _key_high(number: int, community: int) -> tuple[int]:
    return (number,)


def _key_low(number: int, community: int) -> tuple[int]:
    return (-number,)


def _key_closest_high(number: int, community: int) -> tuple[int, int]:
    return (-abs(number - community), number)


def _key_closest_low(number: int, community: int) -> tuple[int, int]:
    return (-abs(number - community), -number)


def _key_farthest_high(number: int, community: int) -> tuple[int, int]:
    return (abs(number - community), number)


def _key_farthest_low(number: int, community: int) -> tuple[int, int]:
    return (abs(number - community), -number)


def _key_pair_bad_high(number: int, community: int) -> tuple[int, int]:
    return (-int(number == community), number)


def _key_pair_bad_low(number: int, community: int) -> tuple[int, int]:
    return (-int(number == community), -number)


def _key_pair_closest(number: int, community: int) -> tuple[int, int, int]:
    return (int(number == community), -abs(number - community), number)


def _key_pair_farthest(number: int, community: int) -> tuple[int, int, int]:
    return (int(number == community), abs(number - community), number)


def _key_parity_odd(number: int, community: int) -> tuple[int, int]:
    return (number % 2, number)


def _key_parity_even(number: int, community: int) -> tuple[int, int]:
    return (1 - number % 2, number)


def _wrap(number: int, community: int) -> int:
    """Circular distance on a 13-number wheel."""
    raw = abs(number - community)
    return min(raw, DECK - raw)


def _key_wrap_closest(number: int, community: int) -> tuple[int, int]:
    return (-_wrap(number, community), number)


def _key_wrap_farthest(number: int, community: int) -> tuple[int, int]:
    return (_wrap(number, community), number)


def _key_sum_high(number: int, community: int) -> tuple[int, int]:
    return ((number + community) % DECK, number)


def _key_under_community(number: int, community: int) -> tuple[int, int]:
    """Highest number that does not exceed the community number wins."""
    return (int(number <= community), number if number <= community else -number)


def _from_key(key: Callable[[int, int], tuple]) -> Compare:
    return lambda left, right, community: _cmp(
        key(left, community), key(right, community)
    )


# The parity example from the statement is intentionally absent.
HYPOTHESES: dict[str, Compare] = {
    "standard": _from_key(_key_standard),
    "pair_low": _from_key(_key_pair_low),
    "high": _from_key(_key_high),
    "low": _from_key(_key_low),
    "closest_high": _from_key(_key_closest_high),
    "closest_low": _from_key(_key_closest_low),
    "farthest_high": _from_key(_key_farthest_high),
    "farthest_low": _from_key(_key_farthest_low),
    "pair_bad_high": _from_key(_key_pair_bad_high),
    "pair_bad_low": _from_key(_key_pair_bad_low),
    # Wider shapes.  A rule we cannot name is a rule we cannot price, and an
    # unnamed rule costs a whole 60-hand leg, so the ensemble is deliberately
    # broader than the shapes seen so far.
    "pair_closest": _from_key(_key_pair_closest),
    "pair_farthest": _from_key(_key_pair_farthest),
    "parity_odd": _from_key(_key_parity_odd),
    "parity_even": _from_key(_key_parity_even),
    "wrap_closest": _from_key(_key_wrap_closest),
    "wrap_farthest": _from_key(_key_wrap_farthest),
    "sum_high": _from_key(_key_sum_high),
    "under_community": _from_key(_key_under_community),
}


#: ``RANKS[name][community][number]`` - the sort key, precomputed.  Every hot
#: path compares two of these instead of re-invoking the hypothesis lambda.
RANKS: dict[str, tuple] = {}


def _build_rank_tables() -> None:
    keys = {
        "standard": _key_standard, "pair_low": _key_pair_low,
        "high": _key_high, "low": _key_low,
        "closest_high": _key_closest_high, "closest_low": _key_closest_low,
        "farthest_high": _key_farthest_high, "farthest_low": _key_farthest_low,
        "pair_bad_high": _key_pair_bad_high, "pair_bad_low": _key_pair_bad_low,
        "pair_closest": _key_pair_closest, "pair_farthest": _key_pair_farthest,
        "parity_odd": _key_parity_odd, "parity_even": _key_parity_even,
        "wrap_closest": _key_wrap_closest, "wrap_farthest": _key_wrap_farthest,
        "sum_high": _key_sum_high, "under_community": _key_under_community,
    }
    for name, key in keys.items():
        RANKS[name] = tuple(
            None if board == 0 else tuple(
                None if number == 0 else key(number, board)
                for number in range(DECK + 1)
            )
            for board in range(DECK + 1)
        )


_build_rank_tables()


#: Codename -> hypothesis name, for rules already identified in an earlier
#: attempt.  The mapping is fixed for the whole event, so pinning one here
#: turns a 60-hand learning problem into knowledge from the first hand.  Read
#: ``GET /showdown/rules`` after an attempt to fill this in.
KNOWN_CODENAMES: dict[str, str] = {}


@dataclass(frozen=True)
class Observation:
    community: int
    left: int
    right: int
    result: int


class RuleModel:
    """Persistent knowledge for one opaque table-rule codename."""

    def __init__(self, codename: str) -> None:
        self.codename = codename
        self.observations: list[Observation] = []
        self._exact: dict[tuple[int, int, int], int] = {}
        self._seen_hands: set[tuple[str, int, int]] = set()
        self._candidates: set[str] = set(HYPOTHESES)
        self._global_edges: dict[tuple[int, int], list[int]] = {}
        self.conflicts = 0
        self._pinned = False

        # ``standard`` is descriptive in Phase 1 and useful in local tests.  An
        # opaque Phase 2 codename receives no such privileged assumption.
        if codename == "standard":
            self._candidates = {"standard"}
        known = KNOWN_CODENAMES.get(codename)
        if known in HYPOTHESES:
            self._candidates = {known}
            self._pinned = True

    @property
    def candidates(self) -> frozenset[str]:
        return frozenset(self._candidates)

    @property
    def confidence(self) -> float:
        if self.codename == "standard" or self._pinned:
            return 1.0
        # Equal private numbers are guaranteed ties under any deterministic
        # rule and therefore do not distinguish one candidate from another.
        count = sum(item.left != item.right for item in self.observations)
        evidence = min(count / 10.0, 1.0)
        if len(self._candidates) == 1:
            # Do not switch into exploitation because one lucky comparison
            # happened to leave a single simple hypothesis alive.
            return min(0.30 + 0.075 * count, 0.97)
        if self._candidates:
            return 0.15 + 0.35 * evidence / len(self._candidates) ** 0.5
        # We no longer recognize the rule shape, but exact comparisons and the
        # empirical ranking become useful as evidence accumulates.
        return min(0.12 + 0.045 * count, 0.78)

    def observe_recent(
        self,
        recent_hands: Any,
        match_id: Any,
        leg_number: int,
        our_seat: Any,
        opponent_seat: Any,
    ) -> None:
        if our_seat not in (0, 1) or opponent_seat not in (0, 1):
            return
        match_key = match_id if isinstance(match_id, str) else "_default"
        for hand in phase_1._as_list(recent_hands):
            if not isinstance(hand, dict):
                continue
            hand_number = hand.get("hand_number")
            if not isinstance(hand_number, int):
                continue
            identity = (match_key, leg_number, hand_number)
            if identity in self._seen_hands:
                continue
            self._seen_hands.add(identity)

            community = hand.get("community_number")
            shown = hand.get("shown_numbers")
            winners = hand.get("winners")
            if community not in NUMBERS or not isinstance(shown, dict):
                continue
            ours = _shown_number(shown, our_seat)
            theirs = _shown_number(shown, opponent_seat)
            if ours not in NUMBERS or theirs not in NUMBERS:
                continue
            winner_seats = set(phase_1._as_list(winners))
            if our_seat in winner_seats and opponent_seat in winner_seats:
                result = 0
            elif our_seat in winner_seats:
                result = 1
            elif opponent_seat in winner_seats:
                result = -1
            else:
                continue
            self.observe(community, ours, theirs, result)

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
        observation = Observation(community, left, right, result)
        self.observations.append(observation)
        self._candidates = {
            name
            for name in self._candidates
            if HYPOTHESES[name](left, right, community) == result
        }

        # Non-pair comparisons are also evidence for rules based on a global
        # permutation/grouping of the private numbers.
        if left != community and right != community and left != right:
            canonical = (min(left, right), max(left, right))
            canonical_result = result if left < right else -result
            counts = self._global_edges.setdefault(canonical, [0, 0, 0])
            counts[canonical_result + 1] += 1

    def win_share(self, left: int, right: int, community: int) -> float:
        """Posterior win share: 1 win, .5 tie, 0 loss."""
        if left == right:
            return 0.5
        exact = self._exact.get((community, left, right))
        if exact is not None:
            return (exact + 1) / 2

        if self._candidates:
            predictions = [
                HYPOTHESES[name](left, right, community)
                for name in self._candidates
            ]
            return sum((prediction + 1) / 2 for prediction in predictions) / len(
                predictions
            )

        result = self._empirical_compare(left, right, community)
        reliability = min(len(self.observations) / 24.0, 0.72)
        return 0.5 + 0.5 * result * reliability

    def outcome_probs(
        self, left: int, right: int, community: int
    ) -> tuple[float, float]:
        """``(P(left beats right), P(tie))`` - the honest posterior.

        ``win_share`` collapses this to a single scalar, which is fine for a
        heads-up average but loses the distinction between "certainly a tie"
        and "a coin flip between winning and losing".  Multiway pot-splitting
        needs the two apart.
        """
        if left == right:
            return 0.0, 1.0
        exact = self._exact.get((community, left, right))
        if exact is not None:
            return (1.0, 0.0) if exact > 0 else ((0.0, 1.0) if exact == 0 else (0.0, 0.0))
        if self._candidates:
            wins = ties = 0
            for name in self._candidates:
                table = RANKS[name][community]
                outcome = _cmp(table[left], table[right])
                wins += outcome > 0
                ties += outcome == 0
            total = len(self._candidates)
            return wins / total, ties / total
        # No recognised shape left.  Fall back on the empirical ranking, shrunk
        # towards a coin flip by how much evidence actually backs it.
        result = self._empirical_compare(left, right, community)
        reliability = min(len(self.observations) / 24.0, 0.72)
        if result == 0:
            return 0.5 * (1.0 - reliability), reliability
        return 0.5 + 0.5 * result * reliability, 0.0

    def strength(self, number: int, community: int | None) -> float:
        communities = (community,) if community in NUMBERS else NUMBERS
        return sum(
            self.win_share(number, other, board)
            for board in communities
            for other in NUMBERS
        ) / (len(communities) * DECK)

    def _empirical_compare(self, left: int, right: int, community: int) -> int:
        pair_result = self._pair_signal(left, right, community)
        if pair_result is not None:
            return pair_result

        direct = self._global_signal(left, right)
        if direct is not None:
            return direct

        left_score = self._copeland_score(left)
        right_score = self._copeland_score(right)
        if left_score != right_score:
            return _cmp(left_score, right_score)
        return 0

    def _pair_signal(self, left: int, right: int, community: int) -> int | None:
        left_pair, right_pair = left == community, right == community
        if left_pair == right_pair:
            return None
        votes: list[int] = []
        for item in self.observations:
            item_left_pair = item.left == item.community
            item_right_pair = item.right == item.community
            if item_left_pair == item_right_pair:
                continue
            pair_won = item.result if item_left_pair else -item.result
            votes.append(pair_won)
        if not votes or sum(votes) == 0:
            return None
        pair_vs_nonpair = _cmp(sum(votes), 0)
        return pair_vs_nonpair if left_pair else -pair_vs_nonpair

    def _global_signal(self, left: int, right: int) -> int | None:
        canonical = (min(left, right), max(left, right))
        counts = self._global_edges.get(canonical)
        if not counts:
            return None
        signed = counts[2] - counts[0]
        if signed == 0:
            return 0 if counts[1] else None
        result = _cmp(signed, 0)
        return result if left < right else -result

    def _copeland_score(self, number: int) -> int:
        score = 0
        for other in NUMBERS:
            if other == number:
                continue
            signal = self._global_signal(number, other)
            if signal is not None:
                score += signal
        return score


def _shown_number(shown: dict, seat: int) -> Any:
    return shown.get(str(seat), shown.get(seat))


class OpponentModel(phase_1.OpponentModel):
    """Phase 1 style read with leg-safe completed-hand deduplication."""

    def __init__(self) -> None:
        super().__init__()
        self._counted_phase2_hands: set[tuple[str, int, int]] = set()

    def observe(
        self,
        recent_hands: Any,
        opponent_seat: Any,
        match_id: Any,
        leg_number: int,
    ) -> None:
        if opponent_seat not in (0, 1):
            return
        match_key = match_id if isinstance(match_id, str) else "_default"
        for hand in phase_1._as_list(recent_hands):
            if not isinstance(hand, dict):
                continue
            hand_number = hand.get("hand_number")
            if not isinstance(hand_number, int):
                continue
            identity = (match_key, leg_number, hand_number)
            if identity in self._counted_phase2_hands:
                continue
            self._counted_phase2_hands.add(identity)
            for entry in phase_1._as_list(hand.get("actions")):
                if isinstance(entry, dict) and entry.get("seat") == opponent_seat:
                    self._count(entry.get("action"))


class TurnState(phase_1.TurnState):
    def __init__(self, body: dict) -> None:
        super().__init__(body)
        rule = body.get("table_rule")
        self.table_rule = rule if isinstance(rule, str) and rule else "_unknown"
        self.leg_number = phase_1._as_int(body.get("leg_number"))
        self.total_legs = phase_1._as_int(body.get("total_legs"))
        self.button_seat = body.get("button_seat")
        self.small_blind = max(0, phase_1._as_int(body.get("small_blind"), 1))
        self.big_blind = max(0, phase_1._as_int(body.get("big_blind"), 2))
        opponent_name = self.opponent.get("name")
        self.opponent_name = (
            opponent_name if isinstance(opponent_name, str) else "_opponent"
        )


_RULES: "OrderedDict[str, RuleModel]" = OrderedDict()
_OPPONENTS: "OrderedDict[str, OpponentModel]" = OrderedDict()
_STATE_LOCK = RLock()


def _rule_for(codename: str) -> RuleModel:
    with _STATE_LOCK:
        model = _RULES.get(codename)
        if model is None:
            model = RuleModel(codename)
            _RULES[codename] = model
            while len(_RULES) > MAX_RULES:
                _RULES.popitem(last=False)
        _RULES.move_to_end(codename)
        return model


def _opponent_for(name: str) -> OpponentModel:
    with _STATE_LOCK:
        model = _OPPONENTS.get(name)
        if model is None:
            model = OpponentModel()
            _OPPONENTS[name] = model
            while len(_OPPONENTS) > MAX_OPPONENTS:
                _OPPONENTS.popitem(last=False)
        _OPPONENTS.move_to_end(name)
        return model


def reset_models() -> None:
    with _STATE_LOCK:
        _RULES.clear()
        _OPPONENTS.clear()


def rule_snapshot() -> dict[str, dict[str, Any]]:
    """A test/debug-friendly copy; never expose mutable live model state."""
    with _STATE_LOCK:
        return {
            name: {
                "observations": len(model.observations),
                "candidates": sorted(model.candidates),
                "confidence": model.confidence,
                "conflicts": model.conflicts,
            }
            for name, model in _RULES.items()
        }


def equity_vs_range(
    rule: RuleModel,
    number: int,
    community: int | None,
    weights: dict[int, float],
) -> float:
    mass = sum(max(weight, 0.0) for weight in weights.values())
    if mass <= 0:
        return rule.strength(number, community)
    communities = (community,) if community in NUMBERS else NUMBERS
    total = 0.0
    for board in communities:
        for theirs, weight in weights.items():
            if theirs in NUMBERS and weight > 0:
                total += weight * rule.win_share(number, theirs, board)
    return total / (mass * len(communities))


def _infer_range(
    state: TurnState, rule: RuleModel, opponent: OpponentModel
) -> dict[int, float]:
    weights = {number: 1.0 for number in NUMBERS}
    open_width = min(max(opponent.aggression, 0.18), 0.78)
    for entry in state.hand_actions:
        if entry.get("seat") != state.opponent_seat:
            continue
        community = None if entry.get("round") == "pre_reveal" else state.community
        action = entry.get("action")
        if action in phase_1.AGGRESSIVE_ACTIONS:
            keep, softness = open_width, 0.38
        elif action == "call":
            keep, softness = 0.85, 0.55
        else:
            continue
        ranked = sorted(
            NUMBERS,
            key=lambda number: rule.strength(number, community),
            reverse=True,
        )
        cutoff = max(1, round(keep * DECK))
        strong = set(ranked[:cutoff])
        weights = {
            number: weight * (1.0 if number in strong else softness)
            for number, weight in weights.items()
        }
    return weights


def _future_blind_cost(state: TurnState) -> int:
    """Exact mandatory cost after the current hand if we never invest more."""
    if state.total_hands <= state.hand_number or state.seat not in (0, 1):
        return 0
    button = state.button_seat
    if button not in (0, 1):
        # Malformed/missing position: conservative upper bound.
        return state.big_blind * (state.total_hands - state.hand_number)
    cost = 0
    for _ in range(state.hand_number + 1, state.total_hands + 1):
        button = 1 - button
        cost += state.small_blind if state.seat == button else state.big_blind
    return cost


def _endgame_locked(state: TurnState) -> bool:
    guaranteed = (
        state.chip_delta - state.committed_this_hand - _future_blind_cost(state)
    )
    return guaranteed >= TARGET_DELTA


def _commit_cap(state: TurnState, equity: float, confidence: float) -> int:
    if confidence < EXPLORE_MIN_CONFIDENCE:
        return min(state.stack, EXPLORE_CALL_CAP)
    if equity >= NUTS_EQUITY:
        return state.stack
    fraction = 0.50 if equity >= 0.76 else 0.27
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


def _raise_ev(
    state: TurnState,
    amount: int,
    equity_of: Callable[[dict[int, float]], float],
    weights: dict[int, float],
    rule: RuleModel,
    opponent: OpponentModel,
) -> float:
    our_add = amount - state.my_bet
    their_add = max(0, amount - state.their_bet)
    final_pot = state.pot + our_add + their_add
    threshold = their_add / final_pot if final_pot > 0 else 0.0
    mass = sum(weights.values()) or 1.0
    continuing = {
        number: weight
        for number, weight in weights.items()
        if rule.strength(number, state.community) >= threshold
    }
    rational_fold = 1.0 - sum(continuing.values()) / mass
    p_fold = min(
        max(0.5 * rational_fold + 0.5 * opponent.fold_to_bet, 0.03), 0.90
    )
    if not continuing:
        continuing = dict(weights)
    return p_fold * state.pot + (1.0 - p_fold) * (
        equity_of(continuing) * final_pot - our_add
    )


def _best_raise(
    state: TurnState,
    equity_of: Callable[[dict[int, float]], float],
    weights: dict[int, float],
    rule: RuleModel,
    opponent: OpponentModel,
    max_add: int,
) -> tuple[float, int] | None:
    choices = [
        (
            _raise_ev(state, amount, equity_of, weights, rule, opponent),
            amount,
        )
        for amount in _sizing_menu(state, max_add)
    ]
    return max(choices, default=None)


def _explore(state: TurnState) -> dict:
    if state.to_call <= 0 and "check" in state.legal:
        return {"action": "check"}
    price_limit = min(
        EXPLORE_CALL_CAP,
        max(2, round((state.pot + state.to_call) * EXPLORE_MAX_POT_FRACTION)),
    )
    if state.to_call <= price_limit and "call" in state.legal:
        return {"action": "call"}
    if "fold" in state.legal:
        return {"action": "fold"}
    return state.fallback()


def decide(state: TurnState) -> dict:
    if state.number not in NUMBERS or not state.legal:
        return state.fallback()

    rule = _rule_for(state.table_rule)
    opponent = _opponent_for(state.opponent_name)
    with _STATE_LOCK:
        rule.observe_recent(
            state.recent_hands,
            state.match_id,
            state.leg_number,
            state.seat,
            state.opponent_seat,
        )
        opponent.observe(
            state.recent_hands,
            state.opponent_seat,
            state.match_id,
            state.leg_number,
        )

    locked = _endgame_locked(state)
    if locked and state.to_call > 0:
        return {"action": "fold"} if "fold" in state.legal else state.fallback()
    if rule.confidence < EXPLORE_MIN_CONFIDENCE:
        return _explore(state)

    weights = _infer_range(state, rule, opponent)

    def equity_of(range_weights: dict[int, float]) -> float:
        return equity_vs_range(
            rule, state.number, state.community, range_weights
        )

    equity = equity_of(weights)
    max_add = _commit_cap(state, equity, rule.confidence)

    if state.to_call <= 0:
        check_ev = equity * state.pot
        value = equity >= (NUTS_EQUITY if locked else VALUE_BET_MIN)
        bluff = not locked and equity <= BLUFF_BET_MAX and opponent.bluff_budget_left
        action = "bet" if "bet" in state.legal else (
            "raise" if "raise" in state.legal else None
        )
        if action and (value or bluff):
            best = _best_raise(
                state, equity_of, weights, rule, opponent, max_add
            )
            if best and best[0] > check_ev:
                opponent.our_aggressive += 1
                if bluff and not value:
                    opponent.our_bluffs += 1
                return {"action": action, "amount": best[1]}
        if "check" in state.legal:
            return {"action": "check"}
        return state.fallback()

    call_ev = equity * (state.pot + state.to_call) - state.to_call
    value = equity >= VALUE_RAISE_MIN
    bluff = equity <= BLUFF_RAISE_MAX and opponent.bluff_budget_left
    if "raise" in state.legal and (value or bluff):
        best = _best_raise(state, equity_of, weights, rule, opponent, max_add)
        if best and best[0] > max(call_ev, 0.0):
            opponent.our_aggressive += 1
            if bluff and not value:
                opponent.our_bluffs += 1
            return {"action": "raise", "amount": best[1]}

    thin_stack_off = (
        equity < STACK_OFF_EQUITY
        and state.stack > 0
        and state.to_call >= 0.9 * state.stack
        and state.hand_number < state.total_hands
    )
    if call_ev > 0 and "call" in state.legal and not thin_stack_off:
        return {"action": "call"}
    if "fold" in state.legal:
        return {"action": "fold"}
    return state.fallback()


def move_from_body(body: dict) -> dict:
    """Safe Phase 2 entry point used by the shared Flask handler."""
    legal = phase_1._legal_actions(body)
    try:
        state = TurnState(body)
    except Exception:  # noqa: BLE001 - an illegal reply costs the hand
        return phase_1._fallback(
            legal, body.get("min_raise_to"), body.get("max_raise_to")
        )
    try:
        move = decide(state)
    except Exception:  # noqa: BLE001 - keep the endpoint fail-safe
        move = state.fallback()
    return phase_1.legalise(move, state)
