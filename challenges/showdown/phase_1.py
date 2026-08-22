"""SHOWDOWN - Phase 1: First Contact.

Heads-up against one house bot: 100 hands, ``table_rule`` reads "standard" the
whole way through, and the phase clears at a final chip delta of +10 or better.

The bot is built in layers, outermost first:

  0. safety shell   - never return an illegal action and never raise out of the
                      request handler; five bad replies in a row forfeit the match
  1. equity         - closed-form win share under the standard ruleset
  2. opponent model - per-match statistics mined from the action logs
  3. EV decision    - argmax over fold/check/call plus a discrete sizing menu
  4. variance cap   - busting is a flat -200, so never flip for the stack
  5. endgame lock   - once +10 is mathematically safe, stop gambling

Endpoints, mounted under ``/showdown``::

    POST /showdown/move
    GET  /showdown/health
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Callable

from flask import jsonify, request

from challenges.showdown import blueprint


DECK = 13
NUMBERS = tuple(range(1, DECK + 1))

TARGET_DELTA = 10           # the phase clears here
LOCK_MARGIN = 8             # cushion before we switch to lock-down mode
BLIND_COST_PER_HAND = 1.5   # heads-up you alternate posting 1 and 2
LOCK_MIN_EQUITY = 0.75      # in lock-down mode, only these hands keep playing

NUTS_EQUITY = 0.90          # at or above this we will play for the whole stack
STACK_OFF_EQUITY = 0.80     # the thinnest edge we will call off a whole stack on
POT_FRACTIONS = (0.5, 0.75, 1.25)

VALUE_BET_MIN = 0.60        # open the betting for value at or above this
VALUE_RAISE_MIN = 0.65
BLUFF_BET_MAX = 0.28        # only the bottom of the range bluffs
BLUFF_RAISE_MAX = 0.22
MAX_BLUFF_RATIO = 0.28      # cap bluffs as a share of our own aggressive actions
MIN_BLUFF_SAMPLE = 6

AGGRESSION_PRIOR = 0.35     # smoothing, so a short log cannot swing the read
AGGRESSION_PRIOR_WEIGHT = 4
FOLD_PRIOR = 0.45
FOLD_PRIOR_WEIGHT = 3

RANGE_SOFTNESS = 0.35       # weight left on hands outside their assumed betting range
CALL_TIGHTEN_Q = 0.85
CALL_SOFTNESS = 0.50

MAX_TRACKED_MATCHES = 64

AGGRESSIVE_ACTIONS = ("bet", "raise")
FACING_BET_ACTIONS = ("call", "raise", "fold")
KNOWN_ACTIONS = ("check", "call", "bet", "raise", "fold")
FALLBACK_ORDER = ("check", "call", "fold")


# --------------------------------------------------------------------------
# Layer 1: equity
# --------------------------------------------------------------------------

def _score(number: int, community: int | None) -> tuple[int, int]:
    """Showdown strength under the standard rule: any pair beats any non-pair."""
    return (1 if number == community else 0, number)


def equity_post(number: int, community: int) -> float:
    """Exact win share of ``number`` against one uniform opponent, community known."""
    if number == community:
        # Nothing beats a pair here; only the matching number ties.
        return (DECK - 1 + 0.5) / DECK
    # We beat every lower number apart from the community number itself, which
    # would be a pair, and we tie with the identical number.
    return (number - 0.5 - (1 if community < number else 0)) / DECK


#: Pre-reveal equity, averaged over the community number. Linear in the number.
PRE_EQUITY = (0.0,) + tuple(
    sum(equity_post(n, c) for c in NUMBERS) / DECK for n in NUMBERS
)


def _strength(number: int, community: int | None) -> float:
    """Win share of a single number against a uniform opponent."""
    return PRE_EQUITY[number] if community is None else equity_post(number, community)


def equity_vs_range(
    number: int, community: int | None, weights: dict[int, float]
) -> float:
    """Win share of ``number`` against a weighted opponent range.

    A ``community`` of None averages over the reveal, which is what we need in
    the pre-reveal betting round.
    """
    mass = sum(weights.values())
    if mass <= 0:
        return _strength(number, community)

    communities = (community,) if community is not None else NUMBERS
    total = 0.0
    for board in communities:
        ours = _score(number, board)
        for theirs, weight in weights.items():
            if weight <= 0:
                continue
            other = _score(theirs, board)
            if ours > other:
                total += weight
            elif ours == other:
                total += 0.5 * weight
    return total / (mass * len(communities))


def _uniform_range() -> dict[int, float]:
    return {n: 1.0 for n in NUMBERS}


def _tighten(
    weights: dict[int, float],
    community: int | None,
    keep_fraction: float,
    softness: float,
) -> dict[int, float]:
    """Damp the weakest hands, leaving the top ``keep_fraction`` untouched."""
    ranked = sorted(NUMBERS, key=lambda n: _strength(n, community), reverse=True)
    keep = max(1, round(keep_fraction * DECK))
    return {
        number: weights[number] * (1.0 if rank < keep else softness)
        for rank, number in enumerate(ranked)
    }


# --------------------------------------------------------------------------
# Layer 2: opponent model
# --------------------------------------------------------------------------

class OpponentModel:
    """Per-match read on the house bot, mined from the completed hand logs."""

    def __init__(self) -> None:
        self.actions = 0
        self.aggressive = 0
        self.faced_bet = 0
        self.folds = 0
        self.our_aggressive = 0
        self.our_bluffs = 0
        self._counted_hands: set[int] = set()

    def observe_completed_hands(
        self, recent_hands: Any, opponent_seat: int | None
    ) -> None:
        """Fold finished hands into the long-run counts, each hand only once.

        ``recent_hands`` is a rolling 20-hand window and we are called many
        times per hand, so dedupe on the hand number to build a read that spans
        the whole 100-hand match.
        """
        if opponent_seat is None:
            return
        for hand in _as_list(recent_hands):
            if not isinstance(hand, dict):
                continue
            number = hand.get("hand_number")
            if not isinstance(number, int) or number in self._counted_hands:
                continue
            self._counted_hands.add(number)
            for entry in _as_list(hand.get("actions")):
                if isinstance(entry, dict) and entry.get("seat") == opponent_seat:
                    self._count(entry.get("action"))

    def _count(self, action: Any) -> None:
        if action not in KNOWN_ACTIONS:
            return
        self.actions += 1
        if action in AGGRESSIVE_ACTIONS:
            self.aggressive += 1
        # call, raise and fold are only reachable with a live bet in front of them.
        if action in FACING_BET_ACTIONS:
            self.faced_bet += 1
        if action == "fold":
            self.folds += 1

    @property
    def aggression(self) -> float:
        """How wide they open the betting, smoothed towards a neutral prior."""
        return (self.aggressive + AGGRESSION_PRIOR * AGGRESSION_PRIOR_WEIGHT) / (
            self.actions + AGGRESSION_PRIOR_WEIGHT
        )

    @property
    def fold_to_bet(self) -> float:
        """How often they give up when someone bets at them."""
        return (self.folds + FOLD_PRIOR * FOLD_PRIOR_WEIGHT) / (
            self.faced_bet + FOLD_PRIOR_WEIGHT
        )

    @property
    def bluff_budget_left(self) -> bool:
        if self.our_aggressive < MIN_BLUFF_SAMPLE:
            return True
        return self.our_bluffs / self.our_aggressive < MAX_BLUFF_RATIO


_MODELS: "OrderedDict[str, OpponentModel]" = OrderedDict()


def _model_for(match_id: Any) -> OpponentModel:
    key = match_id if isinstance(match_id, str) and match_id else "_default"
    model = _MODELS.get(key)
    if model is None:
        model = OpponentModel()
        _MODELS[key] = model
        while len(_MODELS) > MAX_TRACKED_MATCHES:
            _MODELS.popitem(last=False)
    _MODELS.move_to_end(key)
    return model


def reset_models() -> None:
    """Drop every accumulated read. Used by the tests."""
    _MODELS.clear()


# --------------------------------------------------------------------------
# Turn state
# --------------------------------------------------------------------------

def _as_int(value: Any, default: int = 0) -> int:
    return int(value) if isinstance(value, (int, float)) else default


def _as_bound(value: Any) -> int | None:
    """A raise bound, or None when they did not send a usable one.

    Accepts a float: a bound arriving as 199.0 must not silently switch off
    every bet and raise we would otherwise make.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value)


def _raise_range(state: "TurnState") -> tuple[int, int] | None:
    low, high = _as_bound(state.min_raise_to), _as_bound(state.max_raise_to)
    if low is None or high is None or high < low:
        return None
    return low, high


def _as_list(value: Any) -> list:
    """Anything we iterate has to survive a field arriving as the wrong type."""
    return value if isinstance(value, list) else []


def _legal_actions(body: dict) -> list[str]:
    return [a for a in _as_list(body.get("legal_actions")) if a in KNOWN_ACTIONS]


class TurnState:
    """One request, parsed defensively - unknown fields are ignored."""

    def __init__(self, body: dict) -> None:
        self.legal = _legal_actions(body)
        self.number = body.get("your_number")
        self.community = body.get("community_number")
        self.pot = _as_int(body.get("pot"))
        self.to_call = _as_int(body.get("to_call"))
        self.stack = _as_int(body.get("your_stack"))
        self.min_raise_to = body.get("min_raise_to")
        self.max_raise_to = body.get("max_raise_to")
        self.seat = body.get("your_seat")
        self.match_id = body.get("match_id")
        self.recent_hands = body.get("recent_hands")
        self.hand_number = _as_int(body.get("hand_number"))
        self.total_hands = _as_int(body.get("total_hands"))

        self.hand_actions = [
            entry
            for entry in _as_list(body.get("current_hand_actions"))
            if isinstance(entry, dict)
        ]

        # players is the table's seating, not the list of live opponents.
        players = [p for p in _as_list(body.get("players")) if isinstance(p, dict)]
        self.me = next((p for p in players if p.get("seat") == self.seat), {})
        live = [
            p
            for p in players
            if p.get("seat") != self.seat
            and not p.get("busted")
            and not p.get("folded")
        ]
        self.opponent = live[0] if live else {}
        self.opponent_seat = self.opponent.get("seat")

        self.my_bet = _as_int(self.me.get("bet_this_round"))
        self.their_bet = _as_int(self.opponent.get("bet_this_round"))
        self.chip_delta = self._read_delta(body)

        # chip_delta is frozen at the start of the hand while stack is live, so
        # the difference is exactly what this hand has already cost us.
        self.stack_at_hand_start = (
            _as_int(body.get("starting_stack"), 200) + self.chip_delta
        )
        self.committed_this_hand = max(0, self.stack_at_hand_start - self.stack)

    def _read_delta(self, body: dict) -> int:
        for candidate in (body.get("chip_delta"), self.me.get("chip_delta")):
            if isinstance(candidate, (int, float)):
                return int(candidate)
        return self.stack - _as_int(body.get("starting_stack"), 200)

    def fallback(self) -> dict:
        return _fallback(self.legal, self.min_raise_to, self.max_raise_to)


# --------------------------------------------------------------------------
# Layer 3: EV decision
# --------------------------------------------------------------------------

def _infer_range(state: TurnState, model: OpponentModel) -> dict[int, float]:
    """Narrow the opponent's range using what they have done this hand."""
    weights = _uniform_range()
    open_width = min(max(model.aggression, 0.15), 0.75)
    for entry in state.hand_actions:
        if entry.get("seat") != state.opponent_seat:
            continue
        # Pre-reveal actions were taken blind to the community number, so they
        # have to be read against the pre-reveal strength ordering.
        community = None if entry.get("round") == "pre_reveal" else state.community
        action = entry.get("action")
        if action in AGGRESSIVE_ACTIONS:
            weights = _tighten(weights, community, open_width, RANGE_SOFTNESS)
        elif action == "call":
            weights = _tighten(weights, community, CALL_TIGHTEN_Q, CALL_SOFTNESS)
    return weights


def _split_on_bet(
    weights: dict[int, float],
    community: int | None,
    their_call: int,
    final_pot: int,
    model: OpponentModel,
) -> tuple[float, dict[int, float]]:
    """How often a bet folds them out, and what is left calling when it does not.

    Their price is ``their_call`` to win ``final_pot``, so a rational opponent
    folds everything below that break-even equity. We score their hands against
    a uniform range - we cannot know how they read us - and then blend that
    theoretical fold rate with the one we have actually measured.
    """
    threshold = their_call / final_pot if final_pot > 0 else 0.0
    mass = sum(weights.values()) or 1.0

    continuing = {
        n: w for n, w in weights.items() if _strength(n, community) >= threshold
    }
    above = sum(continuing.values())
    below = mass - above

    rational_fold = below / mass
    p_fold = min(max(0.5 * rational_fold + 0.5 * model.fold_to_bet, 0.03), 0.92)

    # Leak the folding hands back in so the calling range's mass matches p_fold:
    # nobody folds exactly on the break-even point.
    leak = 0.0
    if below > 0:
        leak = min(max(((1.0 - p_fold) * mass - above) / below, 0.0), 1.0)
    calling = {n: (w if n in continuing else w * leak) for n, w in weights.items()}
    if sum(calling.values()) <= 0:
        calling = dict(weights)
    return p_fold, calling


def _commit_cap(state: "TurnState", equity: float) -> int:
    """How much we are willing to put in *of our own initiative* this hand.

    This bounds bets and raises only. Calls are left to pot odds: capping them
    too creates the worst line in the game, calling one street and folding the
    next, which donates the called chips for nothing.

    Busting is not the cliff it first looks like - lose a 200 stack and the
    delta is -200 either way. What busting really costs is the chance to win it
    back over the hands that are left, so the budget loosens as the match runs
    out of those hands.
    """
    if equity >= NUTS_EQUITY:
        return state.stack
    fraction = 0.55 if equity >= LOCK_MIN_EQUITY else 0.30
    budget = int(state.stack_at_hand_start * fraction) - state.committed_this_hand
    return max(0, min(budget, state.stack))


def _sizing_menu(state: TurnState, max_add: int) -> list[int]:
    """Legal ``amount`` values worth evaluating - round totals, not raw adds."""
    bounds = _raise_range(state)
    if bounds is None:
        return []
    low, high = bounds

    targets = [state.my_bet + max(1, round(f * state.pot)) for f in POT_FRACTIONS]
    targets.append(high)  # all-in is always on the menu; the cap decides

    menu: list[int] = []
    seen: set[int] = set()
    for target in sorted(targets):
        amount = min(max(int(target), low), high)
        if amount - state.my_bet > max_add or amount in seen:
            continue
        seen.add(amount)
        menu.append(amount)
    return menu


def _raise_ev(
    state: TurnState,
    amount: int,
    equity_of: Callable[[dict[int, float]], float],
    weights: dict[int, float],
    model: OpponentModel,
) -> float:
    """Chips gained from here by raising to ``amount``, folds and calls blended."""
    our_add = amount - state.my_bet
    their_add = amount - state.their_bet
    final_pot = state.pot + our_add + their_add
    p_fold, calling = _split_on_bet(
        weights, state.community, their_add, final_pot, model
    )
    # Folding them out wins the pot as it stands; our own extra chips come back.
    return p_fold * state.pot + (1 - p_fold) * (
        equity_of(calling) * final_pot - our_add
    )


def _best_raise(
    state: TurnState,
    equity_of: Callable[[dict[int, float]], float],
    weights: dict[int, float],
    model: OpponentModel,
    max_add: int,
) -> tuple[float, int] | None:
    best: tuple[float, int] | None = None
    for amount in _sizing_menu(state, max_add):
        ev = _raise_ev(state, amount, equity_of, weights, model)
        if best is None or ev > best[0]:
            best = (ev, amount)
    return best


# --------------------------------------------------------------------------
# Layer 5: endgame lock
# --------------------------------------------------------------------------

def _thin_stack_off(state: TurnState, equity: float) -> bool:
    """True for a call that risks the whole stack on an edge we do not have."""
    if equity >= STACK_OFF_EQUITY or state.stack <= 0:
        return False
    if state.to_call < state.stack * 0.9:
        return False
    # With the match nearly over there is nothing left to protect.
    return state.total_hands <= 0 or state.hand_number < state.total_hands * 0.9


def _endgame_locked(state: TurnState) -> bool:
    """True once folding every remaining hand would still clear the phase.

    Heads-up you post 1 and 2 on alternate hands, so sitting out costs 1.5 a
    hand. When the delta absorbs that with room to spare the +10 is banked, and
    every further gamble is downside only.
    """
    if state.total_hands <= 0:
        return False
    hands_left = max(state.total_hands - state.hand_number, 0)
    floor = state.chip_delta - BLIND_COST_PER_HAND * hands_left
    return floor >= TARGET_DELTA + LOCK_MARGIN


# --------------------------------------------------------------------------
# The decision itself
# --------------------------------------------------------------------------

def decide(state: TurnState) -> dict:
    if state.number not in NUMBERS or not state.legal:
        return state.fallback()

    model = _model_for(state.match_id)
    model.observe_completed_hands(state.recent_hands, state.opponent_seat)

    weights = _infer_range(state, model)

    def equity_of(range_weights: dict[int, float]) -> float:
        return equity_vs_range(state.number, state.community, range_weights)

    equity = equity_of(weights)
    locked = _endgame_locked(state)
    max_add = _commit_cap(state, equity)

    if state.to_call <= 0:
        return _act_unopened(state, model, weights, equity_of, equity, locked, max_add)
    return _act_facing_bet(state, model, weights, equity_of, equity, locked, max_add)


def _act_unopened(
    state: TurnState,
    model: OpponentModel,
    weights: dict[int, float],
    equity_of: Callable[[dict[int, float]], float],
    equity: float,
    locked: bool,
    max_add: int,
) -> dict:
    """Nothing to match: we can check for free or open the betting."""
    # Checking realises our share of the pot as it stands. A rough baseline, but
    # it puts checking and betting on the same scale.
    check_ev = equity * state.pot

    value = equity >= VALUE_BET_MIN
    # Only the bottom of the range bluffs. Middling hands have showdown value
    # and no fold equity worth buying: betting them folds out what we already
    # beat and gets called by what beats us.
    bluff = equity <= BLUFF_BET_MAX and model.bluff_budget_left
    if locked:
        value, bluff = equity >= NUTS_EQUITY, False

    if "bet" in state.legal:
        action = "bet"
    elif "raise" in state.legal:
        action = "raise"
    else:
        action = None

    if action and (value or bluff):
        best = _best_raise(state, equity_of, weights, model, max_add)
        if best and best[0] > check_ev:
            model.our_aggressive += 1
            if bluff and not value:
                model.our_bluffs += 1
            return {"action": action, "amount": best[1]}

    if "check" in state.legal:
        return {"action": "check"}
    return state.fallback()


def _act_facing_bet(
    state: TurnState,
    model: OpponentModel,
    weights: dict[int, float],
    equity_of: Callable[[dict[int, float]], float],
    equity: float,
    locked: bool,
    max_add: int,
) -> dict:
    """They have bet at us, so folding is on the table and costs nothing further."""
    if locked and equity < LOCK_MIN_EQUITY:
        # The phase is already won on paper. Do not hand it back.
        return {"action": "fold"} if "fold" in state.legal else state.fallback()

    call_ev = equity * (state.pot + state.to_call) - state.to_call

    value = equity >= VALUE_RAISE_MIN
    bluff = equity <= BLUFF_RAISE_MAX and model.bluff_budget_left
    if locked:
        value, bluff = equity >= NUTS_EQUITY, False

    if "raise" in state.legal and (value or bluff):
        best = _best_raise(state, equity_of, weights, model, max_add)
        if best and best[0] > max(call_ev, 0.0):
            model.our_aggressive += 1
            if bluff and not value:
                model.our_bluffs += 1
            return {"action": "raise", "amount": best[1]}

    if call_ev > 0 and "call" in state.legal and not _thin_stack_off(state, equity):
        return {"action": "call"}
    if "fold" in state.legal:
        return {"action": "fold"}
    return state.fallback()


# --------------------------------------------------------------------------
# Layer 0: safety shell
# --------------------------------------------------------------------------

def _fallback(
    legal: list[str], min_raise_to: Any = None, max_raise_to: Any = None
) -> dict:
    """What they substitute for us anyway: check, or fold if checking is not legal."""
    for action in FALLBACK_ORDER:
        if action in legal:
            return {"action": action}
    # Nothing passive is on offer, so put in the smallest raise they will take.
    # An aggressive action without an amount is an illegal move, not a clamp.
    low, high = _as_bound(min_raise_to), _as_bound(max_raise_to)
    if low is not None and high is not None and high >= low:
        for action in AGGRESSIVE_ACTIONS:
            if action in legal:
                return {"action": action, "amount": low}
    return {"action": "fold"}


def legalise(move: dict, state: TurnState) -> dict:
    """Last gate before the wire. A bad amount is not clamped for us, it is illegal."""
    action = move.get("action")
    if action not in state.legal:
        return state.fallback()
    if action not in AGGRESSIVE_ACTIONS:
        return {"action": action}

    bounds = _raise_range(state)
    if bounds is None:
        return state.fallback()
    low, high = bounds

    amount = move.get("amount")
    if not isinstance(amount, (int, float)):
        amount = low
    return {"action": action, "amount": int(min(max(round(amount), low), high))}


@blueprint.post("/move")
def handle_move():
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify(action="fold")

    # Keep the proven Phase 1 implementation intact while allowing the same
    # registered endpoint to serve the multi-leg, learnable-rule phase.
    if body.get("phase") == 2:
        from challenges.showdown.phase_2 import move_from_body

        return jsonify(move_from_body(body))

    legal = _legal_actions(body)
    try:
        state = TurnState(body)
    except Exception:  # noqa: BLE001 - one bad reply costs a hand, five cost the match
        return jsonify(_fallback(legal))

    try:
        move = decide(state)
    except Exception:  # noqa: BLE001
        move = state.fallback()
    return jsonify(legalise(move, state))


@blueprint.get("/health")
def handle_health():
    """Their warm-up probe, so a cold start does not eat our first move."""
    return jsonify(status="ok")