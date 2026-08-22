"""SHOWDOWN - Phase 4: The Final Table.

Phase 4 is a knockout against other teams' bots: up to seven seats, 200 hands
per game, one codenamed table rule per table, and the bottom third of every
table is cut between rounds.  Two things follow, and both differ from Phase 3.

*Rank is worth less than survival.*  The final table pays 400 down to 160,
a 240 point spread, while being cut early drops the whole run to 40-120.  So
the objective is "stay out of the bottom third", not "finish first" - and
busting is the one outcome that guarantees the cut.

*The opponents are unknown.*  There is no fixed cast to profile across legs,
and no retry to learn from, so the read is per-name and starts neutral.  What
does carry over is the rule knowledge: the codename mapping is fixed for the
whole event, so a table rule already identified in Phase 2 or Phase 3 is
identified here from the first hand.
"""

from __future__ import annotations

from collections import OrderedDict
from threading import RLock
from typing import Any

from challenges.showdown import phase_1, phase_2, phase_3


_ROUNDS_LOCK = RLock()


NUMBERS = phase_3.NUMBERS
DECK = phase_3.DECK

#: Survival is worth more than chips, so every bar sits above its Phase 3
#: counterpart and the stack-off bar is the strictest of the three.
VALUE_BET_BASE = 0.36
VALUE_RAISE_BASE = 0.47
STACK_OFF_BASE = 0.70
BLUFF_BASE = 0.50
EXPLORE_COMMIT_CAP = 10

#: Cushion, in chips, we want above the cut line before we stop taking risks.
#: A simulated sweep (14 games per setting, so weak evidence) showed 25 taking
#: the occasional bust that 50 did not, at no cost to rank.  Busting is the one
#: unrecoverable outcome here, so the tie breaks towards the wider cushion.
SAFE_MARGIN = 50

#: Rank ambition once survival is already comfortable.  Making the final table
#: is the dominant term - an early cut pays 40-120 against 160 for last place
#: there - so the early rounds are played for survival.  But the final table
#: itself spreads 400 down to 160, and by then coasting is worth 240 fewer
#: points than winning, so ambition is scaled up by rounds already survived.
RANK_AMBITION_STEP = 0.045
RANK_AMBITION_MAX = 0.12

#: Bracket games seen by this process, oldest first.  The count is the best
#: available proxy for how deep the bracket has run: the wire never says which
#: round this is, or how much of the field is left.
_ROUNDS: "OrderedDict[str, int]" = OrderedDict()
MAX_TRACKED_ROUNDS = 64


class TurnState(phase_3.TurnState):
    """Phase 3's table parsing; Phase 4 only adds the cut line."""

    def __init__(self, body: dict) -> None:
        super().__init__(body)
        self.total_hands = phase_1._as_int(body.get("total_hands"), 200) or 200
        # "Player 3" at the first table and "Player 3" at the final table are
        # different teams, so a read must not outlive the game it was taken in.
        self.opponent_scope = (
            self.match_id if isinstance(self.match_id, str) and self.match_id
            else "_p4"
        )


def _live_deltas(state: TurnState) -> list[int]:
    """Every seat's delta, ours included, measured off live stacks.

    Busted seats are included at a flat ``-starting_stack``: they are still
    part of the table being cut, and they are guaranteed to be at the bottom
    of it.  Counting only the survivors both shrinks the field the cut is
    taken from and hides the seats already filling it, which reads as danger
    at exactly the moment the danger has passed.

    ``chip_delta`` is frozen at the start of the hand, so mid-hand it lies
    about anyone who has already put chips in.  Stacks do not.
    """
    return [
        max(0, phase_1._as_int(player.get("stack"))) - state.starting_stack
        for player in state.players
    ]


def _cut_line(state: TurnState) -> int:
    """The highest delta that still gets cut.  Beat it and we go through.

    "Bottom third" is rounded to nearest rather than truncated: at seven seats
    truncation claims two are cut and rounding up claims three, and neither
    reading is given anywhere.  The nearest-integer split sits between them,
    and ``SAFE_MARGIN`` covers the remaining seat of doubt.
    """
    deltas = sorted(_live_deltas(state))
    if len(deltas) < 2:
        return -state.starting_stack
    cut = max(1, round(len(deltas) / 3))
    return deltas[min(cut, len(deltas)) - 1]


def _round_number(state: TurnState) -> int:
    """How many bracket games this process has seen, this one included."""
    key = state.match_id if isinstance(state.match_id, str) and state.match_id else "_p4"
    with _ROUNDS_LOCK:
        if key not in _ROUNDS:
            _ROUNDS[key] = len(_ROUNDS) + 1
            while len(_ROUNDS) > MAX_TRACKED_ROUNDS:
                _ROUNDS.popitem(last=False)
        return _ROUNDS[key]


def reset_rounds() -> None:
    with _ROUNDS_LOCK:
        _ROUNDS.clear()


def _our_delta(state: TurnState) -> int:
    return state.stack - state.starting_stack


def _cut_pressure(state: TurnState) -> float:
    """Extra willingness to gamble, and only when the cut is actually close.

    Below the line with the game running out, folding is not safe - it is a
    slow way of finishing last.  Above the line this stays at zero and the
    bot simply plays its equity.
    """
    deficit = max(0, _cut_line(state) + SAFE_MARGIN - _our_delta(state))
    if deficit <= 0:
        return 0.0
    hands_left = max(1, state.total_hands - state.hand_number + 1)
    urgency = 1.0 - min(hands_left / max(state.total_hands, 1), 1.0)
    return min(0.24, deficit / max(state.starting_stack, 1) * urgency)


def _rank_pressure(state: TurnState) -> float:
    """Ambition to climb, but only from a position that can afford it.

    Zero until survival is comfortable, zero once we already lead, and zero in
    the closing hands where a lost race cannot be recovered.  In between it
    grows with the round, because the deeper the bracket has run the more the
    remaining points are decided by rank rather than by simply surviving.
    """
    ours, line = _our_delta(state), _cut_line(state)
    if ours <= line + SAFE_MARGIN:
        return 0.0
    leader = max(
        (
            max(0, phase_1._as_int(player.get("stack"))) - state.starting_stack
            for player in state.opponents
        ),
        default=ours,
    )
    if ours >= leader:
        return 0.0
    hands_left = max(0, state.total_hands - state.hand_number)
    if hands_left <= max(4, len(state.active_players)):
        return 0.0
    ambition = min(RANK_AMBITION_MAX, RANK_AMBITION_STEP * _round_number(state))
    gap = min(1.0, (leader - ours) / max(state.starting_stack, 1))
    return ambition * gap


def _pressure(state: TurnState) -> float:
    """Survival first; ambition only from safety."""
    return max(_cut_pressure(state), _rank_pressure(state))


def _survival_locked(state: TurnState) -> bool:
    """True once the cut is mathematically out of reach for the hands left.

    Unlike Phase 3 this does not require topping the table: finishing second
    at the final table pays 360 of a possible 400, and finishing the game at
    all pays enormously better than busting out of it.
    """
    hands_left = max(0, state.total_hands - state.hand_number)
    if hands_left > max(4, len(state.active_players)):
        return False
    blind_cost = phase_3._future_blind_cost(state)
    guaranteed = max(0, state.stack - blind_cost) - state.starting_stack
    # Everything an opponent could still win from the others plus this pot.
    reachable = _cut_line(state) + state.pot + blind_cost + SAFE_MARGIN
    return guaranteed > reachable


PROFILE = phase_3.Profile(
    value_bet=VALUE_BET_BASE,
    value_raise=VALUE_RAISE_BASE,
    stack_off=STACK_OFF_BASE,
    bluff=BLUFF_BASE,
    explore_cap=EXPLORE_COMMIT_CAP,
    max_bluff_opponents=2,
    pressure=_pressure,
    locked=_survival_locked,
)


def decide(state: TurnState) -> dict:
    return phase_3.decide(state, PROFILE)


def move_from_body(body: dict) -> dict:
    """Fail-safe Phase 4 entry point used by the shared Flask route."""
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


def reset_models() -> None:
    phase_3.reset_models()
    reset_rounds()


# Rule knowledge is shared: the codename mapping is fixed for the whole event.
RuleModel = phase_2.RuleModel
_rule_for = phase_2._rule_for
rule_snapshot = phase_2.rule_snapshot
