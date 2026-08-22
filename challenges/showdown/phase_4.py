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

from typing import Any

from challenges.showdown import phase_1, phase_2, phase_3


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
SAFE_MARGIN = 25


class TurnState(phase_3.TurnState):
    """Phase 3's table parsing; Phase 4 only adds the cut line."""

    def __init__(self, body: dict) -> None:
        super().__init__(body)
        self.total_hands = phase_1._as_int(body.get("total_hands"), 200) or 200


def _live_deltas(state: TurnState) -> list[int]:
    """Every active seat's delta, ours included, measured off live stacks.

    ``chip_delta`` is frozen at the start of the hand, so mid-hand it lies
    about anyone who has already put chips in.  Stacks do not.
    """
    return [
        max(0, phase_1._as_int(player.get("stack"))) - state.starting_stack
        for player in state.active_players
    ]


def _cut_line(state: TurnState) -> int:
    """The highest delta that still gets cut.  Beat it and we go through."""
    deltas = sorted(_live_deltas(state))
    if len(deltas) < 2:
        return -state.starting_stack
    cut = max(1, len(deltas) // 3)
    return deltas[min(cut, len(deltas)) - 1]


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
    pressure=_cut_pressure,
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


# Rule knowledge is shared: the codename mapping is fixed for the whole event.
RuleModel = phase_2.RuleModel
_rule_for = phase_2._rule_for
rule_snapshot = phase_2.rule_snapshot
