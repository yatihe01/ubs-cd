import random

import pytest

from app import create_app
from challenges.showdown import phase_2, phase_3, phase_4


@pytest.fixture(autouse=True)
def clean_models():
    phase_4.reset_models()
    yield
    phase_4.reset_models()


@pytest.fixture()
def client():
    app = create_app()
    app.config.update(TESTING=True)
    return app.test_client()


def player(seat, name, *, folded=False, busted=False, stack=200, bet=0, delta=0):
    return {
        "seat": seat, "name": name, "folded": folded, "busted": busted,
        "chip_delta": delta, "bet_this_round": bet, "stack": stack,
        "all_in": False,
    }


def build_body(**overrides):
    body = {
        "protocol_version": 2, "match_id": "p4-table-3", "phase": 4,
        "table_rule": "chalcedony", "small_blind": 1, "big_blind": 2,
        "starting_stack": 200, "your_stack": 200, "hand_number": 20,
        "total_hands": 200, "round": "post_reveal", "your_number": 7,
        "community_number": 7, "your_seat": 0, "button_seat": 2,
        "pot": 30, "to_call": 0, "min_raise_to": 2, "max_raise_to": 200,
        "legal_actions": ["check", "bet"],
        "players": [player(0, "you")] + [
            player(i, f"Player {i}") for i in range(1, 7)
        ],
        "current_hand_actions": [], "recent_hands": [],
    }
    body.update(overrides)
    return body


def test_phase4_dispatches_at_both_registered_urls(client):
    for url in ("/showdown/move", "/move"):
        response = client.post(url, json=build_body())
        assert response.status_code == 200
        assert response.get_json()["action"] in ("check", "bet")


def test_cut_line_is_the_highest_delta_that_still_gets_cut():
    # Seven seats, bottom third cut = two seats. Stacks 200/150/160/…
    stacks = [200, 120, 140, 260, 300, 190, 210]
    body = build_body(
        your_stack=stacks[0],
        players=[player(i, "you" if i == 0 else f"Player {i}", stack=s)
                 for i, s in enumerate(stacks)],
    )
    state = phase_4.TurnState(body)
    # Sorted deltas: -80, -60, -10, 0, +10, +60, +100 -> two cut, line = -60.
    assert phase_4._cut_line(state) == -60
    assert phase_4._our_delta(state) == 0


def test_cut_pressure_is_zero_when_comfortably_clear_and_rises_when_short():
    safe = phase_4.TurnState(build_body(
        your_stack=340, hand_number=190,
        players=[player(0, "you", stack=340)] + [
            player(i, f"Player {i}", stack=140) for i in range(1, 7)
        ],
    ))
    assert phase_4._cut_pressure(safe) == 0.0

    short = phase_4.TurnState(build_body(
        your_stack=60, hand_number=190,
        players=[player(0, "you", stack=60)] + [
            player(i, f"Player {i}", stack=240) for i in range(1, 7)
        ],
    ))
    assert phase_4._cut_pressure(short) > 0.0


def test_survival_lock_does_not_require_topping_the_table():
    """Phase 3 must be first; Phase 4 only has to clear the cut."""
    body = build_body(
        your_stack=300, hand_number=199, total_hands=200, pot=4,
        players=[player(0, "you", stack=300)] + [
            player(i, f"Player {i}", stack=400 if i == 1 else 100)
            for i in range(1, 7)
        ],
    )
    state = phase_4.TurnState(body)
    # Someone else is chip leader, so Phase 3's "protect the lead" is false...
    assert not phase_3._protect_late_lead(state)
    # ...but we are far clear of the cut line, which is all Phase 4 needs.
    assert phase_4._survival_locked(state)


def test_a_locked_survivor_stops_paying_off_bets(client):
    body = build_body(
        your_stack=300, hand_number=199, total_hands=200,
        your_number=2, community_number=9, pot=120, to_call=90,
        legal_actions=["fold", "call", "raise"], min_raise_to=180,
        players=[player(0, "you", stack=300)] + [
            player(i, f"Player {i}", stack=100, bet=90 if i == 1 else 0)
            for i in range(1, 7)
        ],
    )
    assert client.post("/showdown/move", json=body).get_json() == {"action": "fold"}


def test_phase4_fuzz_always_returns_a_legal_move(client):
    rng = random.Random(11)
    for _ in range(250):
        seats = rng.randint(2, 7)
        to_call = rng.choice([0, 2, 15, 80])
        legal = ["check", "bet"] if to_call == 0 else ["fold", "call", "raise"]
        body = build_body(
            table_rule=rng.choice(["chalcedony", "obsidian", "standard"]),
            your_stack=rng.randint(1, 400), hand_number=rng.randint(1, 200),
            round=rng.choice(["pre_reveal", "post_reveal"]),
            your_number=rng.randint(1, 13),
            community_number=rng.choice([None, rng.randint(1, 13)]),
            pot=rng.randint(3, 400), to_call=to_call, legal_actions=legal,
            min_raise_to=rng.randint(2, 50), max_raise_to=rng.randint(50, 400),
            button_seat=rng.randint(0, seats - 1),
            players=[player(0, "you", stack=rng.randint(0, 400))] + [
                player(i, f"Player {i}", stack=rng.randint(0, 400),
                       folded=rng.random() < 0.3, busted=rng.random() < 0.15,
                       bet=rng.choice([0, 2, 15]))
                for i in range(1, seats)
            ],
        )
        response = client.post("/showdown/move", json=body)
        assert response.status_code == 200
        move = response.get_json()
        assert move["action"] in body["legal_actions"]
        if move["action"] in ("bet", "raise"):
            assert body["min_raise_to"] <= move["amount"] <= body["max_raise_to"]


def test_known_codenames_are_trusted_from_the_first_hand(monkeypatch):
    monkeypatch.setitem(phase_2.KNOWN_CODENAMES, "obsidian", "parity_odd")
    phase_4.reset_models()
    rule = phase_2._rule_for("obsidian")
    assert sorted(rule.candidates) == ["parity_odd"]
    assert rule.confidence == 1.0


def test_busted_seats_still_count_towards_the_cut():
    """Seats already at zero fill the cut, and that makes survivors safer.

    Counting only active seats both shrinks the field the bottom third is
    taken from and hides the seats already filling it, so the bot read danger
    at exactly the moment the danger had passed - and gambled accordingly.
    """
    players = (
        [player(0, "you", stack=150)]
        + [player(i, f"Player {i}", stack=0, busted=True) for i in (1, 2, 3)]
        + [player(i, f"Player {i}", stack=350) for i in (4, 5, 6)]
    )
    state = phase_4.TurnState(build_body(
        your_stack=150, hand_number=150, players=players,
    ))
    # Bottom two of seven are cut and three seats have already busted.
    assert phase_4._cut_line(state) == -200
    assert phase_4._our_delta(state) > phase_4._cut_line(state)
    assert phase_4._cut_pressure(state) == 0.0


def test_round_number_counts_distinct_bracket_games():
    phase_4.reset_rounds()
    first = phase_4.TurnState(build_body(match_id="table-a"))
    second = phase_4.TurnState(build_body(match_id="table-b"))
    assert phase_4._round_number(first) == 1
    assert phase_4._round_number(second) == 2
    assert phase_4._round_number(first) == 1  # stable, not a counter per call


def _ranked_body(our_stack, leader_stack, **overrides):
    players = [player(0, "you", stack=our_stack), player(1, "Player 1", stack=leader_stack)]
    players += [player(i, f"Player {i}", stack=40) for i in range(2, 7)]
    body = dict(your_stack=our_stack, hand_number=40, total_hands=200, players=players)
    body.update(overrides)
    return phase_4.TurnState(build_body(**body))


def test_rank_ambition_is_zero_until_survival_is_comfortable():
    phase_4.reset_rounds()
    # Below the cut line: survival logic owns this region, not ambition.
    short = _ranked_body(30, 400)
    assert phase_4._rank_pressure(short) == 0.0
    # Comfortably clear but behind the leader: now it is worth climbing.
    safe = _ranked_body(260, 400)
    assert phase_4._rank_pressure(safe) > 0.0
    # Already leading: nothing to chase.
    leading = _ranked_body(400, 260)
    assert phase_4._rank_pressure(leading) == 0.0


def test_rank_ambition_grows_with_rounds_survived():
    """The deeper the bracket runs, the more the points ride on rank."""
    phase_4.reset_rounds()
    early = _ranked_body(260, 400, match_id="round-1")
    first = phase_4._rank_pressure(early)
    for index in range(2, 6):
        phase_4._round_number(phase_4.TurnState(build_body(match_id=f"round-{index}")))
    later = _ranked_body(260, 400, match_id="round-6")
    assert phase_4._rank_pressure(later) > first
    assert phase_4._rank_pressure(later) <= phase_4.RANK_AMBITION_MAX


def test_rank_ambition_switches_off_in_the_closing_hands():
    phase_4.reset_rounds()
    late = _ranked_body(260, 400, hand_number=198, total_hands=200)
    assert phase_4._rank_pressure(late) == 0.0


def test_a_resolved_codename_is_pinned_for_the_next_bracket_game():
    """Phase 4 has no retries, so a rule learned once must not be relearned."""
    phase_2.KNOWN_CODENAMES.pop("fresh-codename", None)
    try:
        rule = phase_2._rule_for("fresh-codename")
        truth = phase_2.HYPOTHESES["standard"]
        rng = random.Random(4)
        for _ in range(40):
            community, left, right = (rng.randint(1, 13) for _ in range(3))
            rule.observe(community, left, right, truth(left, right, community))

        assert sorted(rule.candidates) == ["standard"]
        assert phase_2.KNOWN_CODENAMES["fresh-codename"] == "standard"
        # A fresh model for the same codename now starts from certainty.
        phase_4.reset_models()
        assert phase_2._rule_for("fresh-codename").confidence == 1.0
    finally:
        phase_2.KNOWN_CODENAMES.pop("fresh-codename", None)
