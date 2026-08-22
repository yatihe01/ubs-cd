import random

import pytest

from app import create_app
from challenges.showdown import phase_2, phase_3


@pytest.fixture(autouse=True)
def clean_models():
    phase_3.reset_models()
    yield
    phase_3.reset_models()


@pytest.fixture()
def client():
    app = create_app()
    app.config.update(TESTING=True)
    return app.test_client()


def player(seat, name, *, folded=False, busted=False, delta=0, bet=0, stack=200):
    return {
        "seat": seat,
        "name": name,
        "folded": folded,
        "busted": busted,
        "chip_delta": delta,
        "bet_this_round": bet,
        "stack": stack,
        "all_in": False,
    }


def build_body(**overrides):
    body = {
        "protocol_version": 2,
        "match_id": "phase3-attempt-1-leg-1",
        "phase": 3,
        "table_rule": "standard",
        "leg_number": 1,
        "total_legs": 4,
        "small_blind": 1,
        "big_blind": 2,
        "starting_stack": 200,
        "your_stack": 200,
        "hand_number": 5,
        "total_hands": 60,
        "round": "pre_reveal",
        "your_number": 7,
        "community_number": None,
        "your_seat": 0,
        "button_seat": 0,
        "pot": 3,
        "to_call": 2,
        "min_raise_to": 4,
        "max_raise_to": 200,
        "legal_actions": ["fold", "call", "raise"],
        "players": [
            player(0, "you", bet=0),
            player(1, "Dana", bet=1, stack=199),
            player(2, "Miles", bet=2, stack=198),
            player(3, "Theo"),
            player(4, "Rhea"),
            player(5, "Bram"),
        ],
        "current_hand_actions": [],
        "recent_hands": [],
    }
    body.update(overrides)
    return body


def test_phase3_dispatches_at_both_registered_urls(client):
    for endpoint in ("/move", "/showdown/move"):
        response = client.post(endpoint, json=build_body())
        assert response.status_code == 200
        assert response.get_json()["action"] in build_body()["legal_actions"]


def test_state_filters_folded_and_busted_but_keeps_table_seating():
    players = build_body()["players"]
    players[2]["folded"] = True
    players[4]["busted"] = True
    state = phase_3.TurnState(build_body(players=players))

    assert len(state.players) == 6
    assert {p["seat"] for p in state.active_players} == {0, 1, 2, 3, 5}
    assert {p["seat"] for p in state.live_opponents} == {1, 3, 5}


def test_joint_equity_is_lower_than_heads_up_and_handles_split_pots_exactly():
    rule = phase_2.RuleModel("standard")
    uniform = {number: 1.0 for number in phase_3.NUMBERS}
    heads_up = phase_3.multiway_equity(rule, 5, 5, [uniform])
    six_way = phase_3.multiway_equity(rule, 5, 5, [uniform] * 5)

    # A pair cannot lose under standard rules. It only splits with opponents
    # also holding 5, so K ~ Binomial(5, 1/13).
    exact = sum(
        probability / (ties + 1)
        for ties, probability in enumerate(_binomial_probabilities(5, 1 / 13))
    )
    assert heads_up == pytest.approx(12.5 / 13)
    assert six_way == pytest.approx(exact)
    assert six_way < heads_up


def _binomial_probabilities(trials, probability):
    values = [1.0]
    for _ in range(trials):
        updated = [0.0] * (len(values) + 1)
        for count, mass in enumerate(values):
            updated[count] += mass * (1 - probability)
            updated[count + 1] += mass * probability
        values = updated
    return values


def test_multiway_showdown_adds_only_proven_pairwise_constraints(client):
    hand = {
        "hand_number": 4,
        "community_number": 6,
        "winners": [0],
        "shown_numbers": {"0": 12, "1": 10, "2": 8, "3": 4},
        "actions": [],
    }
    body = build_body(table_rule="crowded-rule", recent_hands=[hand])
    client.post("/showdown/move", json=body)
    client.post("/showdown/move", json=body)

    model = phase_3._rule_for("crowded-rule")
    assert len(model.observations) == 3  # winner versus each loser, not loser/loser
    assert model.win_share(12, 10, 6) == 1.0
    assert phase_3.rule_snapshot()["crowded-rule"]["observations"] == 3


def test_future_blinds_rotate_six_ways_and_skip_busted_seats():
    # Five future hands from button 0 make seat 0 BB once and SB once: cost 3.
    state = phase_3.TurnState(build_body(hand_number=55, total_hands=60))
    assert phase_3._future_blind_cost(state) == 3

    # Bust seats 2 and 4. Future buttons are 1, 3, 5, 0; seat 0 posts BB
    # behind button 3 and SB behind button 5.
    players = build_body()["players"]
    players[2]["busted"] = True
    players[4]["busted"] = True
    state = phase_3.TurnState(
        build_body(players=players, hand_number=56, total_hands=60)
    )
    assert phase_3._future_blind_cost(state) == 3


def test_lock_requires_plus_ten_and_strictly_beating_every_seat():
    players = build_body()["players"]
    players[0].update(chip_delta=15, stack=215)
    players[3].update(chip_delta=15, stack=215)
    tied = phase_3.TurnState(
        build_body(
            players=players,
            your_stack=215,
            pot=0,
            hand_number=60,
            total_hands=60,
        )
    )
    assert not phase_3._endgame_locked(tied)

    players[0].update(chip_delta=16, stack=216)
    ahead = phase_3.TurnState(
        build_body(
            players=players,
            your_stack=216,
            pot=0,
            hand_number=60,
            total_hands=60,
        )
    )
    assert phase_3._endgame_locked(ahead)


def test_a_majority_stack_is_banked_instead_of_value_bet_back():
    players = build_body()["players"]
    players[0].update(chip_delta=450, stack=650, bet_this_round=0)
    players[1].update(busted=True, stack=0, chip_delta=-200)
    players[2].update(busted=True, stack=0, chip_delta=-200)
    players[5].update(busted=True, stack=0, chip_delta=-200)
    players[3].update(stack=300, chip_delta=100)
    players[4].update(stack=250, chip_delta=50)
    state = phase_3.TurnState(
        build_body(
            players=players,
            your_stack=650,
            pot=0,
            to_call=0,
            legal_actions=["check", "bet"],
            hand_number=20,
            total_hands=60,
        )
    )

    assert phase_3._endgame_locked(state)
    assert phase_3.decide(state) == {"action": "check"}


def test_a_protected_lead_folds_when_facing_a_bet():
    players = build_body()["players"]
    players[0].update(chip_delta=80, stack=275, bet_this_round=5)
    players[3].update(stack=240, chip_delta=40, bet_this_round=20)
    state = phase_3.TurnState(
        build_body(
            players=players,
            your_stack=275,
            pot=25,
            to_call=15,
            legal_actions=["fold", "call", "raise"],
            hand_number=60,
            total_hands=60,
        )
    )

    assert phase_3._protect_late_lead(state)
    assert phase_3.decide(state) == {"action": "fold"}


def test_phase3_fuzz_always_returns_a_legal_move(client):
    rng = random.Random(20260824)
    actions = ["fold", "call", "check", "bet", "raise"]
    for index in range(250):
        legal = rng.sample(actions, rng.randint(1, 4))
        low = high = None
        if {"bet", "raise"} & set(legal):
            low = rng.randint(2, 40)
            high = low + rng.randint(0, 180)
        players = build_body()["players"]
        for opponent in players[1:]:
            opponent["folded"] = rng.random() < 0.25
            opponent["busted"] = rng.random() < 0.08
        body = build_body(
            match_id=f"fuzz-{index // 50}",
            table_rule=f"rule-{index % 4}",
            players=players,
            legal_actions=legal,
            min_raise_to=low,
            max_raise_to=high,
            your_number=rng.choice([*phase_3.NUMBERS, None, "bad"]),
            community_number=rng.choice([*phase_3.NUMBERS, None]),
            to_call=rng.randint(0, 100),
            pot=rng.randint(0, 300),
        )
        response = client.post("/showdown/move", json=body)
        move = response.get_json()
        assert response.status_code == 200
        assert move["action"] in legal
        if move["action"] in ("bet", "raise"):
            assert low <= move["amount"] <= high
        else:
            assert "amount" not in move


def test_partial_rule_knowledge_does_not_price_the_nuts_as_a_loser():
    """Regression: uncertainty must not read as a certain loss.

    ``win_share`` is a posterior probability.  The previous equity engine
    bucketed it with ``share == 1.0`` / ``share == 0.5`` and let every value
    in between fall through as a loss, so while several hypotheses were still
    alive a pair - unbeatable under ``standard`` - scored exactly 0.0 equity
    and the bot folded it getting 5:1.
    """
    truth = phase_2.HYPOTHESES["standard"]
    rule = phase_2._rule_for("partly-known")
    for community, left, right in [(5, 9, 2), (11, 3, 7)]:
        rule.observe(community, left, right, truth(left, right, community))

    assert len(rule.candidates) > 1, "the rule must still be genuinely uncertain"
    uniform = {number: 1.0 for number in phase_3.NUMBERS}
    pair = phase_3.multiway_equity(rule, 7, 7, [uniform] * 5)
    trash = phase_3.multiway_equity(rule, 2, 9, [uniform] * 5)

    assert pair > phase_3.fair_share(5)
    assert pair > trash


def test_an_unrecognised_rule_still_separates_strong_from_weak():
    """With no hypothesis left, the empirical ranking must still price hands.

    Previously ``_empirical_compare`` returned values that were never exactly
    1.0, so every holding - best and worst alike - collapsed to ~0 equity for
    the rest of the leg.
    """
    rule = phase_2._rule_for("unrecognised")
    rule._candidates = set()
    for left in range(1, 14):
        for right in range(1, 14):
            if left != right:
                rule.observe(7, left, right, 1 if left > right else -1)

    uniform = {number: 1.0 for number in phase_3.NUMBERS}
    best = phase_3.multiway_equity(rule, 13, 7, [uniform] * 5)
    worst = phase_3.multiway_equity(rule, 1, 7, [uniform] * 5)
    assert best > phase_3.fair_share(5) > worst


def test_thresholds_scale_with_the_number_of_live_opponents():
    """A 0.52 share is average heads-up and enormous six-handed."""
    assert phase_3.fair_share(1) == pytest.approx(0.5)
    assert phase_3.fair_share(5) == pytest.approx(1 / 6)
    assert phase_3._relative_floor(5, 0.34) < phase_3._relative_floor(1, 0.34)
