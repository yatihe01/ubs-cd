import random

import pytest

from app import create_app
from challenges.showdown import phase_2


@pytest.fixture(autouse=True)
def clean_models():
    phase_2.reset_models()
    yield
    phase_2.reset_models()


@pytest.fixture()
def client():
    app = create_app()
    app.config.update(TESTING=True)
    return app.test_client()


def build_body(**overrides):
    body = {
        "protocol_version": 2,
        "match_id": "phase2-attempt-1-leg-1",
        "phase": 2,
        "table_rule": "chalcedony",
        "leg_number": 1,
        "total_legs": 4,
        "small_blind": 1,
        "big_blind": 2,
        "starting_stack": 200,
        "your_stack": 199,
        "hand_number": 5,
        "total_hands": 40,
        "round": "pre_reveal",
        "your_number": 7,
        "community_number": None,
        "your_seat": 0,
        "button_seat": 0,
        "pot": 3,
        "to_call": 1,
        "min_raise_to": 4,
        "max_raise_to": 199,
        "legal_actions": ["fold", "call", "raise"],
        "players": [
            {
                "seat": 0,
                "name": "you",
                "folded": False,
                "chip_delta": 0,
                "bet_this_round": 1,
                "stack": 199,
                "all_in": False,
                "busted": False,
            },
            {
                "seat": 1,
                "name": "Gaston",
                "folded": False,
                "chip_delta": 0,
                "bet_this_round": 2,
                "stack": 198,
                "all_in": False,
                "busted": False,
            },
        ],
        "current_hand_actions": [],
        "recent_hands": [],
    }
    body.update(overrides)
    return body


def showdown_hand(hand_number=4, community=6, ours=11, theirs=3, winners=None):
    return {
        "hand_number": hand_number,
        "community_number": community,
        "winners": [0] if winners is None else winners,
        "pot": 3,
        "shown_numbers": {"0": ours, "1": theirs},
        "actions": [
            {"round": "pre_reveal", "seat": 0, "action": "call", "amount": 2},
            {"round": "pre_reveal", "seat": 1, "action": "check"},
            {"round": "post_reveal", "seat": 1, "action": "check"},
            {"round": "post_reveal", "seat": 0, "action": "check"},
        ],
    }


def test_unknown_rule_checks_and_buys_only_a_cheap_showdown(client):
    free = build_body(
        round="post_reveal",
        community_number=4,
        to_call=0,
        legal_actions=["check", "bet"],
    )
    assert client.post("/showdown/move", json=free).get_json() == {
        "action": "check"
    }

    cheap = build_body(to_call=3)
    assert client.post("/showdown/move", json=cheap).get_json() == {
        "action": "call"
    }

    expensive = build_body(pot=20, to_call=30)
    assert client.post("/showdown/move", json=expensive).get_json() == {
        "action": "fold"
    }


def test_completed_showdown_is_hard_evidence_and_is_deduplicated(client):
    body = build_body(recent_hands=[showdown_hand()])

    client.post("/showdown/move", json=body)
    client.post("/showdown/move", json=body)

    snapshot = phase_2.rule_snapshot()["chalcedony"]
    assert snapshot["observations"] == 1
    model = phase_2._rule_for("chalcedony")
    assert model.win_share(11, 3, 6) == 1.0
    assert model.win_share(3, 11, 6) == 0.0


def test_rule_knowledge_survives_leg_and_match_boundaries(client):
    first = build_body(recent_hands=[showdown_hand()])
    client.post("/showdown/move", json=first)

    retry = build_body(
        match_id="phase2-attempt-2-leg-3",
        leg_number=3,
        hand_number=1,
        recent_hands=[],
    )
    client.post("/showdown/move", json=retry)

    assert phase_2.rule_snapshot()["chalcedony"]["observations"] == 1


def test_folded_hand_does_not_create_a_showdown_constraint(client):
    hand = showdown_hand()
    hand["community_number"] = None
    hand["shown_numbers"] = {}
    hand["winners"] = [0]

    client.post("/showdown/move", json=build_body(recent_hands=[hand]))

    assert phase_2.rule_snapshot()["chalcedony"]["observations"] == 0


def test_candidate_ensemble_identifies_standard_shape():
    model = phase_2.RuleModel("opaque-standard-like")
    comparator = phase_2.HYPOTHESES["standard"]
    for community in phase_2.NUMBERS:
        for left in phase_2.NUMBERS:
            for right in phase_2.NUMBERS:
                model.observe(
                    community,
                    left,
                    right,
                    comparator(left, right, community),
                )

    assert model.candidates == {"standard"}
    assert model.confidence >= phase_2.EXPLORE_MIN_CONFIDENCE
    weights = {number: 1.0 for number in phase_2.NUMBERS}
    assert phase_2.equity_vs_range(model, 5, 5, weights) == pytest.approx(
        12.5 / 13
    )


def test_an_unrecognized_rule_falls_back_to_exact_empirical_comparisons():
    model = phase_2.RuleModel("strange")
    # A deliberately irregular cycle eliminates every total-order hypothesis.
    model.observe(4, 1, 2, 1)
    model.observe(4, 2, 3, 1)
    model.observe(4, 3, 1, 1)

    assert not model.candidates
    assert model.win_share(1, 2, 4) == 1.0
    assert model.win_share(2, 1, 4) == 0.0


def test_exact_lock_accounts_for_current_commitment_and_future_blinds():
    body = build_body(
        hand_number=38,
        total_hands=40,
        button_seat=0,
        your_stack=225,
        pot=30,
        to_call=10,
        legal_actions=["fold", "call", "raise"],
    )
    body["players"][0]["chip_delta"] = 30
    body["players"][0]["stack"] = 225  # five already committed this hand
    state = phase_2.TurnState(body)

    # Current worst-case loss 5; future blinds are 2 then 1.  +30 cannot lock
    # +25 yet, whereas +33 can do so exactly.
    assert not phase_2._endgame_locked(state)
    body["players"][0]["chip_delta"] = 33
    body["players"][0]["stack"] = 228
    body["your_stack"] = 228
    assert phase_2._endgame_locked(phase_2.TurnState(body))


def test_phase2_fuzz_never_returns_an_illegal_action(client):
    rng = random.Random(20260823)
    actions = ["fold", "call", "check", "bet", "raise"]
    for index in range(250):
        legal = rng.sample(actions, rng.randint(1, 4))
        low = high = None
        if {"bet", "raise"} & set(legal):
            low = rng.randint(2, 40)
            high = low + rng.randint(0, 160)
        body = build_body(
            match_id=f"fuzz-{index // 40}",
            table_rule=f"rule-{index % 4}",
            legal_actions=legal,
            min_raise_to=low,
            max_raise_to=high,
            your_number=rng.choice([*phase_2.NUMBERS, None, "bad"]),
            community_number=rng.choice([*phase_2.NUMBERS, None]),
            to_call=rng.randint(0, 100),
            pot=rng.randint(0, 200),
        )

        response = client.post("/showdown/move", json=body)
        move = response.get_json()

        assert response.status_code == 200
        assert move["action"] in legal
        if move["action"] in ("bet", "raise"):
            assert low <= move["amount"] <= high
        else:
            assert "amount" not in move
