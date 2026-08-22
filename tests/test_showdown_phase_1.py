import random

import pytest

from app import create_app
from challenges.showdown import phase_1


@pytest.fixture()
def client():
    app = create_app()
    app.config.update(TESTING=True)
    return app.test_client()


@pytest.fixture(autouse=True)
def clean_models():
    phase_1.reset_models()
    yield
    phase_1.reset_models()


def build_body(**overrides):
    """A plausible /move request. Override only what a test cares about."""
    body = {
        "protocol_version": 2,
        "match_id": "phase1-test",
        "phase": 1,
        "table_rule": "standard",
        "small_blind": 1,
        "big_blind": 2,
        "starting_stack": 200,
        "your_stack": 200,
        "hand_number": 5,
        "total_hands": 100,
        "round": "pre_reveal",
        "your_number": 7,
        "community_number": None,
        "your_seat": 0,
        "button_seat": 1,
        "pot": 3,
        "to_call": 1,
        "min_raise_to": 4,
        "max_raise_to": 200,
        "legal_actions": ["fold", "call", "raise"],
        "players": [
            {"seat": 0, "name": "you", "folded": False, "chip_delta": 0,
             "bet_this_round": 1, "stack": 199, "all_in": False, "busted": False},
            {"seat": 1, "name": "Gaston", "folded": False, "chip_delta": 0,
             "bet_this_round": 2, "stack": 198, "all_in": False, "busted": False},
        ],
        "current_hand_actions": [],
        "recent_hands": [],
    }
    body.update(overrides)
    return body


# --------------------------------------------------------------------------
# Layer 1: the equity is a closed form, so it can be asserted exactly
# --------------------------------------------------------------------------

def test_pair_is_almost_the_nuts():
    # Only an opponent holding the community number ties; nothing beats us.
    assert phase_1.equity_post(5, 5) == pytest.approx(12.5 / 13)


def test_unpaired_equity_accounts_for_the_community_number():
    # Community below us kills one of our outs: it makes them a pair instead.
    assert phase_1.equity_post(13, 1) == pytest.approx(11.5 / 13)
    # Community above us was beating us anyway, so nothing is lost.
    assert phase_1.equity_post(12, 13) == pytest.approx(11.5 / 13)


def test_pre_reveal_equity_is_monotone_and_centred_on_seven():
    assert phase_1.PRE_EQUITY[7] == pytest.approx(0.5)
    ladder = phase_1.PRE_EQUITY[1:]
    assert ladder == tuple(sorted(ladder))


def test_equity_against_a_uniform_range_matches_the_closed_form():
    weights = {n: 1.0 for n in phase_1.NUMBERS}
    assert phase_1.equity_vs_range(9, 4, weights) == pytest.approx(
        phase_1.equity_post(9, 4)
    )
    assert phase_1.equity_vs_range(9, None, weights) == pytest.approx(
        phase_1.PRE_EQUITY[9]
    )


# --------------------------------------------------------------------------
# Layer 2: the opponent read is mined from the completed hand logs
# --------------------------------------------------------------------------

def test_opponent_model_counts_each_hand_once():
    model = phase_1.OpponentModel()
    hands = [{
        "hand_number": 2,
        "actions": [
            {"round": "pre_reveal", "seat": 1, "action": "raise", "amount": 5},
            {"round": "post_reveal", "seat": 0, "action": "bet", "amount": 7},
            {"round": "post_reveal", "seat": 1, "action": "fold"},
        ],
    }]

    model.observe_completed_hands(hands, opponent_seat=1)
    model.observe_completed_hands(hands, opponent_seat=1)  # same window, seen again

    assert model.actions == 2          # seat 0 is us and is not counted
    assert model.aggressive == 1
    assert model.faced_bet == 2        # a raise and a fold both face a live bet
    assert model.folds == 1


# --------------------------------------------------------------------------
# Layer 3/4: the decisions themselves
# --------------------------------------------------------------------------

def test_trash_folds_to_a_big_bet(client):
    body = build_body(
        round="post_reveal", your_number=2, community_number=9,
        pot=60, to_call=40, min_raise_to=80, max_raise_to=200,
        legal_actions=["fold", "call", "raise"],
    )

    assert client.post("/showdown/move", json=body).get_json() == {"action": "fold"}


def test_a_pair_raises_and_stays_inside_the_legal_range(client):
    body = build_body(
        round="post_reveal", your_number=9, community_number=9,
        pot=20, to_call=0, min_raise_to=4, max_raise_to=190,
        legal_actions=["check", "bet"],
    )

    move = client.post("/showdown/move", json=body).get_json()

    assert move["action"] == "bet"
    assert 4 <= move["amount"] <= 190


def test_free_card_is_taken_with_a_middling_hand(client):
    # Nothing to match and no fold equity worth buying: check it down.
    body = build_body(
        round="post_reveal", your_number=7, community_number=3,
        pot=8, to_call=0, min_raise_to=2, max_raise_to=196,
        legal_actions=["check", "bet"],
    )

    assert client.post("/showdown/move", json=body).get_json() == {"action": "check"}


def test_endgame_lock_folds_a_good_hand_once_the_phase_is_banked(client):
    # +90 with 4 hands left: folding out costs 6 and still clears +10.
    body = build_body(
        hand_number=96, total_hands=100,
        round="post_reveal", your_number=11, community_number=4,
        pot=40, to_call=20, min_raise_to=40, max_raise_to=200,
        legal_actions=["fold", "call", "raise"],
    )
    body["players"][0]["chip_delta"] = 90

    assert client.post("/showdown/move", json=body).get_json() == {"action": "fold"}


def test_the_same_spot_is_played_when_the_phase_is_not_banked(client):
    body = build_body(
        hand_number=96, total_hands=100,
        round="post_reveal", your_number=11, community_number=4,
        pot=40, to_call=20, min_raise_to=40, max_raise_to=200,
        legal_actions=["fold", "call", "raise"],
    )
    body["players"][0]["chip_delta"] = 0

    assert client.post("/showdown/move", json=body).get_json()["action"] != "fold"


def shove_at_us(number, **overrides):
    body = build_body(
        round="post_reveal", your_number=number, community_number=3,
        your_stack=200, pot=210, to_call=190,
        min_raise_to=None, max_raise_to=None,
        legal_actions=["fold", "call"],
    )
    body.update(overrides)
    return body


def test_a_thin_edge_does_not_get_the_whole_stack_in(client):
    # A middling number is ahead of the bottom of their range and nothing else.
    move = client.post("/showdown/move", json=shove_at_us(9)).get_json()

    assert move == {"action": "fold"}


def test_a_big_price_on_a_strong_hand_is_taken(client):
    # 190 to win 400 needs 47.5%; a 12 with the community at 3 has 80.8%.
    move = client.post("/showdown/move", json=shove_at_us(12)).get_json()

    assert move == {"action": "call"}


def test_a_thin_edge_is_called_off_once_there_is_nothing_left_to_protect(client):
    # Busting only really costs the hands you had left to win it back in.
    body = shove_at_us(9, hand_number=100, total_hands=100)

    assert client.post("/showdown/move", json=body).get_json() == {"action": "call"}


# --------------------------------------------------------------------------
# Layer 0: the safety shell is worth more than any of the strategy above
# --------------------------------------------------------------------------

@pytest.mark.parametrize("body", [None, {}, [], {"legal_actions": []}])
def test_a_malformed_request_still_gets_a_legal_reply(client, body):
    response = client.post("/showdown/move", json=body)

    assert response.status_code == 200
    assert response.get_json()["action"] in phase_1.KNOWN_ACTIONS


def test_health_is_answered(client):
    response = client.get("/showdown/health")

    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}


def test_fuzzed_requests_never_produce_an_illegal_move(client):
    """Protocol-shaped noise: the reply must always be one they offered."""
    rng = random.Random(20260822)

    for _ in range(500):
        legal = rng.sample(
            ["fold", "call", "check", "bet", "raise"],
            rng.randint(1, 4),
        )
        # They only send a raise range when raising is actually on the table.
        low = high = None
        if {"bet", "raise"} & set(legal):
            low = rng.randint(2, 60)
            high = low + rng.randint(0, 200)

        body = build_body(
            round=rng.choice(["pre_reveal", "post_reveal"]),
            your_number=rng.choice([*phase_1.NUMBERS, None, "x"]),
            community_number=rng.choice([*phase_1.NUMBERS, None]),
            your_stack=rng.randint(1, 400),
            pot=rng.randint(0, 300),
            to_call=rng.randint(0, 200),
            min_raise_to=low,
            max_raise_to=high,
            legal_actions=legal,
            hand_number=rng.randint(1, 100),
            match_id=f"fuzz-{rng.randint(0, 5)}",
        )

        move = client.post("/showdown/move", json=body).get_json()

        assert move["action"] in legal, (move, legal)
        if move["action"] in ("bet", "raise"):
            assert low <= move["amount"] <= high, (move, low, high)
        else:
            assert "amount" not in move


def test_fuzzed_garbage_never_crashes_the_endpoint(client):
    """Fields we do not recognise, or nonsense in the ones we do, must not 500."""
    rng = random.Random(1234)
    junk = [None, "x", -1, 10**9, [], {}, True, 1.5]

    for _ in range(300):
        body = build_body()
        for key in rng.sample(sorted(body), rng.randint(1, 8)):
            body[key] = rng.choice(junk)
        body["future_field_we_have_never_seen"] = rng.choice(junk)

        response = client.post("/showdown/move", json=body)

        assert response.status_code == 200
        assert response.get_json()["action"] in phase_1.KNOWN_ACTIONS
