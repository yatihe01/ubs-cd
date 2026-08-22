"""Time Travelling Stonks Man.

The rule checks below replay each answer against the brief rather than asserting a
particular action list, because many itineraries earn the same money and pinning
one of them would make the tests fight future improvements.
"""

import pytest

from app import create_app
from challenges.stonks.solution import START_YEAR, solve_case


SAMPLE = {
    "energy": 2,
    "capital": 500,
    "timeline": {
        "2037": {"Apple": {"price": 100, "qty": 10}},
        "2036": {"Apple": {"price": 10, "qty": 50}},
    },
}


def replay(case, actions):
    """Execute an action list under the brief's rules, returning the final cash.

    Raises on anything the rules forbid: jumping from the wrong year, overspending,
    overselling, exhausting a year's supply, running the battery flat, or failing
    to get home.
    """
    energy = case["energy"]
    cash = case["capital"]
    supply = {
        (int(year), stock): listing["qty"]
        for year, market in case["timeline"].items()
        for stock, listing in market.items()
    }
    prices = {
        (int(year), stock): listing["price"]
        for year, market in case["timeline"].items()
        for stock, listing in market.items()
    }

    year = START_YEAR
    spent = 0
    holdings = {}

    for action in actions:
        kind, first, second = action.split("-")
        if kind == "j":
            assert int(first) == year, f"{action}: standing in {year}"
            destination = int(second)
            assert 0 < destination <= START_YEAR, f"{action}: year out of range"
            spent += abs(destination - year)
            assert spent <= energy, f"{action}: used {spent} of {energy} energy"
            year = destination
        elif kind == "b":
            quantity = int(second)
            assert quantity > 0, f"{action}: non-positive quantity"
            assert (year, first) in prices, f"{action}: {first} not listed in {year}"
            assert quantity <= supply[(year, first)], f"{action}: not enough supply"
            cost = quantity * prices[(year, first)]
            assert cost <= cash, f"{action}: costs {cost}, holding {cash}"
            supply[(year, first)] -= quantity
            cash -= cost
            holdings[first] = holdings.get(first, 0) + quantity
        elif kind == "s":
            quantity = int(second)
            assert quantity > 0, f"{action}: non-positive quantity"
            assert (year, first) in prices, f"{action}: {first} not listed in {year}"
            assert quantity <= holdings.get(first, 0), f"{action}: not holding that"
            holdings[first] -= quantity
            cash += quantity * prices[(year, first)]
        else:
            raise AssertionError(f"unknown action type in {action!r}")

    assert year == START_YEAR, f"ended in {year}, must return to {START_YEAR}"
    return cash


# --- the brief's worked example ---------------------------------------------------

def test_sample_reproduces_the_documented_itinerary():
    assert solve_case(SAMPLE) == [
        "j-2037-2036",
        "b-Apple-50",
        "j-2036-2037",
        "s-Apple-50",
    ]


def test_sample_earns_the_documented_profit():
    assert replay(SAMPLE, solve_case(SAMPLE)) - SAMPLE["capital"] == 4500


# --- the rules ---------------------------------------------------------------------

def test_energy_limits_how_far_back_we_reach():
    """Two energy buys one year of round trip, so the bargain two years back is out
    of reach and the answer must not pretend otherwise."""
    case = {
        "energy": 2,
        "capital": 100,
        "timeline": {
            "2037": {"Widget": {"price": 50, "qty": 10}},
            "2035": {"Widget": {"price": 1, "qty": 10}},
        },
    }
    assert solve_case(case) == []


def test_a_deeper_bargain_is_taken_once_the_battery_allows_it():
    case = {
        "energy": 4,
        "capital": 100,
        "timeline": {
            "2037": {"Widget": {"price": 50, "qty": 10}},
            "2035": {"Widget": {"price": 1, "qty": 10}},
        },
    }
    assert replay(case, solve_case(case)) == 100 - 10 + 500


def test_a_years_supply_is_never_oversold():
    case = {
        "energy": 2,
        "capital": 10_000,
        "timeline": {
            "2037": {"Widget": {"price": 100, "qty": 0}},
            "2036": {"Widget": {"price": 1, "qty": 7}},
        },
    }
    actions = solve_case(case)
    assert replay(case, actions) == 10_000 - 7 + 700
    assert "b-Widget-7" in actions


def test_zero_quantity_listings_are_still_somewhere_to_sell():
    """`qty` caps buying, not selling - the brief's own sample sells 50 into a year
    listing 10 - so a zero-quantity year is a real exit at that price."""
    case = {
        "energy": 2,
        "capital": 10,
        "timeline": {
            "2037": {"Widget": {"price": 2, "qty": 5}},
            "2036": {"Widget": {"price": 9, "qty": 0}},
        },
    }
    assert replay(case, solve_case(case)) == 45


def test_nothing_is_bought_when_no_trade_turns_a_profit():
    case = {
        "energy": 6,
        "capital": 100,
        "timeline": {
            "2037": {"Widget": {"price": 7, "qty": 5}},
            "2036": {"Widget": {"price": 7, "qty": 5}},
            "2035": {"Widget": {"price": 7, "qty": 5}},
        },
    }
    # A flat price everywhere means every round trip breaks even at best.  The
    # itinerary runs down and back, so a price that merely *rose* into the past
    # would still be tradeable on the way down - flatness is what rules it out.
    assert solve_case(case) == []


def test_a_price_that_rises_into_the_past_is_traded_on_the_way_down():
    """Buying before descending is a real move: the descent reaches the higher
    price, and the ascent still gets us home."""
    case = {
        "energy": 4,
        "capital": 100,
        "timeline": {
            "2037": {"Widget": {"price": 1, "qty": 5}},
            "2035": {"Widget": {"price": 9, "qty": 0}},
        },
    }
    assert replay(case, solve_case(case)) == 100 - 5 + 45


def test_capital_is_never_overspent():
    case = {
        "energy": 2,
        "capital": 5,
        "timeline": {
            "2037": {"Widget": {"price": 100, "qty": 10}},
            "2036": {"Widget": {"price": 2, "qty": 10}},
        },
    }
    assert replay(case, solve_case(case)) == 5 - 4 + 200


# --- sequencing ---------------------------------------------------------------------

def test_profit_from_an_early_trade_funds_a_later_one():
    """The 2035 bargain costs more than the starting capital, but the 2036 sale
    lands before we get there, so a plan that spends first can still afford it."""
    case = {
        "energy": 4,
        "capital": 100,
        "timeline": {
            "2037": {"Alpha": {"price": 100, "qty": 1}},
            "2036": {"Alpha": {"price": 400, "qty": 0}, "Beta": {"price": 20, "qty": 20}},
            "2035": {"Beta": {"price": 1, "qty": 20}},
        },
    }
    actions = solve_case(case)
    earned = replay(case, actions)
    assert "b-Alpha-1" in actions
    assert earned > 400  # the Alpha flip alone would stop at 400


def test_a_cheap_trade_does_not_starve_a_better_one():
    """Spending everything on the mediocre 2037 trade would leave nothing for the
    far better 2036 one, which has to be bought before it can be sold."""
    case = {
        "energy": 4,
        "capital": 43,
        "timeline": {
            "2037": {"Delta": {"price": 15, "qty": 5}},
            "2036": {"Delta": {"price": 7, "qty": 4}},
            "2035": {"Delta": {"price": 20, "qty": 3}},
        },
    }
    assert replay(case, solve_case(case)) == 100


def test_holdings_are_carried_to_their_best_price_not_the_first_one():
    case = {
        "energy": 6,
        "capital": 100,
        "timeline": {
            "2037": {"Widget": {"price": 30, "qty": 0}},
            "2036": {"Widget": {"price": 12, "qty": 0}},
            "2035": {"Widget": {"price": 2, "qty": 10}},
        },
    }
    # Bought at 2035, it must ride past 2036's 12 to reach 2037's 30.
    assert replay(case, solve_case(case)) == 100 - 20 + 300


# --- spending the battery on laps rather than distance --------------------------------

def test_a_short_stretch_is_run_repeatedly_to_compound():
    """One pass buys only what the starting capital affords.  Running the same
    stretch again turns each lap's profit into the next lap's stake, which is worth
    far more than the extra years a deeper trip would have reached."""
    case = {
        "energy": 10,
        "capital": 10,
        "timeline": {
            "2037": {"W": {"price": 2, "qty": 0}},
            "2036": {"W": {"price": 1, "qty": 1000}},
        },
    }
    # Five laps at doubling: 10 -> 20 -> 40 -> 80 -> 160 -> 320.  A single pass,
    # which is all a straight there-and-back allows, would stop at 20.
    assert replay(case, solve_case(case)) == 320


def test_laps_stop_once_the_supply_is_exhausted():
    """Compounding is capped by what there is to buy, not by the battery."""
    case = {
        "energy": 20,
        "capital": 10,
        "timeline": {
            "2037": {"W": {"price": 5, "qty": 0}},
            "2036": {"W": {"price": 1, "qty": 12}},
        },
    }
    # Only 12 shares exist; every one ends up sold at 5, so 10 - 12 + 60.
    assert replay(case, solve_case(case)) == 58


def test_depth_still_wins_when_the_bargain_is_only_deep():
    """Laps must not crowd out a trip that has to be long to be worth anything."""
    case = {
        "energy": 6,
        "capital": 100,
        "timeline": {
            "2037": {"Gem": {"price": 100, "qty": 0}},
            "2036": {"Dud": {"price": 99, "qty": 5}},
            "2034": {"Gem": {"price": 1, "qty": 50}},
        },
    }
    actions = solve_case(case)
    # All 50 Gems at 1, resold at 100: 100 - 50 + 5000.
    assert replay(case, actions) == 5050
    assert "j-2037-2034" in actions


def test_compounding_near_home_can_pay_for_a_deeper_trip():
    """With too little capital to use the deep bargain on arrival, the winning
    shape is laps close to home first, then one dive spending the winnings."""
    case = {
        "energy": 6,
        "capital": 2,
        "timeline": {
            "2037": {"A": {"price": 1, "qty": 1}, "B": {"price": 5, "qty": 2}},
            "2036": {"B": {"price": 1, "qty": 2}, "A": {"price": 1, "qty": 0}},
            "2035": {"B": {"price": 2, "qty": 1}, "A": {"price": 2, "qty": 1}},
        },
    }
    # Diving straight to 2035 reaches the bargain with nothing to spend.
    assert replay(case, solve_case(case)) == 14


def test_laps_never_overrun_the_battery():
    case = {
        "energy": 7,
        "capital": 5,
        "timeline": {
            "2037": {"W": {"price": 3, "qty": 0}},
            "2036": {"W": {"price": 1, "qty": 500}},
        },
    }
    actions = solve_case(case)
    spent = sum(
        abs(int(parts[2]) - int(parts[1]))
        for parts in (action.split("-") for action in actions)
        if parts[0] == "j"
    )
    assert spent <= 7
    replay(case, actions)  # re-checks every rule, including the energy budget


# --- malformed input ------------------------------------------------------------------

@pytest.mark.parametrize(
    "case",
    [
        {},
        {"energy": 2, "capital": 100},
        {"energy": 2, "capital": 100, "timeline": {}},
        {"energy": 2, "capital": 100, "timeline": None},
        {"energy": 0, "capital": 100, "timeline": {"2036": {"A": {"price": 1, "qty": 1}}}},
        {"energy": 2, "capital": 0, "timeline": {"2036": {"A": {"price": 1, "qty": 1}}}},
        {"energy": 2, "capital": 100, "timeline": {"2036": {"A": {"price": 0, "qty": 1}}}},
        {"energy": 2, "capital": 100, "timeline": {"2036": {"A": None}}},
        {"energy": 2, "capital": 100, "timeline": {"nope": {"A": {"price": 1, "qty": 1}}}},
        "not a case",
        None,
    ],
)
def test_unusable_cases_yield_an_empty_plan_rather_than_failing(case):
    assert solve_case(case) == []


def test_years_after_2037_are_ignored():
    """Time travel only runs backwards, so a future listing is unreachable."""
    case = {
        "energy": 4,
        "capital": 100,
        "timeline": {
            "2038": {"Widget": {"price": 9_999, "qty": 10}},
            "2037": {"Widget": {"price": 100, "qty": 10}},
            "2036": {"Widget": {"price": 1, "qty": 10}},
        },
    }
    actions = solve_case(case)
    assert all("2038" not in action for action in actions)
    assert replay(case, actions) == 100 - 10 + 1000


def test_unknown_fields_do_not_break_processing():
    case = {
        "energy": 2,
        "capital": 500,
        "somethingNew": True,
        "timeline": {
            "2037": {"Apple": {"price": 100, "qty": 10, "extra": "x"}},
            "2036": {"Apple": {"price": 10, "qty": 50}},
        },
    }
    assert replay(case, solve_case(case)) == 5000


# --- endpoint --------------------------------------------------------------------------

@pytest.fixture()
def client():
    app = create_app()
    app.config.update(TESTING=True)
    return app.test_client()


def test_endpoint_returns_one_plan_per_case_in_order(client):
    second = {
        "energy": 4,
        "capital": 100,
        "timeline": {
            "2037": {"Widget": {"price": 50, "qty": 10}},
            "2035": {"Widget": {"price": 1, "qty": 10}},
        },
    }
    response = client.post("/stonks", json=[SAMPLE, second])
    assert response.status_code == 200
    body = response.get_json()
    assert isinstance(body, list) and len(body) == 2
    assert body[0] == solve_case(SAMPLE)
    assert body[1] == solve_case(second)


def test_endpoint_accepts_an_empty_batch(client):
    assert client.post("/stonks", json=[]).get_json() == []


def test_endpoint_rejects_a_non_array_body(client):
    assert client.post("/stonks", json={"energy": 2}).status_code == 400


def test_endpoint_reads_the_body_without_a_json_content_type(client):
    """The grader is not guaranteed to set the header."""
    import json

    response = client.post(
        "/stonks", data=json.dumps([SAMPLE]), content_type="text/plain"
    )
    assert response.status_code == 200
    assert response.get_json() == [solve_case(SAMPLE)]
