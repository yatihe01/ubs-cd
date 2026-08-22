"""Ghost Chains Phase 3: value signal on top of the structural and identity ones.

A Phase 3 evaluation re-tests Phases 1 and 2, so the first section pins the Phase 3
model to the *exact* Phase 2 scores whenever amounts carry no information.  The rest
covers the value layer and the cross-signal scenarios.
"""

from datetime import datetime, timedelta, timezone

import pytest

from app import create_app
from challenges.ghost_chains.solution2 import GhostChainsModel as PhaseTwoModel
from challenges.ghost_chains.solution2 import make_transaction as phase_two_transaction
from challenges.ghost_chains.solution3 import (
    GhostChainsModel,
    LOOKBACK,
    make_transaction,
)


BASE = datetime(2026, 6, 8, 12, 0, 0, tzinfo=timezone.utc)

M, A, C, H, S, O, N = (
    "meridian_holdings",
    "apex_logistics",
    "cascade_payments",
    "horizon_capital",
    "sterling_bridge",
    "oakridge_imports",
    "nimbus_trading",
)

DEV_A = "dev_ios_7f3a91"
DEV_B = "dev_android_c2e4b8"
IP_A = "10.0.0.1"
IP_B = "10.0.0.2"


def iso(moment: datetime) -> str:
    return moment.isoformat().replace("+00:00", "Z")


def payload_for(index, leg, step):
    sender, receiver, amount = leg[0], leg[1], leg[2]
    extra = leg[3] if len(leg) > 3 else {}
    body = {
        "txId": f"tx{index}",
        "fromUserId": sender,
        "toUserId": receiver,
        "amount": amount,
        "createdAt": iso(BASE + index * step),
    }
    body.update(extra)
    return body


def feed(legs, model=None, step=timedelta(minutes=1)):
    """Feed `(sender, receiver, amount[, extra fields])` legs one minute apart."""
    model = model or GhostChainsModel()
    return [
        model.process(make_transaction(payload_for(index, leg, step)))
        for index, leg in enumerate(legs)
    ]


def last(legs, model=None):
    return feed(legs, model)[-1]


def feed_phase_two(legs, step=timedelta(minutes=1)):
    model = PhaseTwoModel()
    return [
        model.process(phase_two_transaction(payload_for(index, leg, step)))
        for index, leg in enumerate(legs)
    ]


def dev(value):
    return {"deviceId": value}


def ip(value):
    return {"ipAddress": value}


# --- Phases 1 and 2 are re-tested in a Phase 3 run --------------------------------

UNIFORM_EXAMPLES = {
    "isolated": [(M, A, 100.0)],
    "extension": [(M, A, 100.0), (A, C, 100.0)],
    "convergence": [(M, A, 100.0), (M, H, 100.0), (A, S, 100.0), (H, S, 100.0)],
    "return": [(M, A, 100.0), (A, C, 100.0), (C, O, 100.0), (O, A, 100.0)],
    "multi_loop": [
        (M, A, 100.0),
        (A, C, 100.0),
        (C, M, 100.0),
        (A, N, 100.0),
        (N, M, 100.0),
    ],
    "identity_chain": [
        (M, A, 100.0, dev(DEV_A)),
        (A, C, 100.0, dev(DEV_A)),
        (C, H, 100.0, dev(DEV_A)),
    ],
    "identity_shift": [
        (M, A, 100.0, dev(DEV_A)),
        (A, C, 100.0, dev(DEV_A)),
        (C, H, 100.0, dev(DEV_B)),
    ],
    "identity_dropped": [
        (M, A, 100.0, dev(DEV_A)),
        (A, C, 100.0, dev(DEV_A)),
        (C, H, 100.0),
    ],
    "cross_component": [
        (M, A, 100.0, ip(IP_A)),
        (C, H, 100.0, ip(IP_A)),
        (O, S, 100.0, ip(IP_A)),
    ],
}


@pytest.mark.parametrize("name", sorted(UNIFORM_EXAMPLES))
def test_uniform_amounts_reproduce_phase_two_exactly(name):
    """The measured Phase 2 model is the reference.  Phase 3 may only *add* signal,
    so with no amount variation anywhere the two must agree to the last decimal."""
    legs = UNIFORM_EXAMPLES[name]
    assert feed(legs) == feed_phase_two(legs)


def test_phase_one_ordering_survives():
    ordering = ["isolated", "extension", "convergence", "return", "multi_loop"]
    values = [last(UNIFORM_EXAMPLES[name]) for name in ordering]
    assert values == sorted(values)
    assert len(set(values)) == len(values)
    assert all(0.0 <= value <= 1.0 for value in values)


# --- the four Phase 3 briefing examples -------------------------------------------

EX1_DECAY = [
    (M, A, 10000.0),
    (A, C, 9910.0),
    (C, H, 9820.81),
    (H, N, 9732.42),
]
EX2_BRANCHES = [
    (M, A, 10000.0),
    (A, C, 9800.0),
    (A, S, 5000.0),
    (C, H, 9700.0),
    (S, O, 4900.0),
]
EX3_REVERSAL = [
    (M, A, 10000.0),
    (A, C, 9950.0),
    (C, H, 9800.0),
    (H, N, 9950.0),
]
EX4_CONVERGENCE = [
    (M, A, 10000.0),
    (A, C, 9800.0),
    (A, S, 5000.0),
    (C, H, 9700.0),
    (S, H, 4950.0),
]

BRIEFING = [EX1_DECAY, EX2_BRANCHES, EX3_REVERSAL, EX4_CONVERGENCE]


@pytest.mark.parametrize("legs", BRIEFING, ids=range(1, 5))
def test_briefing_examples_stay_in_range(legs):
    assert all(0.0 <= value <= 1.0 for value in feed(legs))


def test_consistent_decay_is_the_lowest_of_the_four():
    """'Example 1 should receive the lowest risk score of the four.'  Consistent
    decay along one path is the characteristic layering pattern, not a deviation."""
    decay = last(EX1_DECAY)
    assert decay < last(EX2_BRANCHES)
    assert decay < last(EX3_REVERSAL)
    assert decay < last(EX4_CONVERGENCE)


def test_value_reversal_is_the_highest_of_the_four():
    """'Example 3 should receive the highest risk score of the four.'  A reversal
    against a structurally intact path is a direct contradiction."""
    reversal = last(EX3_REVERSAL)
    assert reversal > last(EX1_DECAY)
    assert reversal > last(EX2_BRANCHES)
    assert reversal > last(EX4_CONVERGENCE)


def test_consistent_decay_scores_as_though_amounts_carried_nothing():
    """Decay is the expected pattern, so it must add nothing over the same shape
    with flat amounts - the risk in Example 1 is structural, not value-driven."""
    flat = [(sender, receiver, 100.0) for sender, receiver, _ in EX1_DECAY]
    assert last(EX1_DECAY) == last(flat)


# --- structural segmentation of the value signal ----------------------------------

def test_sibling_branches_do_not_pool_their_amounts():
    """The core principle: 'do not blindly aggregate amounts across unrelated
    branches'.  A second branch out of the same ancestor, at a completely
    different scale, must not change the first branch's score."""
    alone = [(M, A, 10000.0), (A, C, 9800.0), (C, H, 9700.0)]
    with_sibling = [
        (M, A, 10000.0),
        (A, C, 9800.0),
        (A, S, 12.0),
        (C, H, 9700.0),
    ]
    assert last(with_sibling) == last(alone)


def test_each_branch_is_judged_on_its_own_trail():
    """Example 2: two branches from one ancestor, each internally consistent.  The
    branch being scored sees its own amounts, not the other's."""
    scores = feed(EX2_BRANCHES)
    assert all(0.0 <= value <= 1.0 for value in scores)
    # The final leg extends the Sterling Bridge branch (10000 -> 5000 -> 4900).
    assert scores[-1] > 0.0


def test_convergence_keeps_the_arriving_trails_distinct():
    """Example 4: two branches arrive at one destination.  Structure changes, but
    the value trajectory the final leg is judged against is still only its own."""
    converging = last(EX4_CONVERGENCE)
    diverging = last(EX2_BRANCHES)
    # Same amounts on the scored leg's own trail; only the destination differs, so
    # any gap between them is structural, which is what the brief asks for.
    assert converging > diverging


# --- the value signal itself -------------------------------------------------------

def test_a_single_amount_alone_carries_no_value_signal():
    """'A single amount means little alone.'  With no predecessor edge there is no
    trail to confirm or contradict."""
    assert last([(M, A, 10000.0)]) == 0.0
    assert last([(M, A, 10000.0)]) == last([(M, A, 1.0)])


def test_growth_across_a_hop_outranks_the_same_path_that_decays():
    """The reversal in isolation: identical structure, identical identity, the only
    difference is that the last hop grows instead of shrinking."""
    decays = [(M, A, 10000.0), (A, C, 9900.0), (C, H, 9800.0), (H, N, 9700.0)]
    grows = [(M, A, 10000.0), (A, C, 9900.0), (C, H, 9800.0), (H, N, 9900.0)]
    assert last(grows) > last(decays)


def test_a_larger_reversal_outranks_a_smaller_one():
    base = [(M, A, 10000.0), (A, C, 9900.0), (C, H, 9800.0)]
    small = base + [(H, N, 9810.0)]
    large = base + [(H, N, 11000.0)]
    assert last(large) > last(small) > last(base + [(H, N, 9700.0)])


def test_an_incoherent_trail_outranks_a_coherent_one():
    """'The trail of amounts can confirm or contradict a pattern.'  Same structure,
    same endpoints: one path retains a steady fraction, the other lurches."""
    steady = [(M, A, 10000.0), (A, C, 9900.0), (C, H, 9800.0)]
    lurching = [(M, A, 10000.0), (A, C, 2000.0), (C, H, 1980.0)]
    assert last(lurching) > last(steady)


def test_value_signal_does_not_manufacture_risk_without_structure():
    """A wild amount on an isolated transaction has no inferred flow to contradict."""
    assert last([(M, A, 10000.0), (C, H, 999999.0)]) == 0.0


def test_value_evidence_expires_with_the_lookback_window():
    """An amount that has aged out cannot still be the trail's previous step."""
    model = GhostChainsModel()
    model.process(make_transaction(payload_for(0, (M, A, 10000.0), timedelta())))
    stale = model.process(
        make_transaction(
            {
                "txId": "late",
                "fromUserId": A,
                "toUserId": C,
                "amount": 99999.0,
                "createdAt": iso(BASE + LOOKBACK + timedelta(minutes=1)),
            }
        )
    )
    assert stale == 0.0


# --- cross-signal scenarios --------------------------------------------------------

def test_structure_and_value_together_outrank_either_alone():
    """Phase 1 + Phase 3 briefing example: a return path whose amount also grows."""
    return_only = [(M, A, 10000.0), (A, C, 9800.0), (C, H, 9700.0), (H, A, 9600.0)]
    return_and_growth = [
        (M, A, 10000.0),
        (A, C, 9800.0),
        (C, H, 9700.0),
        (H, A, 9850.0),
    ]
    assert last(return_and_growth) > last(return_only)


def test_identity_and_value_together_outrank_either_alone():
    """Phase 2 + Phase 3 briefing example: convergence joining two address-linked
    chains, with both the address and the amount trajectory breaking at the join."""
    legs = [
        (M, A, 10000.0, ip(IP_A)),
        (C, H, 10000.0, ip(IP_A)),
        (A, N, 9800.0, ip(IP_A)),
        (H, N, 10100.0, ip(IP_B)),
    ]
    value_only = [
        (M, A, 10000.0),
        (C, H, 10000.0),
        (A, N, 9800.0),
        (H, N, 10100.0),
    ]
    identity_only = [
        (M, A, 10000.0, ip(IP_A)),
        (C, H, 10000.0, ip(IP_A)),
        (A, N, 9800.0, ip(IP_A)),
        (H, N, 10000.0, ip(IP_B)),
    ]
    combined = last(legs)
    assert combined > last(value_only)
    assert combined > last(identity_only)


def test_identity_vanishing_mid_flow_still_signals_under_phase_three():
    """Phase 3 keeps Phase 2's rule that a dropped identifier on a connected path is
    an observable state, not merely a missing value."""
    dropped = [(M, A, 10000.0), (A, C, 9900.0), (C, H, 9800.0)]
    carried = [
        (M, A, 10000.0, dev(DEV_A)),
        (A, C, 9900.0, dev(DEV_A)),
        (C, H, 9800.0),
    ]
    assert last(carried) > last(dropped)


# --- robustness --------------------------------------------------------------------

def test_missing_or_malformed_amounts_do_not_fail_processing():
    model = GhostChainsModel()
    for index, amount in enumerate([None, "lots", True, {"v": 1}]):
        parsed = make_transaction(
            {
                "txId": f"odd{index}",
                "fromUserId": M,
                "toUserId": A,
                "amount": amount,
                "createdAt": iso(BASE + index * timedelta(minutes=1)),
            }
        )
        assert 0.0 <= model.process(parsed) <= 1.0


def test_zero_and_negative_amounts_do_not_break_the_trail():
    assert 0.0 <= last([(M, A, 0.0), (A, C, 100.0), (C, H, 50.0)]) <= 1.0
    assert 0.0 <= last([(M, A, -100.0), (A, C, 100.0), (C, H, 50.0)]) <= 1.0


def test_a_cycle_terminates_the_trail_walk():
    """The backward walk must not loop forever on a closed path."""
    assert 0.0 <= last([(M, A, 100.0), (A, C, 90.0), (C, M, 80.0), (M, A, 70.0)]) <= 1.0


def test_duplicate_txid_returns_the_original_score_without_mutating_state():
    model = GhostChainsModel()
    feed([(M, A, 10000.0), (A, C, 9900.0)], model)
    tx = make_transaction(
        {
            "txId": "repeat",
            "fromUserId": C,
            "toUserId": H,
            "amount": 9800.0,
            "createdAt": iso(BASE + timedelta(minutes=2)),
        }
    )
    first = model.process(tx)
    edges = dict(model._edges)
    assert model.process(tx) == first
    assert dict(model._edges) == edges


def test_reset_clears_value_state():
    model = GhostChainsModel()
    feed(EX3_REVERSAL, model)
    model.reset()
    assert model._edge_amount == {}
    assert model.process(make_transaction(payload_for(0, (M, A, 10000.0), timedelta()))) == 0.0


def test_identical_input_after_reset_is_reproducible():
    model = GhostChainsModel()
    run_one = feed(EX3_REVERSAL, model)
    model.reset()
    assert feed(EX3_REVERSAL, model) == run_one


def test_edge_amount_state_is_bounded_by_the_window():
    """Value state must not outlive the edges it describes."""
    model = GhostChainsModel()
    feed([(M, A, 10000.0), (A, C, 9900.0)], model)
    model.process(
        make_transaction(
            {
                "txId": "far",
                "fromUserId": O,
                "toUserId": S,
                "amount": 5.0,
                "createdAt": iso(BASE + LOOKBACK + timedelta(minutes=1)),
            }
        )
    )
    assert set(model._edge_amount) == {(O, S)}


# --- endpoints ---------------------------------------------------------------------

@pytest.fixture()
def client():
    app = create_app()
    app.config.update(TESTING=True)
    with app.test_client() as test_client:
        test_client.post("/ghost-chains/reset", json={"clearTransactions": True})
        yield test_client


def test_amounts_flow_through_the_endpoint(client):
    body = {
        "transactions": [
            payload_for(index, leg, timedelta(minutes=1))
            for index, leg in enumerate(EX3_REVERSAL)
        ]
    }
    response = client.post("/ghost-chains/transactions", json=body).get_json()
    assert [item["txId"] for item in response["transactions"]] == [
        "tx0",
        "tx1",
        "tx2",
        "tx3",
    ]
    scores = [item["riskScore"] for item in response["transactions"]]
    assert all(0.0 <= value <= 1.0 for value in scores)
    assert scores[-1] == last(EX3_REVERSAL)


def test_endpoint_reports_the_briefing_ordering(client):
    def score(legs):
        client.post("/ghost-chains/reset", json={"clearTransactions": True})
        body = {
            "transactions": [
                payload_for(index, leg, timedelta(minutes=1))
                for index, leg in enumerate(legs)
            ]
        }
        response = client.post("/ghost-chains/transactions", json=body).get_json()
        return response["transactions"][-1]["riskScore"]

    decay, branches, reversal, convergence = (score(legs) for legs in BRIEFING)
    assert decay == min(decay, branches, reversal, convergence)
    assert reversal == max(decay, branches, reversal, convergence)
