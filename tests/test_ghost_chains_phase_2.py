"""Ghost Chains Phase 2: identity signal on top of the Phase 1 structural signal.

A Phase 2 evaluation re-tests Phase 1, so the first section pins the Phase 2 model
to the *exact* Phase 1 scores whenever no identity fields are present.  The rest
covers the identity layer itself.
"""

from datetime import datetime, timedelta, timezone

import pytest

from app import create_app
from challenges.ghost_chains.solution import GhostChainsModel as PhaseOneModel
from challenges.ghost_chains.solution2 import (
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


def tx(tx_id, sender, receiver, created_at, **extra):
    payload = {
        "txId": tx_id,
        "fromUserId": sender,
        "toUserId": receiver,
        "amount": 100.0,
        "createdAt": iso(created_at),
    }
    payload.update(extra)
    return make_transaction(payload)


def feed(legs, model=None, step=timedelta(minutes=1)):
    """Feed `(sender, receiver[, extra fields])` legs one minute apart.

    Returns every score, so a test can assert on any leg, not just the last."""
    model = model or GhostChainsModel()
    scores = []
    for index, leg in enumerate(legs):
        sender, receiver = leg[0], leg[1]
        extra = leg[2] if len(leg) > 2 else {}
        scores.append(
            model.process(tx(f"tx{index}", sender, receiver, BASE + index * step, **extra))
        )
    return scores


def last(legs, model=None):
    return feed(legs, model)[-1]


def dev(value):
    return {"deviceId": value}


def ip(value):
    return {"ipAddress": value}


# --- Phase 1 is re-tested in a Phase 2 run ----------------------------------------

PHASE_ONE_EXAMPLES = {
    "isolated": [(M, A)],
    "extension": [(M, A), (A, C)],
    "convergence": [(M, A), (M, H), (A, S), (H, S)],
    "return": [(M, A), (A, C), (C, O), (O, A)],
    "multi_loop": [(M, A), (A, C), (C, M), (A, N), (N, M)],
}


@pytest.mark.parametrize("name", sorted(PHASE_ONE_EXAMPLES))
def test_identity_free_stream_reproduces_phase_one_exactly(name):
    """The measured Phase 1 model is the reference.  Phase 2 may only *add* signal,
    so with no identity fields anywhere the two must agree to the last decimal."""
    legs = PHASE_ONE_EXAMPLES[name]
    reference = PhaseOneModel()
    expected = [
        reference.process(
            make_transaction(
                {
                    "txId": f"tx{i}",
                    "fromUserId": s,
                    "toUserId": r,
                    "amount": 100.0,
                    "createdAt": iso(BASE + i * timedelta(minutes=1)),
                }
            )
        )
        for i, (s, r) in enumerate(legs)
    ]
    assert feed(legs) == expected


def test_phase_one_ordering_survives():
    scores = {name: last(legs) for name, legs in PHASE_ONE_EXAMPLES.items()}
    ordering = ["isolated", "extension", "convergence", "return", "multi_loop"]
    values = [scores[name] for name in ordering]
    assert values == sorted(values)
    assert len(set(values)) == len(values)
    assert all(0.0 <= value <= 1.0 for value in scores.values())


# --- the four Phase 2 briefing examples -------------------------------------------

CHAIN_NO_IDENTITY = [(M, A), (A, C), (C, H)]
EX1_CONSISTENT = [(M, A, dev(DEV_A)), (A, C, dev(DEV_A)), (C, H, dev(DEV_A))]
EX2_BRANCH = [
    (M, A, dev(DEV_A)),
    (A, C, dev(DEV_A)),
    (A, S, dev(DEV_A)),
    (C, O, dev(DEV_B)),
]
EX3_SHIFT = [
    (M, A, dev(DEV_A)),
    (A, C, dev(DEV_A)),
    (C, H, dev(DEV_B)),
    (H, N, dev(DEV_B)),
]
EX4_CROSS = [(M, A, ip(IP_A)), (C, H, ip(IP_A)), (O, S, ip(IP_A))]


@pytest.mark.parametrize(
    "legs", [EX1_CONSISTENT, EX2_BRANCH, EX3_SHIFT, EX4_CROSS], ids=range(1, 5)
)
def test_briefing_examples_stay_in_range(legs):
    assert all(0.0 <= value <= 1.0 for value in feed(legs))


def test_every_identity_observation_outranks_the_same_structure_without_identity():
    """The brief does not order its four examples against each other, but each one
    carries identity evidence a bare chain does not, and evidence may only add."""
    bare = last(CHAIN_NO_IDENTITY)
    assert last(EX1_CONSISTENT) > bare
    assert last(EX2_BRANCH) > bare
    assert last(EX3_SHIFT) > bare
    assert last(EX4_CROSS) > 0.0


def test_consistent_identity_outranks_a_mid_flow_change():
    """Example 1 vs Example 3 at equal structure: one device driving the whole chain
    is stronger evidence of common control than a chain whose identity story breaks."""
    chain = lambda *legs: [(M, A, legs[0]), (A, C, legs[1]), (C, H, legs[2])]
    consistent = last(chain(dev(DEV_A), dev(DEV_A), dev(DEV_A)))
    changed = last(chain(dev(DEV_A), dev(DEV_A), dev(DEV_B)))
    assert consistent > changed > last(CHAIN_NO_IDENTITY)


def test_identity_amplifies_structure_rather_than_replacing_it():
    """Ranking by structure must survive at equal identity evidence: a device-linked
    loop still beats a device-linked chain, exactly as it does without identity."""
    chain = [(M, A), (A, C), (C, H)]
    loop = [(M, A), (A, C), (C, O), (O, A)]
    with_device = lambda legs: [(s, r, dev(DEV_A)) for s, r in legs]
    assert last(with_device(loop)) > last(with_device(chain))
    assert last(with_device(chain)) > last(chain)
    assert last(with_device(loop)) > last(loop)


# --- independent dimensions -------------------------------------------------------

def test_device_and_ip_are_scored_as_independent_dimensions():
    both = [(M, A, {**dev(DEV_A), **ip(IP_A)}),
            (A, C, {**dev(DEV_A), **ip(IP_A)}),
            (C, H, {**dev(DEV_A), **ip(IP_A)})]
    device_only = [(M, A, dev(DEV_A)), (A, C, dev(DEV_A)), (C, H, dev(DEV_A))]
    assert last(both) > last(device_only) > last(CHAIN_NO_IDENTITY)


def test_a_shared_device_counts_for_more_than_a_shared_address():
    """NAT and shared office egress put unrelated entities behind one address."""
    device = [(M, A, dev(DEV_A)), (A, C, dev(DEV_A)), (C, H, dev(DEV_A))]
    address = [(M, A, ip(IP_A)), (A, C, ip(IP_A)), (C, H, ip(IP_A))]
    assert last(device) > last(address) > last(CHAIN_NO_IDENTITY)


def test_one_dimension_agreeing_while_the_other_diverges_is_scored_on_both():
    agree_both = [(M, A, {**dev(DEV_A), **ip(IP_A)}),
                  (A, C, {**dev(DEV_A), **ip(IP_A)}),
                  (C, H, {**dev(DEV_A), **ip(IP_A)})]
    device_agrees_ip_moves = [(M, A, {**dev(DEV_A), **ip(IP_A)}),
                              (A, C, {**dev(DEV_A), **ip(IP_A)}),
                              (C, H, {**dev(DEV_A), **ip(IP_B)})]
    assert last(agree_both) > last(device_agrees_ip_moves)


# --- missing identity on a connected path -----------------------------------------

def test_dropping_a_consistent_identifier_mid_flow_is_a_signal():
    """Phase 2: 'the suspicious case is a consistent flow that stops carrying its
    identity'.  The dropped leg must outrank the same chain that never had one."""
    dropped = [(M, A, dev(DEV_A)), (A, C, dev(DEV_A)), (C, H)]
    assert last(dropped) > last(CHAIN_NO_IDENTITY)


def test_dropping_an_identifier_ranks_below_carrying_it_consistently():
    """Absence is inferred evidence; an observed shared device is direct evidence."""
    dropped = [(M, A, dev(DEV_A)), (A, C, dev(DEV_A)), (C, H)]
    assert last(EX1_CONSISTENT) > last(dropped)


def test_absent_identity_on_an_unrelated_transaction_is_not_suspicious():
    """'Missing fields are normal on unrelated transactions.'"""
    assert last([(M, A, dev(DEV_A)), (C, H)]) == 0.0
    assert last([(M, A, dev(DEV_A)), (C, H)]) == last([(M, A), (C, H)])


def test_absence_is_weighed_against_how_consistent_the_flow_was():
    """A flow that was already a mixture of identifiers has less of a trail to
    break than one that carried a single identifier throughout."""
    consistent_then_dropped = [(M, A, dev(DEV_A)), (A, C, dev(DEV_A)), (C, H)]
    mixed_then_dropped = [(M, A, dev(DEV_A)), (A, C, dev(DEV_B)), (C, H)]
    assert last(consistent_then_dropped) > last(mixed_then_dropped)


# --- shared identity across disconnected components -------------------------------

def test_shared_address_across_disconnected_components_lifts_off_the_floor():
    """Example 4: three unconnected pairs sharing one address.  Structure alone
    scores every one of them 0.0; the shared address is a coordination hint."""
    assert last([(M, A), (C, H), (O, S)]) == 0.0
    assert last(EX4_CROSS) > 0.0


def test_cross_component_reuse_grows_with_the_number_of_components():
    one = last([(M, A, ip(IP_A)), (C, H, ip(IP_A))])
    two = last([(M, A, ip(IP_A)), (C, H, ip(IP_A)), (O, S, ip(IP_A))])
    assert 0.0 < one < two


def test_cross_component_reuse_stays_below_real_structural_signal():
    """'Not automatic proof of risk on its own': a shared address across unrelated
    pairs must not outrank an actual return path."""
    crowd = [(f"src{i}", f"dst{i}", ip(IP_A)) for i in range(8)]
    assert last(crowd) < last(PHASE_ONE_EXAMPLES["return"])


def test_identity_inside_one_component_is_not_counted_as_cross_component():
    """A device seen only along the flow being scored is agreement, not reuse
    across components, so it cannot also collect the standalone term."""
    connected = [(M, A, dev(DEV_A)), (A, C, dev(DEV_A))]
    model = GhostChainsModel()
    feed(connected, model)
    assert model._local_component(C, H) == {M, A, C, H}


def test_shared_identity_that_later_joins_two_components_is_scored_higher():
    """Two device-linked components that then transact with each other: identity
    and structure now agree, which is the brief's 'stronger combined signal'."""
    linked = [(M, A, dev(DEV_A)), (C, H, dev(DEV_A)), (A, C, dev(DEV_A))]
    unlinked = [(M, A, dev(DEV_A)), (C, H, dev(DEV_B)), (A, C, dev("dev_third"))]
    assert last(linked) > last(unlinked)


# --- ordinary flow stays on the floor ---------------------------------------------

def test_an_isolated_transaction_with_full_identity_is_still_ordinary():
    assert last([(M, A, {**dev(DEV_A), **ip(IP_A)})]) == 0.0


def test_one_sender_paying_two_counterparties_from_one_device_is_ordinary():
    """The most common legitimate shape there is: the same person, the same phone,
    two payments.  Identity may not manufacture risk where no flow exists."""
    assert last([(M, A, dev(DEV_A)), (M, H, dev(DEV_A))]) == 0.0


def test_a_plain_repeat_with_identity_is_still_on_the_floor():
    assert last([(M, A, dev(DEV_A)), (M, A, dev(DEV_A))]) == 0.0


# --- lookback, idempotency, reset -------------------------------------------------

def test_identity_state_expires_with_the_window():
    """An identifier that has aged out must not still be 'the flow's identity'."""
    model = GhostChainsModel()
    model.process(tx("t0", M, A, BASE, deviceId=DEV_A))
    model.process(tx("t1", A, C, BASE + timedelta(minutes=1), deviceId=DEV_A))
    stale = model.process(tx("t2", C, H, BASE + LOOKBACK + timedelta(minutes=1)))
    assert stale == 0.0
    assert model._node_values["dev"] == {}
    assert model._value_txs["dev"] == {}


def test_identity_state_is_released_when_a_parallel_edge_expires():
    """Two transactions on one edge: expiring the first must drop only its own
    identity counts, not the edge's."""
    model = GhostChainsModel()
    model.process(tx("t0", M, A, BASE, deviceId=DEV_A))
    model.process(tx("t1", M, A, BASE + timedelta(hours=12), deviceId=DEV_B))
    model.process(tx("t2", C, H, BASE + LOOKBACK + timedelta(minutes=1)))
    assert set(model._value_txs["dev"]) == {DEV_B}
    assert model._node_values["dev"][M] == {DEV_B: 1}
    assert dict(model._edges) == {(M, A): 1, (C, H): 1}


def test_duplicate_txid_returns_the_original_score_and_mutates_no_identity_state():
    model = GhostChainsModel()
    model.process(tx("t0", M, A, BASE, deviceId=DEV_A))
    first = model.process(tx("t1", A, C, BASE + timedelta(minutes=1), deviceId=DEV_A))
    snapshot = {k: dict(v) for k, v in model._node_values["dev"].items()}
    assert model.process(tx("t1", A, C, BASE + timedelta(minutes=1), deviceId=DEV_A)) == first
    assert {k: dict(v) for k, v in model._node_values["dev"].items()} == snapshot


def test_reset_clears_identity_state():
    model = GhostChainsModel()
    feed(EX1_CONSISTENT, model)
    model.reset()
    assert model._node_values == {"dev": {}, "ip": {}}
    assert model._value_txs == {"dev": {}, "ip": {}}
    assert model.process(tx("fresh", M, A, BASE, deviceId=DEV_A)) == 0.0


def test_identical_input_after_reset_is_reproducible():
    model = GhostChainsModel()
    run_one = feed(EX3_SHIFT, model)
    model.reset()
    assert feed(EX3_SHIFT, model) == run_one


# --- malformed and unknown fields -------------------------------------------------

def test_blank_and_non_string_identity_fields_are_treated_as_absent():
    parsed = make_transaction(
        {"txId": "t", "fromUserId": M, "toUserId": A, "createdAt": iso(BASE),
         "ipAddress": "", "deviceId": 17, "somethingNew": {"nested": 1}}
    )
    assert (parsed.ip_address, parsed.device_id) == (None, None)


def test_null_identity_fields_do_not_fail_processing():
    model = GhostChainsModel()
    parsed = make_transaction(
        {"txId": "t", "fromUserId": M, "toUserId": A, "createdAt": iso(BASE),
         "ipAddress": None, "deviceId": None}
    )
    assert model.process(parsed) == 0.0


# --- endpoints --------------------------------------------------------------------

@pytest.fixture()
def client():
    app = create_app()
    app.config.update(TESTING=True)
    with app.test_client() as test_client:
        test_client.post("/ghost-chains/reset", json={"clearTransactions": True})
        yield test_client


def test_identity_fields_flow_through_the_endpoint(client):
    payload = {
        "transactions": [
            # Uniform amounts: this test is about identity reaching the model, and
            # from Phase 3 on a varying amount is itself a scored signal.
            {"txId": "i0", "fromUserId": M, "toUserId": A, "amount": 100.0,
             "createdAt": iso(BASE), "deviceId": DEV_A},
            {"txId": "i1", "fromUserId": A, "toUserId": C, "amount": 100.0,
             "createdAt": iso(BASE + timedelta(minutes=1)), "deviceId": DEV_A},
            {"txId": "i2", "fromUserId": C, "toUserId": H, "amount": 100.0,
             "createdAt": iso(BASE + timedelta(minutes=2)), "deviceId": DEV_A},
        ]
    }
    body = client.post("/ghost-chains/transactions", json=payload).get_json()
    assert [item["txId"] for item in body["transactions"]] == ["i0", "i1", "i2"]
    scores = [item["riskScore"] for item in body["transactions"]]
    assert all(0.0 <= value <= 1.0 for value in scores)
    assert scores[-1] == last(EX1_CONSISTENT)


def test_partial_identity_across_a_batch_does_not_fail(client):
    payload = {
        "transactions": [
            {"txId": "p0", "fromUserId": M, "toUserId": A, "amount": 1.0,
             "createdAt": iso(BASE), "ipAddress": IP_A, "deviceId": DEV_A},
            {"txId": "p1", "fromUserId": A, "toUserId": C, "amount": 1.0,
             "createdAt": iso(BASE + timedelta(minutes=1))},
            {"txId": "p2", "fromUserId": C, "toUserId": H, "amount": 1.0,
             "createdAt": iso(BASE + timedelta(minutes=2)), "unexpected": True},
        ]
    }
    response = client.post("/ghost-chains/transactions", json=payload)
    assert response.status_code == 200
    assert len(response.get_json()["transactions"]) == 3
