from datetime import datetime, timedelta, timezone

import pytest

from app import create_app
from challenges.ghost_chains.solution import (
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


def score_last(edges, step=timedelta(minutes=1)):
    """Feed a sequence of edges one minute apart and return the final score."""
    model = GhostChainsModel()
    result = 0.0
    for index, (sender, receiver) in enumerate(edges):
        result = model.process(tx(f"tx{index}", sender, receiver, BASE + index * step))
    return result


# --- the five briefing examples ---------------------------------------------------

EXAMPLES = {
    "isolated": [(M, A)],
    "extension": [(M, A), (A, C)],
    "convergence": [(M, A), (M, H), (A, S), (H, S)],
    "return": [(M, A), (A, C), (C, O), (O, A)],
    "multi_loop": [(M, A), (A, C), (C, M), (A, N), (N, M)],
}


@pytest.fixture(scope="module")
def example_scores():
    return {name: score_last(edges) for name, edges in EXAMPLES.items()}


def test_scores_stay_in_range(example_scores):
    assert all(0.0 <= value <= 1.0 for value in example_scores.values())


def test_isolated_is_strictly_lowest(example_scores):
    isolated = example_scores["isolated"]
    others = [v for k, v in example_scores.items() if k != "isolated"]
    assert all(isolated < other for other in others)


def test_structural_signal_increases_monotonically(example_scores):
    ordering = ["isolated", "extension", "convergence", "return", "multi_loop"]
    values = [example_scores[name] for name in ordering]
    assert values == sorted(values)
    assert len(set(values)) == len(values), "no two examples may tie"


def test_return_is_meaningfully_above_extension(example_scores):
    assert example_scores["return"] > example_scores["extension"] + 0.15


def test_multi_loop_is_meaningfully_above_return(example_scores):
    assert example_scores["multi_loop"] > example_scores["return"] + 0.05


# --- the signal the brief names but examples do not show --------------------------

def test_shortcut_scores_above_unrelated_new_edge():
    """`M -> S` collapses a four-hop route to one hop; `M -> N` adds a leaf."""
    chain = [(M, A), (A, C), (C, H), (H, S)]
    assert score_last(chain + [(M, S)]) > score_last(chain + [(M, N)])


# --- lookback window --------------------------------------------------------------

def test_edge_exactly_at_window_boundary_is_still_active():
    model = GhostChainsModel()
    model.process(tx("old", M, A, BASE))
    closing = model.process(tx("new", A, M, BASE + LOOKBACK))
    assert closing == score_last([(M, A), (A, M)])


def test_edge_past_the_boundary_is_expired():
    model = GhostChainsModel()
    model.process(tx("old", M, A, BASE))
    closing = model.process(tx("new", A, M, BASE + LOOKBACK + timedelta(seconds=1)))
    assert closing == score_last([(A, M)])


def test_out_of_order_arrival_still_expires():
    """A stale transaction arriving late must not linger past the window."""
    model = GhostChainsModel()
    model.process(tx("t1", M, A, BASE))
    model.process(tx("t2", C, O, BASE + timedelta(hours=30)))
    model.process(tx("t3", H, S, BASE + timedelta(hours=1)))  # late, out of order
    model.process(tx("t4", N, M, BASE + timedelta(hours=60)))
    assert dict(model._edges) == {(N, M): 1}


def test_expired_edges_do_not_influence_scoring():
    fresh = score_last([(M, A)])
    model = GhostChainsModel()
    model.process(tx("stale1", C, O, BASE))
    model.process(tx("stale2", O, H, BASE))
    assert model.process(tx("t", M, A, BASE + LOOKBACK + timedelta(minutes=1))) == fresh


# --- streaming / idempotency / reset ----------------------------------------------

def test_duplicate_txid_returns_original_score_without_mutating_state():
    model = GhostChainsModel()
    model.process(tx("t0", M, A, BASE))
    first = model.process(tx("t1", A, C, BASE + timedelta(minutes=1)))
    edges = dict(model._edges)
    assert model.process(tx("t1", A, C, BASE + timedelta(minutes=1))) == first
    assert dict(model._edges) == edges


def test_conflicting_payload_does_not_break_the_batch():
    model = GhostChainsModel()
    first = model.process(tx("t0", M, A, BASE))
    assert model.process(tx("t0", C, O, BASE)) == first
    assert dict(model._edges) == {(M, A): 1}


def test_reset_restores_startup_state():
    model = GhostChainsModel()
    for index, (sender, receiver) in enumerate(EXAMPLES["multi_loop"]):
        model.process(tx(f"t{index}", sender, receiver, BASE + index * timedelta(minutes=1)))
    model.reset()
    assert model.process(tx("fresh", M, A, BASE)) == score_last([(M, A)])
    assert model.latest_time == BASE


def test_identical_input_after_reset_is_reproducible():
    model = GhostChainsModel()
    run_one = [model.process(tx(f"t{i}", s, r, BASE + i * timedelta(minutes=1)))
               for i, (s, r) in enumerate(EXAMPLES["multi_loop"])]
    model.reset()
    run_two = [model.process(tx(f"t{i}", s, r, BASE + i * timedelta(minutes=1)))
               for i, (s, r) in enumerate(EXAMPLES["multi_loop"])]
    assert run_one == run_two


# --- degenerate edges -------------------------------------------------------------

def test_repeated_edge_is_treated_as_ordinary_flow():
    """A parallel edge leaves the simple-graph structure unchanged."""
    repeated = score_last([(M, A), (M, A)])
    assert repeated < score_last([(M, A), (A, C)])
    assert repeated > 0.0


def test_self_loop_registers_as_recurring_flow():
    assert score_last([(A, A)]) > score_last([(M, A)])


# --- optional and unknown fields --------------------------------------------------

def test_optional_identity_fields_are_captured_as_present_or_absent():
    present = tx("t1", M, A, BASE, ipAddress="203.0.113.7", deviceId="dev-1")
    absent = tx("t2", M, A, BASE)
    assert (present.ip_address, present.device_id) == ("203.0.113.7", "dev-1")
    assert (absent.ip_address, absent.device_id) == (None, None)


def test_unknown_fields_and_missing_amount_are_ignored_gracefully():
    parsed = make_transaction(
        {"txId": "t", "fromUserId": M, "toUserId": A, "createdAt": iso(BASE),
         "somethingNew": {"nested": 1}}
    )
    assert parsed.amount == 0.0


def test_optional_fields_do_not_change_phase_one_scoring():
    with_identity = score_last([(M, A), (A, C)])
    model = GhostChainsModel()
    model.process(tx("t0", M, A, BASE, ipAddress="203.0.113.7"))
    plain = model.process(tx("t1", A, C, BASE + timedelta(minutes=1), deviceId="dev-1"))
    assert plain == with_identity


# --- endpoints --------------------------------------------------------------------

@pytest.fixture()
def client():
    app = create_app()
    app.config.update(TESTING=True)
    with app.test_client() as test_client:
        test_client.post("/ghost-chains/reset", json={"clearTransactions": True})
        yield test_client


def test_health(client):
    assert client.get("/ghost-chains/health").get_json() == {"status": "ok"}


def test_reset_endpoint(client):
    response = client.post("/ghost-chains/reset", json={"clearTransactions": True})
    assert response.status_code == 200
    assert response.get_json() == {"clearTransactions": True}


def test_batch_preserves_order_and_shape(client):
    payload = {
        "transactions": [
            {"txId": "tx_meridian_001", "fromUserId": M, "toUserId": A,
             "amount": 370.0, "createdAt": iso(BASE)},
            {"txId": "tx_cascade_014", "fromUserId": A, "toUserId": C,
             "amount": 100.0, "createdAt": iso(BASE + timedelta(minutes=1))},
        ]
    }
    body = client.post("/ghost-chains/transactions", json=payload).get_json()
    assert [item["txId"] for item in body["transactions"]] == [
        "tx_meridian_001",
        "tx_cascade_014",
    ]
    assert all(0.0 <= item["riskScore"] <= 1.0 for item in body["transactions"])


def test_batch_is_processed_sequentially(client):
    """Later transactions in one request see earlier ones from the same request."""
    payload = {
        "transactions": [
            {"txId": f"seq{i}", "fromUserId": s, "toUserId": r, "amount": 1.0,
             "createdAt": iso(BASE + i * timedelta(minutes=1))}
            for i, (s, r) in enumerate(EXAMPLES["multi_loop"])
        ]
    }
    body = client.post("/ghost-chains/transactions", json=payload).get_json()
    scores = [item["riskScore"] for item in body["transactions"]]
    assert scores[-1] == score_last(EXAMPLES["multi_loop"])
