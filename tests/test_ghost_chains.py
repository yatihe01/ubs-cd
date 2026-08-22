from datetime import datetime, timedelta, timezone

import pytest

from app import create_app
from challenges.ghost_chains.solution import GhostChainsModel, make_transaction


BASE_TIME = datetime(2026, 6, 8, 12, 0, tzinfo=timezone.utc)


def transaction(
    tx_id: str,
    source: str,
    target: str,
    *,
    created_at: datetime,
):
    return make_transaction(
        {
            "txId": tx_id,
            "fromUserId": source,
            "toUserId": target,
            "amount": 100.0,
            "createdAt": created_at.isoformat(),
        }
    )


def scores(edges: list[tuple[str, str]]) -> list[float]:
    model = GhostChainsModel()
    return [
        model.process(
            transaction(
                f"tx-{index}",
                source,
                target,
                created_at=BASE_TIME + timedelta(minutes=index),
            )
        )
        for index, (source, target) in enumerate(edges)
    ]


def test_restored_372_structural_scores():
    assert scores([("M", "A")])[-1] == 0.0
    assert scores([("M", "A"), ("A", "C")])[-1] == 0.0
    assert scores(
        [("M", "A"), ("M", "H"), ("A", "S"), ("H", "S")]
    )[-1] == 0.16
    assert scores(
        [("M", "A"), ("A", "C"), ("C", "O"), ("O", "A")]
    )[-1] == 0.54
    assert scores(
        [("M", "A"), ("A", "C"), ("C", "M"), ("A", "N"), ("N", "M")]
    )[-1] == 0.86


def test_boundary_experiment_keeps_edge_at_exactly_24_hours():
    model = GhostChainsModel()
    model.process(transaction("ab", "A", "B", created_at=BASE_TIME))

    score = model.process(
        transaction("ba", "B", "A", created_at=BASE_TIME + timedelta(hours=24))
    )

    assert score == 0.38
    assert [item.tx_id for item in model.transactions] == ["ab", "ba"]


def test_stale_late_arrival_is_scored_but_does_not_enter_state():
    model = GhostChainsModel()
    model.process(
        transaction(
            "future",
            "A",
            "B",
            created_at=BASE_TIME + timedelta(hours=48),
        )
    )

    score = model.process(
        transaction("stale", "B", "A", created_at=BASE_TIME)
    )

    assert score == 0.38
    assert [item.tx_id for item in model.transactions] == ["future"]


def test_expiring_one_parallel_transaction_keeps_edge_active():
    model = GhostChainsModel()
    model.process(transaction("ab-old", "A", "B", created_at=BASE_TIME))
    model.process(
        transaction(
            "ab-fresh",
            "A",
            "B",
            created_at=BASE_TIME + timedelta(hours=1),
        )
    )

    score = model.process(
        transaction(
            "ba",
            "B",
            "A",
            created_at=BASE_TIME + timedelta(hours=24, minutes=30),
        )
    )

    assert score == 0.38
    assert [item.tx_id for item in model.transactions] == ["ab-fresh", "ba"]


def test_restored_372_shortcut_and_leaf_scores_match():
    chain = [("M", "A"), ("A", "C"), ("C", "H"), ("H", "S")]

    assert scores(chain + [("M", "S")])[-1] == 0.0
    assert scores(chain + [("M", "N")])[-1] == 0.0


def test_restored_372_repeated_edge_uses_current_graph_context():
    assert scores([("A", "B"), ("B", "A"), ("A", "B")])[-1] == 0.54


def test_restored_372_new_self_loop_has_no_prior_graph_signal():
    assert scores([("A", "A")])[-1] == 0.0


def test_duplicate_transaction_is_idempotent():
    model = GhostChainsModel()
    original = transaction("same", "A", "B", created_at=BASE_TIME)

    first_score = model.process(original)
    second_score = model.process(original)

    assert second_score == first_score
    assert len(model.transactions) == 1


def test_duplicate_id_with_different_payload_is_rejected():
    model = GhostChainsModel()
    model.process(transaction("same", "A", "B", created_at=BASE_TIME))

    with pytest.raises(ValueError, match="different payload"):
        model.process(transaction("same", "A", "C", created_at=BASE_TIME))


def test_http_endpoints_preserve_batch_order_and_unknown_fields():
    client = create_app().test_client()
    reset = client.post("/ghost-chains/reset", json={"clearTransactions": True})
    response = client.post(
        "/ghost-chains/transactions",
        json={
            "transactions": [
                {
                    "txId": "http-ab",
                    "fromUserId": "A",
                    "toUserId": "B",
                    "amount": 100,
                    "createdAt": BASE_TIME.isoformat(),
                    "unknownField": "ignored",
                },
                {
                    "txId": "http-bc",
                    "fromUserId": "B",
                    "toUserId": "C",
                    "amount": 100,
                    "createdAt": (BASE_TIME + timedelta(minutes=1)).isoformat(),
                },
            ]
        },
    )

    assert reset.status_code == 200
    assert response.status_code == 200
    assert response.get_json() == {
        "transactions": [
            {"txId": "http-ab", "riskScore": 0.0},
            {"txId": "http-bc", "riskScore": 0.0},
        ]
    }
