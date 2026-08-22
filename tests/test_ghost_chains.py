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


def final_score(edges: list[tuple[str, str]]) -> float:
    model = GhostChainsModel()
    score = 0.0
    for index, (source, target) in enumerate(edges):
        score = model.process(
            transaction(
                f"tx-{index}",
                source,
                target,
                created_at=BASE_TIME + timedelta(minutes=index),
            )
        )
    return score


def test_phase_one_examples_have_increasing_structural_signal():
    isolated = final_score([("M", "A")])
    extension = final_score([("M", "A"), ("A", "C")])
    convergence = final_score(
        [("M", "A"), ("M", "H"), ("A", "S"), ("H", "S")]
    )
    returned = final_score(
        [("M", "A"), ("A", "C"), ("C", "O"), ("O", "A")]
    )
    multi_loop = final_score(
        [("M", "A"), ("A", "C"), ("C", "M"), ("A", "N"), ("N", "M")]
    )

    assert isolated < extension < convergence < returned < multi_loop


def test_shortening_an_existing_path_has_positive_signal():
    model = GhostChainsModel()
    model.process(transaction("ab", "A", "B", created_at=BASE_TIME))
    model.process(
        transaction("bc", "B", "C", created_at=BASE_TIME + timedelta(minutes=1))
    )

    score = model.process(
        transaction("ac", "A", "C", created_at=BASE_TIME + timedelta(minutes=2))
    )

    assert score > 0.0


def test_exact_lookback_boundary_remains_active():
    model = GhostChainsModel()
    model.process(transaction("ab", "A", "B", created_at=BASE_TIME))

    score = model.process(
        transaction("ba", "B", "A", created_at=BASE_TIME + timedelta(hours=24))
    )

    assert score > 0.0
    assert len(model.transactions) == 2


def test_transaction_outside_lookback_does_not_affect_scoring():
    model = GhostChainsModel()
    model.process(transaction("ab", "A", "B", created_at=BASE_TIME))

    score = model.process(
        transaction(
            "ba",
            "B",
            "A",
            created_at=BASE_TIME + timedelta(hours=24, microseconds=1),
        )
    )

    assert score == 0.0
    assert [(item.from_user, item.to_user) for item in model.transactions] == [
        ("B", "A")
    ]


def test_stale_late_arrival_has_no_structural_delta():
    model = GhostChainsModel()
    model.process(
        transaction("future", "A", "B", created_at=BASE_TIME + timedelta(hours=48))
    )

    score = model.process(transaction("stale", "B", "A", created_at=BASE_TIME))

    assert score == 0.0
    assert [item.tx_id for item in model.transactions] == ["future"]


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


def test_http_endpoints_process_in_order_and_ignore_unknown_fields():
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
                    "futurePhaseField": "ignored",
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
            {"txId": "http-bc", "riskScore": 0.07},
        ]
    }
