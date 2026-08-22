from app import create_app



def transaction(tx_id, source, target, minute):
    return {
        "txId": tx_id,
        "fromUserId": source,
        "toUserId": target,
        "amount": 100,
        "createdAt": f"2026-06-08T12:{minute:02d}:00Z",
    }


def score_sequence(client, values):
    response = client.post("/ghost-chains/transactions", json={"transactions": values})
    assert response.status_code == 200
    return [item["riskScore"] for item in response.get_json()["transactions"]]


def test_phase_1_examples_have_structural_ordering():
    app = create_app()
    app.config.update(TESTING=True)
    client = app.test_client()

    client.post("/ghost-chains/reset", json={"clearTransactions": True})
    example_1 = score_sequence(client, [transaction("one", "M", "A", 0)])[-1]

    client.post("/ghost-chains/reset", json={"clearTransactions": True})
    example_2 = score_sequence(
        client,
        [transaction("one", "M", "A", 0), transaction("two", "A", "C", 1)],
    )[-1]

    client.post("/ghost-chains/reset", json={"clearTransactions": True})
    example_3 = score_sequence(
        client,
        [
            transaction("one", "M", "A", 0),
            transaction("two", "M", "H", 1),
            transaction("three", "A", "S", 2),
            transaction("four", "H", "S", 3),
        ],
    )[-1]

    client.post("/ghost-chains/reset", json={"clearTransactions": True})
    example_4 = score_sequence(
        client,
        [
            transaction("one", "M", "A", 0),
            transaction("two", "A", "C", 1),
            transaction("three", "C", "O", 2),
            transaction("four", "O", "A", 3),
        ],
    )[-1]

    client.post("/ghost-chains/reset", json={"clearTransactions": True})
    example_5 = score_sequence(
        client,
        [
            transaction("one", "M", "A", 0),
            transaction("two", "A", "C", 1),
            transaction("three", "C", "M", 2),
            transaction("four", "A", "N", 3),
            transaction("five", "N", "M", 4),
        ],
    )[-1]

    assert example_1 < example_2 < example_3
    assert example_4 - example_2 > 0.05
    assert example_5 - example_4 > 0.05
