import base64
import json

import pytest

from app import create_app


@pytest.fixture()
def client():
    app = create_app()
    app.config.update(TESTING=True)
    return app.test_client()


def encode_payload(value):
    return base64.b64encode(json.dumps(value).encode("utf-8")).decode("ascii")


def test_adaptive_gateway_returns_slo_output(client):
    payload = encode_payload(
        {
            "adaptInput": {
                "user": {"id": "U42", "fullName": "Jane Doe"},
                "action": "CREATE",
                "metadata": {"priority": "HIGH"},
            },
            "heartbeats": [
                {"service": "auth", "timestamp": 1710000123, "latencyMs": 120, "status": "OK"},
                {"service": "auth", "timestamp": 1710000125, "latencyMs": 180, "status": "FAIL"},
                {"service": "auth", "timestamp": 1710000121, "latencyMs": 95, "status": "OK"},
            ],
            "sloQuery": {"service": "auth", "since": 1710000123},
        }
    )

    response = client.post("/solve", json={"payload": payload})

    assert response.status_code == 200
    assert response.get_json()["sloOutput"] == {
        "availability": 0.5,
        "p95LatencyMs": 180,
    }


def test_adaptive_gateway_slo_ignores_other_services_and_old_heartbeats(client):
    payload = encode_payload(
        {
            "adaptInput": {},
            "heartbeats": [
                {"service": "auth", "timestamp": 9, "latencyMs": 10, "status": "OK"},
                {"service": "auth", "timestamp": 10, "latencyMs": 20, "status": "OK"},
                {"service": "payments", "timestamp": 20, "latencyMs": 999, "status": "FAIL"},
            ],
            "sloQuery": {"service": "auth", "since": 10},
        }
    )

    response = client.post("/solve", json={"payload": payload})

    assert response.get_json()["sloOutput"] == {
        "availability": 1.0,
        "p95LatencyMs": 20,
    }


def test_adaptive_gateway_slo_is_empty_without_matching_heartbeats(client):
    payload = encode_payload(
        {
            "adaptInput": {},
            "heartbeats": [],
            "sloQuery": {"service": "auth", "since": 10},
        }
    )

    response = client.post("/solve", json={"payload": payload})

    assert response.get_json()["sloOutput"] == {
        "availability": 0.0,
        "p95LatencyMs": 0,
    }


def test_adaptive_gateway_returns_only_slo_output_for_slo_only_payload(client):
    payload = encode_payload(
        {
            "heartbeats": [
                {"service": "auth", "timestamp": 10, "latencyMs": 100, "status": "OK"},
            ],
            "sloQuery": {"service": "auth", "since": 10},
        }
    )

    response = client.post("/solve", json={"payload": payload})

    assert response.get_json() == {
        "sloOutput": {
            "availability": 1.0,
            "p95LatencyMs": 100,
        }
    }


def test_adaptive_gateway_slo_deduplicates_repeated_heartbeats(client):
    payload = encode_payload(
        {
            "heartbeats": [
                {"service": "auth", "timestamp": 100, "latencyMs": 50, "status": "OK"},
                {"service": "auth", "timestamp": 100, "latencyMs": 50, "status": "OK"},
                {"service": "auth", "timestamp": 200, "latencyMs": 150, "status": "FAIL"},
            ],
            "sloQuery": {"service": "auth", "since": 100},
        }
    )

    response = client.post("/solve", json={"payload": payload})

    assert response.get_json() == {
        "sloOutput": {
            "availability": 0.5,
            "p95LatencyMs": 150,
        }
    }


def test_adaptive_gateway_slo_status_matching_is_case_insensitive(client):
    payload = encode_payload(
        {
            "heartbeats": [
                {"service": "auth", "timestamp": 100, "latencyMs": 50, "status": "ok"},
                {"service": "auth", "timestamp": 101, "latencyMs": 60, "status": "Fail"},
                {"service": "auth", "timestamp": 102, "latencyMs": 70, "status": "OK"},
            ],
            "sloQuery": {"service": "auth", "since": 100},
        }
    )

    response = client.post("/solve", json={"payload": payload})

    assert response.get_json() == {
        "sloOutput": {
            "availability": 2 / 3,
            "p95LatencyMs": 70,
        }
    }


def test_adaptive_gateway_slo_counts_failures_missing_latency(client):
    payload = encode_payload(
        {
            "heartbeats": [
                {"service": "auth", "timestamp": 100, "latencyMs": 50, "status": "OK"},
                {"service": "auth", "timestamp": 101, "status": "FAIL"},
            ],
            "sloQuery": {"service": "auth", "since": 100},
        }
    )

    response = client.post("/solve", json={"payload": payload})

    assert response.get_json() == {
        "sloOutput": {
            "availability": 0.5,
            "p95LatencyMs": 50,
        }
    }


def test_adaptive_gateway_slo_p95_on_larger_sample(client):
    heartbeats = [
        {"service": "orders", "timestamp": i, "latencyMs": i * 10, "status": "OK"}
        for i in range(1, 21)
    ]
    payload = encode_payload(
        {
            "heartbeats": heartbeats,
            "sloQuery": {"service": "orders", "since": 1},
        }
    )

    response = client.post("/solve", json={"payload": payload})

    assert response.get_json() == {
        "sloOutput": {
            "availability": 1.0,
            "p95LatencyMs": 190,
        }
    }
