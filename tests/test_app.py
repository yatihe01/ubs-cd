import base64
import json

import pytest

from app import create_app


@pytest.fixture()
def client():
    app = create_app()
    app.config.update(TESTING=True)
    return app.test_client()


def encode_payload(value: dict) -> str:
    serialized = json.dumps(value).encode("utf-8")
    return base64.b64encode(serialized).decode("ascii")


def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.get_json() == {"status": "healthy"}


@pytest.mark.parametrize("endpoint", ["/solve", "/adaptive-gateway/solve"])
def test_adaptive_gateway(client, endpoint):
    payload = encode_payload(
        {
            "adaptInput": {
                "user": {"id": "U42", "fullName": "Jane Doe"},
                "action": "CREATE",
                "metadata": {"priority": "HIGH"},
            }
        }
    )

    response = client.post(endpoint, json={"payload": payload})

    assert response.status_code == 200
    assert response.get_json() == {
        "adaptOutput": {
            "id": "U42",
            "name": "Jane Doe",
            "action": "create",
            "priority": 3,
        }
    }


@pytest.mark.parametrize(
    "body",
    [None, {}, {"payload": "not-base64"}, {"payload": "e30="}],
)
def test_adaptive_gateway_rejects_invalid_payload(client, body):
    response = client.post("/solve", json=body)

    assert response.status_code == 400
    assert "error" in response.get_json()
