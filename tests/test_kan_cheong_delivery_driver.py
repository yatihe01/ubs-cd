from app import create_app


def test_kan_cheong_delivery_driver_batch():
    app = create_app()
    app.config.update(TESTING=True)
    client = app.test_client()

    base_case = {
        "start_coordinate": [0, 0],
        "end_coordinate": [1, 0],
        "start_time": "2026-06-10T08:30:00Z",
        "nodes": [[0, 0], [1, 0]],
        "edges": [
            {
                "edge_id": "edge_0",
                "node1": [0, 0],
                "node2": [1, 0],
                "base_duration_sec": 60,
            }
        ],
        "obstructions": [],
    }
    blocked_case = {
        **base_case,
        "obstructions": [
            {
                "edge_id": "edge_0",
                "edge": {"from": [0, 0], "to": [1, 0]},
                "start_time": "2026-06-10T08:00:00Z",
                "end_time": "2026-06-10T09:00:00Z",
                "speed_factor": 0.0,
            }
        ],
    }

    response = client.post(
        "/kan-cheong-delivery-driver",
        json={"open": base_case, "blocked": blocked_case},
    )

    assert response.status_code == 200
    assert response.get_json() == {
        "open": {
            "total_duration_sec": 60,
            "arrival_time": "2026-06-10T08:31:00Z",
            "path": ["edge_0"],
        },
        "blocked": {
            "total_duration_sec": None,
            "arrival_time": None,
            "path": [],
        },
    }


def test_kan_cheong_delivery_driver_rejects_non_object_batch():
    app = create_app()
    app.config.update(TESTING=True)
    client = app.test_client()

    response = client.post("/kan-cheong-delivery-driver", json=[])

    assert response.status_code == 400
    assert "error" in response.get_json()
