import base64
import binascii
import json
import math


PRIORITY_LEVELS = {
    "LOW": 1,
    "MEDIUM": 2,
    "HIGH": 3,
}


def decode_payload(encoded_payload: str) -> dict:
    if not isinstance(encoded_payload, str) or not encoded_payload:
        raise ValueError("payload must be a non-empty base64 string")

    padded_payload = encoded_payload + ("=" * (-len(encoded_payload) % 4))
    try:
        raw_bytes = base64.b64decode(padded_payload, validate=True)
        decoded = json.loads(raw_bytes.decode("utf-8"))
    except (binascii.Error, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("payload must contain base64-encoded JSON") from exc

    if not isinstance(decoded, dict) or not isinstance(decoded.get("adaptInput"), dict):
        raise ValueError("payload must contain an adaptInput object")

    return decoded


def transform(adapt_input: dict) -> dict:
    user = adapt_input.get("user") or {}
    metadata = adapt_input.get("metadata") or {}
    raw_action = adapt_input.get("action")
    raw_priority = metadata.get("priority")

    return {
        "id": user.get("id"),
        "name": user.get("fullName"),
        "action": raw_action.lower() if isinstance(raw_action, str) else None,
        "priority": (
            PRIORITY_LEVELS.get(raw_priority.upper())
            if isinstance(raw_priority, str)
            else None
        ),
    }


def calculate_slo(heartbeats: object, slo_query: object) -> dict:
    if not isinstance(heartbeats, list) or not isinstance(slo_query, dict):
        return {"availability": 0.0, "p95LatencyMs": 0}

    service = slo_query.get("service")
    since = slo_query.get("since")
    if not isinstance(service, str) or not isinstance(since, (int, float)):
        return {"availability": 0.0, "p95LatencyMs": 0}

    matching = [
        heartbeat
        for heartbeat in heartbeats
        if isinstance(heartbeat, dict)
        and heartbeat.get("service") == service
        and isinstance(heartbeat.get("timestamp"), (int, float))
        and heartbeat["timestamp"] >= since
        and isinstance(heartbeat.get("latencyMs"), (int, float))
    ]
    if not matching:
        return {"availability": 0.0, "p95LatencyMs": 0}

    latencies = sorted(heartbeat["latencyMs"] for heartbeat in matching)
    p95_index = max(0, math.ceil(len(latencies) * 0.95) - 1)
    available = sum(heartbeat.get("status") == "OK" for heartbeat in matching)
    return {
        "availability": round(available / len(matching), 6),
        "p95LatencyMs": latencies[p95_index],
    }
