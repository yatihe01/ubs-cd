import base64
import binascii
import json
import math


PRIORITY_LEVELS = {
    "LOW": 1,
    "MEDIUM": 2,
    "HIGH": 3,
    "CRITICAL": 4,
}


def decode_payload(encoded_payload: str) -> dict:
    if not isinstance(encoded_payload, str) or not encoded_payload:
        raise ValueError("payload must be a non-empty base64 string")

    cleaned_payload = "".join(encoded_payload.split())
    padded_payload = cleaned_payload + ("=" * (-len(cleaned_payload) % 4))
    try:
        if "-" in padded_payload or "_" in padded_payload:
            raw_bytes = base64.urlsafe_b64decode(padded_payload)
        else:
            raw_bytes = base64.b64decode(padded_payload)
        decoded = json.loads(raw_bytes.decode("utf-8"))
    except (binascii.Error, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("payload must contain base64-encoded JSON") from exc

    if not isinstance(decoded, dict):
        raise ValueError("payload must contain an object")

    has_adapt_input = isinstance(decoded.get("adaptInput"), dict)
    has_slo_data = "heartbeats" in decoded or "sloQuery" in decoded
    if not has_adapt_input and not has_slo_data:
        raise ValueError("payload must contain adaptInput or SLO data")

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

    in_window = [
        heartbeat
        for heartbeat in heartbeats
        if isinstance(heartbeat, dict)
        and heartbeat.get("service") == service
        and isinstance(heartbeat.get("timestamp"), (int, float))
        and heartbeat["timestamp"] >= since
    ]
    if not in_window:
        return {"availability": 0.0, "p95LatencyMs": 0}

    available = sum(str(heartbeat.get("status", "")).upper() == "OK" for heartbeat in in_window)
    availability = available / len(in_window)

    latencies = sorted(
        heartbeat["latencyMs"]
        for heartbeat in in_window
        if isinstance(heartbeat.get("latencyMs"), (int, float))
    )
    if not latencies:
        p95_latency = 0
    else:
        rank = (len(latencies) - 1) * 0.95
        lower_index = math.floor(rank)
        upper_index = math.ceil(rank)
        if lower_index == upper_index:
            p95_latency = latencies[lower_index]
        else:
            fraction = rank - lower_index
            p95_latency = latencies[lower_index] + fraction * (
                latencies[upper_index] - latencies[lower_index]
            )
            p95_latency = round(p95_latency, 6)
        if isinstance(p95_latency, float) and p95_latency.is_integer():
            p95_latency = int(p95_latency)

    return {"availability": availability, "p95LatencyMs": p95_latency}
