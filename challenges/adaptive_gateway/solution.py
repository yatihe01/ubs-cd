import base64
import binascii
import json


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
