import base64
import json

from fastapi import APIRouter

router = APIRouter()

PRIORITY_LEVELS = {
    "LOW": 1,
    "MEDIUM": 2,
    "HIGH": 3,
}


def decode_payload(encoded_payload: str) -> dict:
    """Base64 string -> parsed JSON dict. Tolerates missing '=' padding."""
    padding_needed = (-len(encoded_payload)) % 4
    padded_payload = encoded_payload + ("=" * padding_needed)

    raw_bytes = base64.b64decode(padded_payload)
    decoded_text = raw_bytes.decode("utf-8")
    return json.loads(decoded_text)


def transform(adapt_input: dict) -> dict:
    """V1 nested shape -> V2 flat shape. Returns the inner object only."""
    user = adapt_input.get("user") or {}
    metadata = adapt_input.get("metadata") or {}

    user_id = user.get("id")
    full_name = user.get("fullName")

    raw_action = adapt_input.get("action")
    action = raw_action.lower() if isinstance(raw_action, str) else None

    raw_priority = metadata.get("priority")
    priority = None
    if isinstance(raw_priority, str):
        priority = PRIORITY_LEVELS.get(raw_priority.upper())

    return {
        "id": user_id,
        "name": full_name,
        "action": action,
        "priority": priority,
    }


@router.post("/solve")
def solve(body: dict):
    encoded_payload = body.get("payload", "")
    decoded = decode_payload(encoded_payload)
    adapt_input = decoded.get("adaptInput") or {}

    adapt_output = transform(adapt_input)
    return {"adaptOutput": adapt_output}