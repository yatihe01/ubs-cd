from flask import jsonify, request

from challenges.adaptive_gateway import blueprint
from challenges.adaptive_gateway.solution import decode_payload, transform


@blueprint.post("/solve")
def handle_solve():
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify(error="request body must be a JSON object"), 400

    try:
        decoded = decode_payload(body.get("payload"))
    except ValueError as exc:
        return jsonify(error=str(exc)), 400

    return jsonify(adaptOutput=transform(decoded["adaptInput"]))
