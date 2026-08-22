from flask import jsonify, request

from challenges.adaptive_gateway import blueprint
from challenges.adaptive_gateway.solution import calculate_slo, decode_payload, transform


@blueprint.route("/solve", methods=["POST"])
def handle_solve():
    body = request.get_json(silent=True, force=True)
    if not isinstance(body, dict):
        return jsonify(error="request body must be a JSON object"), 400

    try:
        decoded = decode_payload(body.get("payload"))
    except ValueError as exc:
        return jsonify(error=str(exc)), 400

    response = {}
    if isinstance(decoded.get("adaptInput"), dict):
        response["adaptOutput"] = transform(decoded["adaptInput"])
    if "heartbeats" in decoded or "sloQuery" in decoded:
        response["sloOutput"] = calculate_slo(
            decoded.get("heartbeats"), decoded.get("sloQuery")
        )
    return jsonify(response)
