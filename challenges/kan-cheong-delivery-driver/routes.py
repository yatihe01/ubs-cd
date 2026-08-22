import json
from decimal import Decimal

from flask import jsonify, request

from . import blueprint
from .solution import solve_case


@blueprint.post("/kan-cheong-delivery-driver")
def handle_solve():
    try:
        batch = json.loads(request.get_data(as_text=True), parse_float=Decimal)
    except json.JSONDecodeError:
        batch = None
    if not isinstance(batch, dict):
        return jsonify(error="request body must be a JSON object"), 400

    results = {}
    try:
        for case_id, case in batch.items():
            if not isinstance(case, dict):
                raise ValueError(f"case {case_id!r} must be a JSON object")
            results[case_id] = solve_case(case)
    except (KeyError, TypeError, ValueError) as exc:
        return jsonify(error=str(exc)), 400

    return jsonify(results)
