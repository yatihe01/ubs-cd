from flask import jsonify, request

from . import blueprint
from .solution import solve_batch


@blueprint.post("/stonks")
def handle_stonks():
    # force=True: the grader is not guaranteed to send application/json, and a
    # missing header must not cost the whole batch its answer.
    batch = request.get_json(silent=True, force=True)
    try:
        results = solve_batch(batch)
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    return jsonify(results)
