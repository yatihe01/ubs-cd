from flask import jsonify, request

from challenges.ghost_chains import blueprint
# Each phase re-tests every earlier one, so the live model is the newest phase.
# `solution.py` and `solution2.py` are kept untouched as the measured Phase 1 and
# Phase 2 baselines to fall back to and to diff against.
from challenges.ghost_chains.solution3 import GhostChainsModel, make_transaction


model = GhostChainsModel()


@blueprint.route("/health", methods=["GET"])
def health():
    return jsonify(status="ok")


@blueprint.route("/reset", methods=["POST"])
def reset():
    body = request.get_json(silent=True)
    if not isinstance(body, dict) or body.get("clearTransactions") is not True:
        return jsonify(error="clearTransactions must be true"), 400
    model.reset()
    return jsonify(clearTransactions=True)


@blueprint.route("/transactions", methods=["POST"])
def transactions():
    body = request.get_json(silent=True)
    values = body.get("transactions") if isinstance(body, dict) else None
    if not isinstance(values, list):
        return jsonify(error="transactions must be an array"), 400

    try:
        parsed = [make_transaction(value) for value in values]
        results = [
            {"txId": transaction.tx_id, "riskScore": model.process(transaction)}
            for transaction in parsed
        ]
    except ValueError as exc:
        return jsonify(error=str(exc)), 400

    return jsonify(transactions=results)
