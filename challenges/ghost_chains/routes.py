from flask import jsonify, request

from challenges.ghost_chains import blueprint
from challenges.ghost_chains.solution import GhostChainsModel, make_transaction


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
        for transaction in parsed:
            previous = model.scores.get(transaction.tx_id)
            if previous is not None and previous[0] != transaction:
                return jsonify(error=f"txId {transaction.tx_id!r} has a different payload"), 409
        results = [
            {"txId": transaction.tx_id, "riskScore": model.process(transaction)}
            for transaction in parsed
        ]
    except ValueError as exc:
        return jsonify(error=str(exc)), 400

    return jsonify(transactions=results)
