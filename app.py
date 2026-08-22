from flask import Flask, jsonify

from challenges import BLUEPRINTS
from challenges.adaptive_gateway.routes import handle_solve as active_challenge


def create_app() -> Flask:
    app = Flask(__name__)

    for blueprint, url_prefix in BLUEPRINTS:
        app.register_blueprint(blueprint, url_prefix=url_prefix)

    # The challenge evaluator currently expects POST /solve. Keep the
    # namespaced endpoint as the permanent URL and point this alias at the
    # challenge being submitted.
    app.add_url_rule(
        "/solve",
        endpoint="active_challenge",
        view_func=active_challenge,
        methods=["POST"],
    )

    @app.get("/")
    def index():
        return jsonify(
            service="ubs-cd",
            status="ok",
            active_challenge="adaptive-gateway",
        )

    @app.get("/health")
    def health():
        return jsonify(status="healthy")

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
