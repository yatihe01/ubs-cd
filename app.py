from contextlib import asynccontextmanager

from a2wsgi import WSGIMiddleware
from flask import Flask, jsonify
from starlette.applications import Starlette
from starlette.routing import Mount

from challenges import BLUEPRINTS
from challenges.adaptive_gateway.routes import handle_solve as active_challenge
from challenges.showdown.phase_1 import handle_move as showdown_move
from challenges.tool_box import mcp_app as tool_box_mcp_app


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

    # SHOWDOWN calls {registered_base_url}/move. Answer at the root as well as
    # under /showdown so the bot plays whichever base URL was registered: the
    # root health check passes either way, so a mismatch here does not look
    # like a failure, it looks like a bot that folds every hand.
    app.add_url_rule(
        "/move",
        endpoint="showdown_move",
        view_func=showdown_move,
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


flask_app = create_app()


@asynccontextmanager
async def lifespan(app: Starlette):
    async with tool_box_mcp_app.lifespan(app):
        yield


# Tool Box has a permanent, challenge-specific base URL. Give the evaluator
# https://<host>/tool-box so it discovers https://<host>/tool-box/mcp.
app = Starlette(
    lifespan=lifespan,
    routes=[
        Mount("/tool-box/mcp", app=tool_box_mcp_app),
        Mount("/", app=WSGIMiddleware(flask_app)),
    ],
)


if __name__ == "__main__":
    import os

    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))
