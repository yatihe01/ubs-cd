from contextlib import asynccontextmanager

from a2wsgi import WSGIMiddleware
from flask import Flask, jsonify
from starlette.applications import Starlette
from starlette.routing import Mount

from challenges import BLUEPRINTS
from challenges.adaptive_gateway.routes import handle_solve as active_challenge
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


# The evaluation portal stores only the team's root URL and always appends
# /mcp, so the currently active MCP challenge must be mounted at root /mcp.
app = Starlette(
    lifespan=lifespan,
    routes=[
        Mount("/mcp", app=tool_box_mcp_app),
        Mount("/", app=WSGIMiddleware(flask_app)),
    ],
)


if __name__ == "__main__":
    import os

    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))
