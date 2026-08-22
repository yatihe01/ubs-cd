# UBS Coding Challenge

Shared Flask service for the NPC Union team. Each challenge lives in its own
package and is exposed through a Flask Blueprint.

## Endpoints

- `GET /` - service information
- `GET /health` - Render health check
- `POST /solve` - alias for the currently active challenge
- `POST /adaptive-gateway/solve` - permanent Adaptive API Gateway endpoint
- `/tool-box/mcp` - Tool Box challenge MCP endpoint

For Tool Box, submit `https://<host>/tool-box` as the team URL. The evaluator
appends `/mcp` when discovering the server.

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
flask --app app run --debug
```

Run the tests with:

```bash
pytest
```

## Adding a challenge

1. Create `challenges/<challenge_name>/routes.py` and `solution.py`.
2. Export a Blueprint from the package.
3. Add the Blueprint and URL prefix to `BLUEPRINTS` in
   `challenges/__init__.py`.
4. Add tests under `tests/`.
5. Work on a `challenge/<challenge-name>` branch and open a pull request into
   `main`.

Render deploys `main` using Uvicorn because the service now hosts both the
Tool Box ASGI MCP application and the existing Flask application.
