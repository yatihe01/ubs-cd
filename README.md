# UBS Coding Challenge

Shared Flask service for the NPC Union team. Each challenge lives in its own
package and is exposed through a Flask Blueprint.

## Endpoints

- `GET /` - service information
- `GET /health` - Render health check
- `POST /solve` - alias for the currently active challenge
- `POST /adaptive-gateway/solve` - permanent Adaptive API Gateway endpoint
- `POST /showdown/move` - SHOWDOWN bot, one move per call
- `GET /showdown/health` - SHOWDOWN warm-up probe
- `/mcp` - active Tool Box challenge MCP endpoint
- `POST /kan-cheong-delivery-driver` - batch Kan Chiong Delivery Driver endpoint

For Tool Box, select the registered `https://<host>` team URL. The evaluator
appends `/mcp` when discovering the server.

## SHOWDOWN

Register `https://<host>/showdown` as the bot URL: the coordinator appends
`/move` and `/health` itself. Phase 1 requests use
`challenges/showdown/phase_1.py`; Phase 2 requests are dispatched to
`challenges/showdown/phase_2.py`, which learns opaque table rules by codename
across the four 40-hand legs and retries.

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
