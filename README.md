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
- `GET /showdown/rules` - what the bot has learned about each table-rule codename
- `/mcp` - active Tool Box challenge MCP endpoint
- `POST /kan-cheong-delivery-driver` - batch Kan Chiong Delivery Driver endpoint
- `GET /ghost-chains/health`, `POST /ghost-chains/reset`,
  `POST /ghost-chains/transactions` - Ghost Chains risk scoring

For Tool Box, select the registered `https://<host>` team URL. The evaluator
appends `/mcp` when discovering the server.

## SHOWDOWN

Register `https://<host>/showdown` as the bot URL: the coordinator appends
`/move` and `/health` itself. `phase_1.handle_move` dispatches on the `phase`
field in the request body, so one registered URL serves every phase.

- `phase_1.py` - heads-up, 100 hands, `table_rule` is always `standard`.
- `phase_2.py` - four 40-hand legs. Owns `RuleModel`: the ensemble of rule
  hypotheses, the per-codename evidence, and `outcome_probs`, which reports a
  comparison as `(P(win), P(tie))` rather than one scalar.
- `phase_3.py` - six seats, four 60-hand legs, must *top the table*. Owns the
  multiway equity engine and the `Profile` that tunes the decision layer.
- `phase_4.py` - knockout against other teams, up to seven seats, 200 hands.
  Reuses Phase 3's decision layer through a survival-weighted `Profile`:
  the objective is clearing the cut, not finishing first.

### Rule knowledge

The codename to ruleset mapping is fixed for the whole event and shared by
Phases 2, 3 and 4, so a rule identified once is worth pinning permanently.
After an attempt, read `GET /showdown/rules`; when a codename has resolved to
a single candidate, add it to `phase_2.KNOWN_CODENAMES` and every later
attempt starts from certainty instead of relearning during the leg.

### Equity

`multiway_equity` averages equity *across surviving rule hypotheses*, not
across per-pair win probabilities. The table rule is one unknown truth rather
than fresh noise per opponent, so the two are not interchangeable: the earlier
implementation averaged `win_share` per pair and then bucketed it with
`share == 1.0` / `== 0.5`, which silently scored every uncertain comparison as
a certain loss. While more than one hypothesis was alive that priced a pair -
unbeatable under `standard` - at exactly 0.0 equity, and the bot folded it
getting 5:1. `tests/test_showdown_phase_3.py` pins both that case and the
unrecognised-rule fallback.

## Ghost Chains

Register `https://<host>` with the coordinator; the evaluator appends
`/ghost-chains/...` itself. Each phase is a separate module and the live model is
whichever one `challenges/ghost_chains/routes.py` imports:

- `solution.py` - Phase 1, structural signal only. The measured 380/400 model
  plus `W_SHORTCUT`, which scores the brief's *shortened* paths: before it, an
  N-hop route collapsed to one hop scored exactly 0.0 for every N, the same as an
  unrelated new leaf. `W_SHORTCUT = 0.0` reproduces the 380 model bit-for-bit
  (verified over 300 randomised streams) and is the one-line rollback. Also the
  parity reference for later phases.
- `solution2.py` - Phase 2, the same structural core plus the `ipAddress` /
  `deviceId` identity layer. With no identity fields anywhere in the stream it
  reproduces `solution.py` score for score, which is what keeps the Phase 1 half
  of a Phase 2 evaluation intact - `tests/test_ghost_chains_phase_2.py` asserts
  that against the Phase 1 model directly.
- `solution3.py` - Phase 3, the same structural core again plus the `amount`
  value layer. Currently live. With uniform amounts it reproduces `solution2.py`
  score for score - `tests/test_ghost_chains_phase_3.py` asserts it.

The structural core is duplicated verbatim across the three modules and the parity
tests pin them together, so a change to it - `W_SHORTCUT`, `GAMMA`, `SQUASH`, any
`W_*` - has to be made in all three or the parity tests fail.

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
