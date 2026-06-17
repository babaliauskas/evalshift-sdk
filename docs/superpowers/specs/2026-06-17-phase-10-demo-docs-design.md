# Phase 10 — End-to-end demo + docs (design)

Date: 2026-06-17
Status: approved (brainstorm complete)
Scope: final phase of `evalshift-sdk` per `IMPLEMENTATION_PLAN.md`.

## Goal

Ship a runnable sample agent, a real README quickstart that mirrors the full
capture → promote → run lifecycle, and a pipeline smoke test — so a new user can
go from `pip install evalshift-sdk` to a scored migration verdict by copy-paste.

## Constraints (inherited)

- **D-deps:** SDK runtime stays **stdlib-only**. Any model SDK (Gemini) is an
  optional extra used only by the example, never a runtime dependency.
- **D1-followup (package clash):** `evalshift` (CLI) and `evalshift-sdk` both own
  the top-level `evalshift` import name, so they **cannot co-install in one venv**.
  Cross-repo e2e must use a *separate* venv for the CLI.
- **CI must stay green with no API key and no network.** Therefore the CI smoke
  path must be dep-free and deterministic.

## Decisions (from brainstorm)

- **Example agent = both variants.** A dep-free deterministic **stub** for CI +
  demo-that-always-works, and a real **Gemini** (`google-genai`) agent for a
  convincing live demo. (User has a Gemini key; Anthropic not used.)
- **E2E = SDK smoke (CI) + optional offline cross-repo script.** pytest smoke
  proves captures are schema-valid + promotable against the vendored CLI models.
  `scripts/e2e.sh` proves the *real* loop offline (no API key) in a separate venv.

## Layout

```
examples/support_agent/
  README.md          # how to run both variants + the full lifecycle
  agent.py           # STUB: dep-free, deterministic, exercises full SDK surface
  agent_live.py      # REAL Gemini agent (google-genai), needs GOOGLE_API_KEY
  tickets.py         # shared sample input tickets (canned support queries)
scripts/
  e2e.sh             # offline cross-repo loop; separate venv installs the CLI
tests/
  test_example_smoke.py   # CI smoke (imports examples via sys.path)
README.md            # root quickstart rewritten to the full lifecycle
pyproject.toml       # add [gemini] optional extra = google-genai
```

`examples/` and `scripts/` ship in the source tree but are **excluded from the
wheel** (same as `tests/`).

## Component 1 — stub agent (`examples/support_agent/agent.py`)

The CI/demo workhorse. Public entry `handle_ticket(query: str) -> str` decorated
`@capture.agent(suite="support_agent")`.

- A fake "model" routes on keywords in the ticket to pick a tool.
- Tools `lookup_order` and `search_kb` decorated `@capture.tool`; return canned
  data keyed by input. No network, fully deterministic.
- Reasoning wrapped in `capture.model_call(...)` (streaming form: `add_text` +
  `set_usage`) and/or `record_model_call(...)` for an atomic call.
- An async entry `handle_ticket_async(query)` that fires the two tools
  concurrently via `asyncio.gather` — demonstrates concurrent-tool parentage.
- One sample ticket routes to a tool that raises → exercises the
  `error` event + partial-capture path.
- A `main()` iterating `tickets.SAMPLE_TICKETS` so the file is runnable:
  `EVALSHIFT_CAPTURE=1 python examples/support_agent/agent.py`.

Demonstrated surface: `capture.agent` (sync+async), `capture.tool`,
`capture.model_call` (streaming), `record_model_call`, `redact=` hook, concurrent
tools, error path.

## Component 2 — live Gemini agent (`examples/support_agent/agent_live.py`)

Same shape as the stub, real calls.

- `from google import genai` (the unified `google-genai` SDK), import-guarded so a
  missing dep gives a clear "install .[gemini]" message, never an import crash.
- Key from `GOOGLE_API_KEY` (fallback `GEMINI_API_KEY`); model from
  `EVALSHIFT_DEMO_MODEL` (default `gemini-2.5-flash`).
- Wraps each model turn in `record_model_call` / `capture.model_call`, recording
  real usage from the response. Real (simple) tool functions under `capture.tool`.
- Not run in CI. Documented as the "live" path.

## Component 3 — shared tickets (`examples/support_agent/tickets.py`)

`SAMPLE_TICKETS: list[str]` — a handful of canned support queries covering each
routing branch (order lookup, KB search, the error case). Imported by both agents
and the smoke test.

## Component 4 — CI smoke (`tests/test_example_smoke.py`)

- Adds `examples/` to `sys.path` (or imports by file path) to load `agent.py`.
- `monkeypatch.setenv("EVALSHIFT_CAPTURE", "1")` + `EVALSHIFT_DIR` → tmp dir
  (reuses the autouse config-isolation conftest fixture).
- Runs the stub over `SAMPLE_TICKETS` (sync and async paths).
- Asserts: N `cap_*.json` files written under
  `<tmp>/captures/support_agent/`; each loads via `load_capture`; each inner
  `AgentTrace` validates against the **vendored CLI model**
  (`tests/conformance/cli_models_vendored.py`) → promotable; the error-ticket
  capture contains an `error`/`tool_result.error` event.

## Component 5 — offline cross-repo script (`scripts/e2e.sh`)

Bash, `set -euo pipefail`. No API key (uses CLI `--offline`).

1. Fresh temp workdir; `EVALSHIFT_CAPTURE=1 EVALSHIFT_DIR=$tmp` run the stub agent
   (in the SDK's venv) → writes `.evalshift/captures/support_agent/cap_*.json`.
2. In a **separate** venv, install the CLI: prefer editable `../evalshift-cli`,
   else `uv pip install evalshift`.
3. `evalshift capture list support_agent`; `evalshift capture promote <cap>
   --as golden_case`.
4. `evalshift run --suite-name support_agent --offline --yes` (+ `evaluate`)
   → assert a verdict/run artifact exists.
5. Print PASS/FAIL; clean up.

Guards: if `../evalshift-cli` and PyPI both unavailable, skip with a clear
message (exit 0, documented as "needs the CLI"). Not wired into CI.

## Component 6 — docs

- **Root `README.md`:** replace the "early scaffold" note with a real quickstart:
  install SDK → decorate agent (`@capture.agent`) → `EVALSHIFT_CAPTURE=1` run →
  captures on disk → (CLI, separate venv) `capture promote` → `run --offline` →
  report. Note both stub + Gemini examples and the package-clash caveat.
- **`examples/support_agent/README.md`:** how to run the stub
  (`EVALSHIFT_CAPTURE=1 python …`), the live Gemini variant
  (`pip install -e .[gemini]`, set key), and `scripts/e2e.sh`.

## pyproject

Add `[project.optional-dependencies] gemini = ["google-genai>=1.0"]`. Confirm
`examples`/`scripts` excluded from the build (hatch packages only
`src/evalshift`). No change to runtime deps.

## Verification (Phase 10 exit)

- `EVALSHIFT_CAPTURE=1` stub run → schema-valid promotable `cap_*.json`; capture
  OFF → zero files.
- `uv run ruff check && ruff format --check && uv run mypy --strict && uv run
  pytest` all green (new smoke test included).
- `scripts/e2e.sh` documented + runnable where the CLI is available; offline, no
  key. (Not asserted in CI.)
- `IMPLEMENTATION_PLAN.md` Phase 10 row flipped to ☑ with a verification note.

## Out of scope (YAGNI)

- Full cross-repo loop inside SDK CI (fragile; the offline script covers it).
- Re-capture / replay during `run` (CLI already scores promoted ground-truth).
- TS example; multiple example agents beyond `support_agent`.
```

