# EvalShift SDK — Implementation Plan

Sub-phased, resumable. Tackle one phase at a time. Update the **Status** column as phases land.

## Status tracker
| Phase | Title | Status |
|-------|-------|--------|
| 0 | Repo scaffold + schema freeze | ☑ done |
| 1 | Trace model + serialization | ☐ not started |
| 2 | Capture core (sync) + off-by-default gate | ☐ not started |
| 3 | Sinks + config + env robustness | ☐ not started |
| 4 | Redaction at capture | ☐ not started |
| 5 | Async + streaming + concurrency | ☐ not started |
| 6 | Hygiene + fail-open hardening | ☐ not started |
| 7 | Framework adapters | ☐ not started |
| 8 | Schema versioning + migration | ☐ not started |
| 9 | CLI capture lifecycle (cross-repo) | ☐ not started |
| 10 | End-to-end demo + docs | ☐ not started |

Legend: ☐ not started · ◐ in progress · ☑ done

---

## Context

EvalShift today is **CLI + server + web client + GitHub Action**, all already built in
`/Users/lukasbabaliauskas/gigs/apps/evalshift`. The missing piece is the **capture SDK**: a
library that installs *inside the user's agent process* and records what the agent does, then
hands traces to the CLI **via files on disk** (the SDK and CLI never call each other).

Why now: the migration story (`source model → 4.6 → 4.7 → 4.8`) needs **real captured agent
behavior** as the source side. Currently the only way to get traces into the CLI is hand-authoring
JSONL and `evalshift traces import`. The SDK makes capture automatic, safe, and ergonomic.

Spec source: `evalshift_sdk_cli_flow.md`.

### What exists (verified)
- **evalshift-cli** — Python 3.14, Typer. Owns PyPI name `evalshift`. Defines the trace contract in
  `evalshift-cli/src/evalshift/traces/models.py` (`AgentTrace` + discriminated `TraceEvent` union:
  `model_call`/`tool_call`/`tool_result`/`retrieval`/`guardrail`/`final_output`/`error`). Reads
  traces via `evalshift traces import` → `.evalshift/runs/<run_id>/traces.jsonl`. **No** capture
  decorator, **no** `capture` subcommands, **no** `.evalshift/captures/` yet.
- **evalshift-client** — React frontend. Not the SDK.
- **evalshift-server** — FastAPI; ingests run *bundles* (`bundle.py`). Not touched by SDK.
- **evalshift-action** — composite GH Action wrapping the CLI.

### Flow-doc vs current-CLI divergence (reconciled below)
| Thing        | Flow doc                                   | CLI today                            | Plan decision |
|--------------|--------------------------------------------|--------------------------------------|---------------|
| capture dir  | `.evalshift/captures/<suite>/cap_*.json`   | none                                 | **build it (SDK writes; CLI reads in Phase 9)** |
| config file  | `aimigrate.yaml`                           | `evalshift.yaml`                     | **`evalshift.yaml`** (aimigrate = legacy) |
| curate       | `capture list/promote/clean`               | only `traces import`                 | **add capture cmds (Phase 9, CLI repo)** |
| suites       | `.evalshift/suites/<suite>/<name>.json`    | `golden.jsonl`                       | **add directory form + promote bridge (Phase 9)** |
| trace events | (unspecified)                              | `AgentTrace`/`TraceEvent`            | **reuse + extend CLI schema** |

---

## Locked decisions

- **D-pkg** New standalone repo `evalshift-sdk/`. Distribution name `evalshift-sdk`; **import name
  `evalshift`** (honors doc's `import evalshift`). Co-installing both `evalshift` (CLI) and
  `evalshift-sdk` in one env clashes on the `evalshift` top-level package — flagged as
  **D1-followup** (unify later: CLI depends on SDK, or `[cli]` extra). Not a v1 blocker; prod
  agents install only the SDK.
- **D-py** `requires-python = ">=3.10"`. Do **not** inherit CLI's 3.14 floor — would block prod
  adoption. (3.10 gives `contextvars`, modern typing, `match`.)
- **D-deps** Near-zero required runtime deps — **stdlib only** (`json`, `contextvars`,
  `dataclasses`, `hashlib`, `os`, `time`, `logging`). Framework adapters + `pydantic` parity
  checks are optional extras / dev-deps only. An embedded telemetry lib must stay light.
- **D-tooling** `uv` (matches repo), `ruff`, `mypy --strict`, `pytest`. TDD throughout.
- **D-lang** Python only for now (TS adapter deferred indefinitely).

## Decisions to freeze in Phase 0 (the doc's "before freezing schemas" list)
1. **Replay divergence policy** → default **halt-and-flag** (CLI concern), but the SDK schema MUST
   store every recorded `tool_result` as a **fixture keyed by `call_id` + input hash** so CLI
   replay can look it up. *Capture doubles as tool-result fixture.*
2. **Nondeterminism (N-sample)** → CLI/run concern. SDK records one observed run; no change.
3. **Span/event model** → **span tree** with `start_ts`/`end_ts` + `parent_call_id`, not a flat
   ordered list — required to represent concurrent tool calls. Serialized down to the CLI's
   ordered `events[]` with a stable `sequence_index` *and* preserved concurrency metadata.
4. **Redaction boundary** → redact **at capture, in-process, before any byte hits disk**. Document
   exactly what a written capture (and a promoted golden case) may still contain.
5. **Trace `schema_version`** → freeze SDK trace schema_version (sibling to CLI artifact version);
   define migration path so v1 captures stay promotable/replayable under v2.

---

## Architecture — `evalshift-sdk/src/evalshift/`

```
evalshift/
  __init__.py          # public surface: capture, configure, __version__
  capture/
    api.py             # @capture.agent (sync+async), capture.tool, record_model_call, context mgr
    span.py            # Span / SpanTree: ordering, concurrency, parentage, timestamps
    state.py           # contextvars current-span (async-safe)
  trace/
    models.py          # event/trace dataclasses mirroring CLI AgentTrace + capture metadata
    schema.py          # SCHEMA_VERSION; field constants
    serialize.py       # SpanTree -> AgentTrace JSONL + cap_<id>.json; tool-result fixtures
    migrate.py         # (Phase 8) old-version trace loaders
  sinks/
    base.py            # Sink protocol: write(capture)
    file.py            # FileSink -> .evalshift/captures/<suite>/cap_<id>.json (configurable/env)
    memory.py          # MemorySink (user flushes; Lambda/ephemeral)
  redaction/
    base.py            # Redactor protocol
    defaults.py        # email/key/PII default masks
  hygiene/
    gc.py              # max_captures / capture_ttl eviction
    dedup.py           # dedup-by-input-hash (opt-in; keep non-dedup mode)
    sample.py          # capture_sample_rate
  adapters/            # optional extras
    langchain.py
    llamaindex.py
    openai_agents.py
  config.py            # env + programmatic configure(); EVALSHIFT_CAPTURE gate
  safety.py            # fail-open guard: swallow-and-log boundary used everywhere
```

Public API target (from doc):
```python
from evalshift import capture, configure

@capture.agent(suite="support_agent", redact=my_redactor)   # off unless EVALSHIFT_CAPTURE=1
def handle_ticket(query): ...
```

---

## Phases (each independently shippable + tested; tackle one at a time)

### Phase 0 — Repo scaffold + schema freeze
**Goal:** new repo builds, lints, types, tests; trace contract frozen + decision records written.
- 0.1 Write durable `evalshift-sdk/IMPLEMENTATION_PLAN.md` (this file). ☑
- 0.2 Scaffold repo: `pyproject.toml` (dist `evalshift-sdk`, import `evalshift`, py>=3.10, uv),
  `ruff`/`mypy`/`pytest` config, `src/evalshift/`, `tests/`, `.gitignore`, `git init`, CI stub. ☑
- 0.3 `docs/DECISIONS.md` — record D-1..D-5 above with chosen defaults. ☑
- 0.4 `trace/schema.py` `SCHEMA_VERSION = "1.0.0"`; freeze field set. ☑
- 0.5 **Parity harness**: vendored frozen copy of CLI `AgentTrace` models in
  `tests/conformance/cli_models_vendored.py`; assert SDK-shaped JSON validates against it + drift
  guard (`schema.py` field set vs vendored model). ☑
- **Verify:** ☑ `uv run ruff check`, `ruff format --check`, `uv run mypy` (strict), `uv run pytest`
  all green (26 tests). schema_version proven envelope-only (D-5b).

### Phase 1 — Trace model + serialization (the contract)
**Goal:** turn a span tree into a CLI-valid `AgentTrace` + capture file with tool-result fixtures.
- 1.1 `trace/models.py` — event dataclasses (`ModelCallEvent`, `ToolCallEvent`, `ToolResultEvent`,
  …) matching CLI field names/types exactly; plus capture envelope (`capture_id`, `suite`,
  `input_hash`, `code_version`, `schema_version`, `created_at`).
- 1.2 `capture/span.py` — `Span`/`SpanTree`: parent links, start/end ts, monotonic order.
- 1.3 `trace/serialize.py` — SpanTree → ordered `events[]` (stable `sequence_index`) + concurrency
  metadata; emit `cap_<id>.json`; **store each `tool_result` keyed by `call_id`+input hash** as
  the replay fixture (D-1).
- **Verify:** unit tests — field-name parity vs CLI; ordering deterministic; concurrent spans
  serialize without `sequence_index` collisions; fixture lookup table present.

### Phase 2 — Capture core (sync) + off-by-default gate
**Goal:** `@capture.agent` (sync) records tools + model calls → writes one capture file.
- 2.1 `capture/state.py` contextvars current span; `capture/api.py` sync decorator + context
  manager + `capture.tool` / `record_model_call` helpers.
- 2.2 Wire to FileSink (basic) → one `cap_<id>.json` per invocation.
- 2.3 `config.py` + `EVALSHIFT_CAPTURE` gate — **off by default**, zero files in normal dev runs
  (problem #10); basic `safety.py` guard wrapping entrypoints.
- **Verify:** fake agent w/ 2 tools + 1 model call → capture file written + schema-valid; gate off
  → nothing written; nested tool calls parented correctly.

### Phase 3 — Sinks + config + environment robustness (problem #5)
**Goal:** SDK works in containers/Lambda/read-only FS without assuming a repo.
- 3.1 `sinks/base.py`, `sinks/file.py` (path resolution: explicit > `EVALSHIFT_DIR` env > default;
  **no repo-root walk**), `sinks/memory.py` (user flushes).
- 3.2 `configure(sink=…, redact=…, sample_rate=…, dedup=…)` programmatic API; optional
  `evalshift.yaml`/`~/.evalshift/config.toml` read (never required).
- **Verify:** path resolution precedence; MemorySink flush returns captured traces; read-only /
  missing dir → no crash, degrades to no-op or memory.

### Phase 4 — Redaction at capture time (problem #3)
**Goal:** PII masked in-process before any disk write.
- 4.1 `redaction/base.py` protocol; `redaction/defaults.py` (emails, API keys, common PII).
- 4.2 `redact=` hook applied to tool args, tool results, model input/output **before** serialize.
- 4.3 Document the data boundary (what a written/promoted/uploaded artifact may still contain) in
  `docs/REDACTION.md`.
- **Verify:** default + custom redactor mask before write; redactor never sees post-disk data;
  redactor exception is swallowed (fail-open) without dropping the user agent.

### Phase 5 — Async + streaming + concurrency (problem #6)
**Goal:** real agents: async, token streaming, concurrent tool calls.
- 5.1 Async-aware decorator + context manager; contextvars propagate across `await`.
- 5.2 Streaming model-call capture (accumulate, record final text + usage on completion).
- 5.3 Concurrent tool calls (`asyncio.gather`) recorded with correct parentage/ordering in the
  span tree.
- **Verify:** async agent; `gather` of 3 tools → all recorded, ordering stable, no contextvar
  bleed across tasks; streamed completion captured once.

### Phase 6 — Hygiene + fail-open hardening (problems #4, #10)
**Goal:** firehose self-manages; SDK can never break the host agent.
- 6.1 `hygiene/gc.py` (`max_captures` / `capture_ttl`, evict oldest); `hygiene/dedup.py`
  (dedup-by-input opt-in; keep non-dedup for target-consistency); `hygiene/sample.py`
  (`capture_sample_rate`).
- 6.2 **Fail-open audit**: every SDK boundary wrapped in `safety.py` swallow-and-log. Fault
  injection: disk full, sink raises, redactor raises, serialize raises → user call still returns.
- **Verify:** GC evicts oldest beyond cap; dedup collapses identical inputs; sample rate honored;
  fault-injection test matrix proves no exception ever propagates to the caller.

### Phase 7 — Framework adapters (problem #7)
**Goal:** zero-hand-instrument capture for popular frameworks.
- 7.1 `adapters/langchain.py` — callback handler → trace events (extra `evalshift[langchain]`).
- 7.2 `adapters/llamaindex.py`, `adapters/openai_agents.py` (LangChain first; others may slip to a
  fast-follow sub-phase).
- **Verify:** drive a minimal LangChain chain w/ one tool through the handler → schema-valid
  capture identical in shape to manual instrumentation.

### Phase 8 — Schema versioning + migration (problem #9)
**Goal:** v1 captures stay promotable/replayable under future SDK/CLI versions.
- 8.1 `trace/migrate.py` — version-tagged loaders; upgrade-on-read.
- 8.2 Document migration policy in `docs/SCHEMA.md`.
- **Verify:** an old-`schema_version` fixture loads/migrates; forward-compat test.

### Phase 9 — CLI capture lifecycle (CROSS-REPO: `evalshift-cli`)
**Goal:** complete the doc's end-to-end flow; the CLI consumes `.evalshift/captures/`.
> Touches the **other repo**. Distinct workstream; do after SDK Phases 0–2 land so there's real
> capture output to build against.
- 9.1 `evalshift capture list <suite>` / `promote <cap> --as <name>` / `clean` / `diff`.
- 9.2 `.evalshift/captures/` + `.evalshift/suites/<suite>/<name>.json`; promote **copies** capture
  into suite + attaches expected-behavior assertions (self-contained golden case).
- 9.3 `evalshift.yaml` `suites: <name>: { source: captured, path: … }`; bridge promote→suite→`run`.
- 9.4 Nice-to-haves (optional): `promote --interactive`, cost pre-flight, suite manifest.
- **Verify:** capture (SDK) → `capture promote` → `run` → verdict end-to-end on a sample agent.

### Phase 10 — End-to-end demo + docs
- Sample agent in `examples/support_agent/`; README quickstart mirroring the doc's lifecycle;
  full pipeline smoke test (capture → promote → run).

---

## Critical files
- New: everything under `evalshift-sdk/src/evalshift/` (layout above).
- Reused contract (read, mirror, parity-test against): `evalshift-cli/src/evalshift/traces/models.py`,
  `evalshift-cli/src/evalshift/traces/loader.py`.
- Phase 9 (other repo): `evalshift-cli/src/evalshift/cli/commands/` (new `capture.py`),
  `evalshift-cli/src/evalshift/cli/main.py`, config + suite loaders.

## End-to-end verification
1. `EVALSHIFT_CAPTURE=1` run the sample agent → `.evalshift/captures/support_agent/cap_*.json`
   written, schema-valid, PII redacted.
2. Capture OFF → zero files.
3. Fault injection → user agent never breaks.
4. (After Phase 9) `evalshift capture promote` → `.evalshift/suites/…` → `evalshift run` → verdict
   artifact, no re-capture.
5. `uv run ruff check && uv run mypy && uv run pytest` green each phase.

## Open follow-ups (not blocking v1)
- **D1-followup**: unify packaging so `evalshift` (CLI) + `evalshift-sdk` co-install cleanly
  (CLI-depends-on-SDK, or single dist with `[cli]` extra).
- TS SDK (deferred).
- Live sandbox tool execution during replay (doc's opt-in, later).
