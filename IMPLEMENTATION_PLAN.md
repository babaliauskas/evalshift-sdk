# EvalShift SDK — Implementation Plan

Sub-phased, resumable. Tackle one phase at a time. Update the **Status** column as phases land.

## Status tracker
| Phase | Title | Status |
|-------|-------|--------|
| 0 | Repo scaffold + schema freeze | ☑ done |
| 1 | Trace model + serialization | ☑ done |
| 2 | Capture core (sync) + off-by-default gate | ☑ done |
| 3 | Sinks + config + env robustness | ☑ done |
| 4 | Redaction at capture | ☑ done |
| 5 | Async + streaming + concurrency | ☑ done |
| 6 | Hygiene + fail-open hardening | ☑ done |
| 7 | Framework adapters | ◐ LangChain done; llamaindex + openai_agents = 7.2 fast-follow |
| 8 | Schema versioning + migration | ☑ done |
| 9 | CLI capture lifecycle (cross-repo) | ☐ not started |
| 10 | End-to-end demo + docs | ☐ not started |

Legend: ☐ not started · ◐ in progress · ☑ done

**Last verified 2026-06-17:** Phases 0–6 + Phase 7.1 (LangChain adapter) + Phase 8 (schema
versioning + migration) complete — 281 tests pass, `mypy --strict` + `ruff` + `ruff format --check`
clean. Phase 7.2 (llamaindex + openai_agents fast-follow) or Phase 9 (CLI capture lifecycle,
cross-repo) next up.

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
  `input_hash`, `code_version`, `schema_version`, `created_at`). ☑ (stdlib dataclasses; `to_jsonable`)
- 1.2 `capture/span.py` — `Span`/`SpanTree`: parent links, start/end ts, monotonic order. ☑
- 1.3 `trace/serialize.py` — SpanTree → ordered `events[]` (stable `sequence_index`) + concurrency
  metadata; emit `cap_<id>.json`; **store each `tool_result` keyed by `call_id`+input hash** as
  the replay fixture (D-1). ☑ (pure — disk write deferred to Phase 2 FileSink)
- **Verify:** ☑ unit tests — field-name parity vs CLI; ordering deterministic; concurrent spans
  serialize without `sequence_index` collisions; fixture lookup table present.
  `ruff`/`format`/`mypy --strict`/`pytest` all green (46 tests). Design notes:
  tool-result fixtures live **inside the trace** (frozen `ENVELOPE_KEYS` has no `fixtures` slot —
  "capture doubles as fixture"); concurrency timing lives in `event.metadata["evalshift"]` (the
  only home surviving the CLI's `extra="forbid"` events); capture identity derives from the
  capture (`role=source`, `prompt_id=suite`, `run_id=example_id=capture_id`), rewritten on promote.

### Phase 2 — Capture core (sync) + off-by-default gate
**Goal:** `@capture.agent` (sync) records tools + model calls → writes one capture file.
- 2.1 `capture/state.py` contextvars current span; `capture/api.py` sync decorator + context
  manager + `capture.tool` / `record_model_call` helpers. ☑ (two `ContextVar`s — current
  `SpanTree` + current parent `call_id`; push/pop via reset tokens, async-safe for Phase 5;
  `_bind` uses `inspect.signature(...).bind().apply_defaults()` with a safe fallback)
- 2.2 Wire to FileSink (basic) → one `cap_<id>.json` per invocation. ☑ (`sinks/file.py`;
  `<base>/captures/<suite>/cap_<id>.json`, base = explicit > `EVALSHIFT_DIR` > `.evalshift`,
  no repo-root walk; routed via `config.active_sink()` seam — Phase 3 swaps the body only)
- 2.3 `config.py` + `EVALSHIFT_CAPTURE` gate — **off by default**, zero files in normal dev runs
  (problem #10); basic `safety.py` guard wrapping entrypoints. ☑ (gate read live, truthy
  allow-list; `safety.fail_open`/`guard` swallow-and-log at debug — wraps **only** bookkeeping,
  never the user fn)
- **Verify:** ☑ gate off → zero files; gate on → one schema-valid `cap_*.json`; nested tool
  parentage; model call recorded; user exception **propagates** + capture-with-`error`-event
  still written (D-2.3 choice); sink/build/open faults never break host; `redact=` accepted +
  stored (applied Phase 4). `ruff`/`format`/`mypy --strict`/`pytest` all green (109 tests).
  Decision: on user exception, record `error` event + still write the (partial) capture, then
  bare `raise` — failed runs are the highest-value telemetry.

### Phase 3 — Sinks + config + environment robustness (problem #5)
**Goal:** SDK works in containers/Lambda/read-only FS without assuming a repo.
- 3.1 `sinks/base.py` (`Sink` `Protocol`, `runtime_checkable`, `write -> Path | None`),
  `sinks/file.py` (path resolution explicit > `EVALSHIFT_DIR` > default, **no repo-root walk**;
  now: `OSError` → drop + debug-log + return `None`; `_safe_segment` sanitizes `suite` vs
  traversal), `sinks/memory.py` (`MemorySink`: `flush()` drains, `captures` non-draining view). ☑
- 3.2 `configure(sink=…, redact=…, sample_rate=…, dedup=…)` programmatic API (merge semantics,
  `_UNSET` sentinel; only `sink` wired — `redact`/`sample_rate`/`dedup` stored placeholders for
  Phase 4/6) + `reset_config()`; `active_sink() -> Sink` swap seam (config-body-only change,
  `_finalize` untouched). Config-file read (`evalshift.yaml`/`config.toml`) **deferred** —
  stdlib-only/py3.10 floor; becomes an optional extra later. ☑
- **Verify:** ☑ path-resolution precedence; both sinks satisfy `Sink`; MemorySink flush returns
  then clears; suite traversal stays under `<base>/captures`; read-only FS → no crash, no file,
  one debug line, capture dropped; `configure(sink=MemorySink())` routes capture to memory with
  zero disk writes; autouse conftest fixture isolates global config. 121 tests pass; `ruff` /
  `ruff format --check` / `mypy --strict` clean.
  Decisions (confirmed this session): degradation = **no-op drop + debug log** (no silent
  memory fallback — unbounded in long-lived hosts); config-file reading **deferred**.

### Phase 4 — Redaction at capture time (problem #3)
**Goal:** PII masked in-process before any disk write.
- 4.1 `redaction/base.py` (`Redactor` `Protocol` + `redact_tree` walker); `redaction/defaults.py`
  (`default_redactor`: emails, `sk-`/`Bearer`/AWS keys, recursive over dict/list/tuple). ☑
- 4.2 `redact=` hook applied to tool args/results, model input/output (+ retrieval/guardrail/
  final_output/error fields) **before** serialize — inside `build_capture`, so the tool
  `input_hash` derives from redacted args. Threaded decorator→`_run_agent`→`_finalize`; precedence
  decorator `redact=` > `configure(redact=)` (`config.active_redactor()`). ☑
- 4.3 `docs/REDACTION.md` — data boundary documented (payloads masked; structure + envelope +
  one-way `input_hash` retained); `DECISIONS.md` D-4 marked implemented + D-4a/D-4b added. ☑
- **Verify:** ☑ default + custom redactor mask before write; global vs decorator precedence;
  no-redactor → verbatim; `input_hash` from redacted args; **redactor exception → capture dropped
  (fail-closed, D-4a) while host agent still returns**. `ruff`/`format`/`mypy --strict`/`pytest`
  all green (139 tests). Decisions: **fail-closed drop** on redactor error (not fail-open-data);
  redaction **opt-in** (`default_redactor` shipped but never auto-runs).

### Phase 5 — Async + streaming + concurrency (problem #6)
**Goal:** real agents: async, token streaming, concurrent tool calls.
- 5.1 Async-aware decorator + context manager; contextvars propagate across `await`. ☑
  (`@capture.agent`/`@capture.tool` auto-detect coroutine fns via `inspect.iscoroutinefunction`
  → async wrapper over `_run_agent_async`/`_run_tool_async`; `agent_session_async`
  `@asynccontextmanager`. `state.py` reused **unchanged** — the keystone.)
- 5.2 Streaming model-call capture (accumulate, record final text + usage on completion). ☑
  (new `capture.model_call(model_id=, input=)` → `_ModelCallRecorder`, dual-protocol
  `with`/`async with`; `add_text`/`set_usage`; opens span on enter, records once on exit;
  `record_model_call` unchanged for atomic calls.)
- 5.3 Concurrent tool calls (`asyncio.gather`) recorded with correct parentage/ordering in the
  span tree. ☑ (correct gather parentage comes free from contextvars copy-on-task-creation;
  added `threading.Lock` to `SpanTree` order-assign/append + `MemorySink` so threaded tools via
  `asyncio.to_thread`/`run_in_executor` can't race `_counter`.)
- **Verify:** ☑ async agent → one schema-valid capture; `gather` of 3 tools → all recorded,
  siblings `parent_call_id` null, `sequence_index` dense/no collisions; nested gather parentage;
  no contextvar bleed across tasks; streamed completion captured once (output accumulated, usage
  stored, omitted-usage derives latency); async exception → error event + partial capture +
  propagate; thread-safety smoke (64-thread SpanTree unique orders, 100-write MemorySink).
  `ruff`/`format`/`mypy --strict`/`pytest` all green (165 tests). Dev-dep `pytest-asyncio` +
  `asyncio_mode="auto"` added; runtime stays stdlib-only. Decisions: auto-detect decorators;
  new `model_call` recorder for streaming; locks for threaded-tool safety. Limitation documented:
  raw `threading.Thread` tools (not via `to_thread`/`run_in_executor`) start a fresh contextvars
  context → `current_tree()` is `None` → silently uncaptured.

### Phase 6 — Hygiene + fail-open hardening (problems #4, #10)
**Goal:** firehose self-manages; SDK can never break the host agent.
- 6.1 `hygiene/gc.py` (`max_captures` / `capture_ttl`, evict oldest by mtime); `hygiene/dedup.py`
  (per-process dedup-by-input opt-in; keep non-dedup for target-consistency); `hygiene/sample.py`
  (`sample_rate`, decided at agent entry). ☑ Composed via `sinks/hygiene.py` `HygieneSink` wrapper
  returned by `config.active_sink()` only when a knob is set (bare sink otherwise — identity
  preserved); sampling gates at all four agent entry points via `config.should_capture_now()`.
- 6.2 **Fail-open audit**: every SDK boundary wrapped in `safety.py` swallow-and-log; dedup + GC
  guarded *individually inside* `HygieneSink` so a hygiene fault never drops the base write. Fault
  injection: sink raises (OSError + non-OSError), redactor raises, serialize raises, sampling RNG
  raises, dedup raises, GC raises, `HygieneSink` ctor raises → user call still returns. ☑
- **Verify:** ☑ GC evicts oldest beyond cap + TTL-aged; dedup collapses identical inputs (and a
  success suppresses a later identical-input failure — best-effort, D-6); sample rate honored at all
  four entry points (`0.0`→zero files, `1.0`/None→one, mid-rate via patched RNG, unsampled run is a
  true pass-through); fault matrix (sync + async) proves no exception reaches the caller.
  `ruff`/`format`/`mypy --strict`/`pytest` all green (221 tests). Decisions (D-6): sampling-on-fault
  → capture; dedup per-process/in-memory/best-effort; GC by filesystem mtime, `capture_ttl` in
  seconds. Follow-ups logged: outcome-aware dedup key, cross-process dedup, GC throttling.

### Phase 7 — Framework adapters (problem #7)
**Goal:** zero-hand-instrument capture for popular frameworks.
- 7.1 `adapters/langchain.py` — `EvalShiftCallbackHandler(BaseCallbackHandler)`. ☑
  Callbacks fire **flat** (`run_id`/`parent_run_id` UUIDs), not nested on the call stack, so the
  handler does **not** use the contextvar machinery (`capture/state.py`): it owns its own
  `SpanTree` per root run, keeps `run_id → span` maps, resolves `parent_call_id` by walking the
  `parent_run_id` chain to the nearest enclosing *tool*, and finalizes via `api._finalize` at the
  root-run boundary. No `state.use_tree` binding → no double-record when mixed with decorators.
  Honors the gate + sampling at root start; redaction via the shared finalize path (decorator
  `redact=` > `configure(redact=)`); every callback body wrapped in `safety.fail_open`. Framework
  payloads (chat messages, retrieved docs, tool args) coerced to JSON-able primitives so a
  non-serializable object can't silently drop the capture (FileSink's `dumps` has no `default=`).
  Provenance stamped at `event.metadata["adapter"]="langchain"` + `["lc_run_id"]` (top-level
  metadata is a freeform dict; survives the CLI's `extra="forbid"` alongside the serializer's own
  `metadata["evalshift"]` block). `langchain-core` is import-guarded (`TYPE_CHECKING` import +
  runtime `try/except`) → runtime stays stdlib-only; added `[project.optional-dependencies]
  langchain = ["langchain-core>=0.2"]` + dev-dep for the smoke test.
- 7.2 `adapters/llamaindex.py`, `adapters/openai_agents.py` — **deferred to fast-follow** (their
  event models differ; LangChain shipped first per the plan's own allowance). ☐
- **Verify:** ☑ synthetic-callback unit tests (no langchain dep): chain→tool→model schema-valid +
  event counts match manual instrumentation; nested-tool parentage (model call's
  `metadata.evalshift.parent_call_id` == enclosing tool's `call_id`; sibling top-level →
  null); tool-error → `tool_result.error`; chain-error → `error` event; gate off → zero files;
  `sample_rate` 0.0 → zero / 1.0 → one; redactor-fault → capture dropped, callback never raises;
  malformed kwargs (`serialized=None`, end for unknown run_id) never raise; provenance metadata;
  handler reuse → 2 distinct captures, no state bleed. Plus one `importorskip`-guarded **real
  `RunnableLambda` + `@tool` chain** smoke (proves real callback signatures match — langchain-core
  1.4.7). `ruff`/`format`/`mypy --strict`/`pytest` all green (233 tests). Decisions: handler owns
  its own tree (no contextvars); finalize reuses `api._finalize` (single redaction/sink path);
  provenance as top-level event metadata keys; payloads coerced to JSON primitives.

### Phase 8 — Schema versioning + migration (problem #9)
**Goal:** v1 captures stay promotable/replayable under future SDK/CLI versions.
- 8.1 `trace/migrate.py` — the SDK's **first read/deserialize path**. ☑ Two layers:
  dict-level **upgrade-on-read** (`load_capture`/`upgrade_envelope_dict` walk a registered linear
  chain of pure `dict→dict` steps to current) + typed **reconstruction** (`load_envelope`/
  `envelope_from_dict`/`event_from_dict`, the inverse of `to_jsonable`; unknown event *type* →
  hard error, unknown *fields* dropped). `SchemaVersion` (frozen, ordered) parses `MAJOR.MINOR.PATCH`;
  `register_migration`/`reset_migrations` populate a lock-guarded `_REGISTRY` (empty at real import —
  `1.0.0` is the only version, so the chain is exercised with **synthetic** migrations in tests, no
  fake history). Read-side tooling → **raises** typed `MigrationError` subclasses (not fail-open like
  the capture path). `schema.py` gains `SUPPORTED_SCHEMA_VERSIONS`; top-level + `trace/__init__` export
  `load_capture`/`load_envelope`/`MigrationError`/`register_migration`.
- 8.2 `docs/SCHEMA.md` — versioning + migration policy documented; `DECISIONS.md` D-8 added, D-5
  marked implemented. ☑
- **Verify:** ☑ synthetic old-version fixture migrates (single + multi-step chain; missing step →
  `NoMigrationPathError`); **forward-compat** newer-minor tolerated with a logged warning, newer-major
  → `UnsupportedSchemaVersionError`; missing/invalid/non-dict version → typed errors; migration never
  mutates input; bad JSON → `UnreadableCaptureError`; reconstruction round-trips
  (`build→dumps→load_envelope` is identity; reloaded trace yields the same fixture table; `_parse_dt`
  handles `Z`). Conformance: migrated + reloaded inner traces validate against the vendored CLI
  `AgentTrace` (promotable + replayable); migration never leaks `schema_version` into the trace; drift
  guard ties the reconstruction map to `schema.EVENT_TYPES`. `ruff`/`format`/`mypy --strict`/`pytest`
  all green (281 tests). Decisions (D-8): dicts-first + opt-in typed reconstruction; raise-not-fail-open
  on read; warn-newer-minor / refuse-newer-major; linear forward-only chain.

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
