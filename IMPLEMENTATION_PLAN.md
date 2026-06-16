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
| 5 | Async + streaming + concurrency | ☐ not started |
| 6 | Hygiene + fail-open hardening | ☐ not started |
| 7 | Framework adapters | ☐ not started |
| 8 | Schema versioning + migration | ☐ not started |
| 9 | CLI capture lifecycle (cross-repo) | ☐ not started |
| 10 | End-to-end demo + docs | ☐ not started |

Legend: ☐ not started · ◐ in progress · ☑ done

**Last verified 2026-06-16:** Phases 0–4 confirmed complete — 139 tests pass,
`mypy --strict` + `ruff` + `ruff format --check` clean. Phase 5 not started (next up).

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
