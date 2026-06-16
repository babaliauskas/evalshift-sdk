# Decision records — evalshift-sdk

Locked design decisions for the capture SDK. Each entry: decision, rationale, status.
See `IMPLEMENTATION_PLAN.md` for the phased roadmap.

## Packaging & toolchain (locked, Phase 0)

### D-pkg — standalone repo, dist `evalshift-sdk`, import `evalshift`
New repo `evalshift-sdk/`. Distribution name `evalshift-sdk`; **top-level import `evalshift`**
(honors the spec's `import evalshift`). Co-installing the CLI (`evalshift`) and `evalshift-sdk`
in one env clashes on the `evalshift` top-level package — tracked as **D1-followup** (unify
later: CLI depends on SDK, or a `[cli]` extra). Not a v1 blocker; prod agents install the SDK only.

### D-py — `requires-python = ">=3.10"`
Do **not** inherit the CLI's 3.14 floor — it would block prod adoption. 3.10 gives `contextvars`,
modern typing, `match`. CI matrix covers 3.10–3.14.

### D-deps — stdlib-only runtime
Runtime imports limited to stdlib (`json`, `contextvars`, `dataclasses`, `hashlib`, `os`, `time`,
`logging`). An embedded telemetry lib must stay light. Framework adapters and `pydantic` parity
checks are optional extras / dev-deps only — **`pydantic` is never imported at runtime.**

### D-tooling — uv + ruff + mypy --strict + pytest, TDD
Matches the CLI repo's toolchain. Tool config consolidated in `pyproject.toml`.

### D-lang — Python only for now
TS/JS adapter deferred indefinitely.

## Schema-freeze decisions (the spec's "before freezing schemas" list)

### 1. Replay divergence / tool-result fixtures
CLI default policy is **halt-and-flag** (a CLI concern). The SDK schema MUST store every recorded
`tool_result` as a **fixture keyed by `call_id` + input hash** so CLI replay can look it up —
capture doubles as a tool-result fixture. *Implemented in Phase 1 serialize.*

### 2. Nondeterminism (N-sample)
A CLI/run concern. The SDK records one observed run; no schema change.

### 3. Span/event model
Capture as a **span tree** (`start_ts`/`end_ts` + `parent_call_id`) to represent concurrent tool
calls, serialized down to the CLI's ordered `events[]` with a stable `sequence_index` plus
preserved concurrency metadata. *Implemented in Phases 1 and 5.*

### 4. Redaction boundary
Redact **at capture, in-process, before any byte hits disk**. *Implemented in Phase 4* — payloads
are masked in `build_capture` **before** serialization, so trace events and the derived tool
`input_hash` see only redacted values. The data boundary (what a written capture / promoted golden
may still contain) is documented in `docs/REDACTION.md`. See D-4a / D-4b for the policy choices.

### 5. Trace `schema_version`
SDK trace schema is versioned independently of the CLI artifact version; `SCHEMA_VERSION = "1.0.0"`
frozen in `src/evalshift/trace/schema.py`. Migration path defined in Phase 8. See D-5b for where
the version is emitted.

## Phase-0 implementation decisions (confirmed this session)

### D-5a — parity harness = vendored frozen copy
The Phase 0.5 parity test validates SDK-shaped JSON against a **frozen verbatim copy** of the CLI
`AgentTrace` models at `tests/conformance/cli_models_vendored.py`, not a live dependency on the CLI.

- **Why:** keeps the parity test hermetic and keeps `pydantic` a dev-only dependency. A live
  dev-dep on `evalshift-cli` would drag its `requires-python = ">=3.14"` floor into the SDK's dev
  and CI environments, conflicting with D-py.
- **Cost:** the copy can drift from the CLI. Mitigated by the drift-guard assertions in
  `test_parity.py` (event types + every field set + roles checked against the vendored model) and
  a header in the vendored file pointing at the source path. Re-sync on CLI contract changes.
- **Sole adaptation vs source:** `Self` is imported from `typing_extensions` (CLI uses
  `typing.Self`, which is py3.11+) so the test runs on py3.10.

### D-5b — `schema_version` lives in the capture envelope only
The CLI `AgentTrace` is `extra="forbid"` and has **no `schema_version` field** (and no root
`metadata`). Therefore `SCHEMA_VERSION` is emitted only in the **capture envelope** wrapper
(`cap_<id>.json`: `schema_version`, `capture_id`, `suite`, `input_hash`, `code_version`,
`created_at`, `trace`) — **never inside the `AgentTrace` JSONL**, which stays byte-identical to
the CLI contract.

- **Why:** embedding the version inside the trace would fail CLI validation (extra field forbidden).
  The envelope is the SDK's own artifact and can carry SDK-specific metadata freely.
- **Verified by:** `test_schema_version_lives_in_envelope_not_trace` and
  `test_schema_version_inside_trace_is_rejected` in `tests/conformance/test_parity.py`.
- **Alternative rejected:** stashing the version inside an event's `metadata` dict — technically
  CLI-valid but couples the version to event payloads.

## Phase-4 implementation decisions (confirmed this session)

### D-4a — fail-closed on redactor error (drop the capture)
If the user's redactor raises mid-capture, the **capture is dropped** (no file written) and a
debug line is logged; the host agent is unaffected (it has already returned).

- **Why:** a redactor crash could otherwise leave a partially- or un-redacted payload on disk —
  the worst outcome for a PII-masking feature. Fail-open is sacred *for the host*, but the data
  must fail **closed**. We never trade a possibly-leaked file for one more telemetry sample.
- **How:** redaction runs inside `build_capture`, which `_finalize` already wraps in
  `safety.guard("build capture", ...)`. A raising redactor propagates to that guard → `None`
  envelope → no write. `redact_tree` deliberately holds no `try`/`except` of its own.
- **Verified by:** `test_redactor_failure_drops_capture_but_host_returns` in
  `tests/test_capture_redaction.py`.

### D-4b — redaction is opt-in, not on-by-default
`default_redactor` (emails / API keys) is shipped as a ready-made callable users pass explicitly
(`@capture.agent(redact=default_redactor)` or `configure(redact=default_redactor)`). It does **not**
auto-run; with no `redact=` set, payloads are captured verbatim (unchanged from Phases 2–3).

- **Why:** auto-masking silently mutates captured data (corrupting goldens the user wanted
  verbatim) and gives a false sense of security via inevitably-incomplete default patterns.
  Opt-in keeps the masking decision explicit and auditable.
- **Precedence:** a decorator-level `redact=` overrides any process-wide `configure(redact=...)`.

## Open follow-ups (not blocking v1)
- **D1-followup:** unify packaging so CLI + SDK co-install cleanly (CLI-depends-on-SDK, or a single
  dist with a `[cli]` extra).
- TS SDK (deferred).
- Live sandbox tool execution during replay (spec's opt-in, later).
