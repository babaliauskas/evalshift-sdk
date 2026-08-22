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
frozen in `src/evalshift/trace/schema.py`. *Migration path implemented in Phase 8 — see D-8 and
`docs/SCHEMA.md`.* See D-5b for where the version is emitted.

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

### D-4b — redaction is opt-in, not on-by-default *(SUPERSEDED by D-4c)*
`default_redactor` (emails / API keys) was shipped as a ready-made callable users pass explicitly
(`@capture.agent(redact=default_redactor)` or `configure(redact=default_redactor)`). It did **not**
auto-run; with no `redact=` set, payloads were captured verbatim.

- **Why:** auto-masking silently mutates captured data (corrupting goldens the user wanted
  verbatim) and gives a false sense of security via inevitably-incomplete default patterns.
  Opt-in kept the masking decision explicit and auditable.
- **Precedence:** a decorator-level `redact=` overrode any process-wide `configure(redact=...)`.
- **Why it was superseded:** the reasoning held for *auto-masking* but not for the third state it
  left open — a caller who never considered the question at all. `@capture.agent(suite="x")` was a
  complete, valid call, so "decided verbatim was fine" and "never thought about it" were
  indistinguishable in the code, in review, and in a grep. See D-4c.

## Phase-11 implementation decisions (confirmed this session)

### D-4c — masking is a required, explicit choice at every capture point
`redact` is a **required** keyword on `capture.agent`, `capture.agent_session`,
`capture.agent_session_async`, and `EvalShiftCallbackHandler`, typed `Redactor | bool`:

| passed | resolves to | effect |
| --- | --- | --- |
| `True` | `default_redactor` | masks emails, `sk-…`, `AKIA…`, `Bearer …` |
| `False` | `None` | `redact_tree` skipped entirely — verbatim |
| a callable | itself | custom redactor |
| anything else, incl. `None` | — | `TypeError` |

- **Why required:** captures record the inside of a run — tool arguments and results, model input
  and output — which routinely holds PII. The failure mode of an optional parameter is silent and
  unrecoverable: unmasked customer data on disk, with nothing in the written code recording that a
  choice was made. Requiring it makes every capture point state its policy, greppable in isolation.
- **Why `bool`, not a callable-only API:** `redact=default_redactor` demands prior knowledge of the
  module before you can write a valid call. `True` / `False` answer the two common cases with no
  import. The callable form is retained because `default_redactor` covers four patterns and cannot
  reach structured secrets (SSNs, account numbers, internal id formats).
- **Cost accepted:** `redact=True` reads as "I am safe now" where `redact=default_redactor` named
  something you could go read. Mitigated in docs — the `agent()` docstring and `docs/REDACTION.md`
  enumerate exactly what `True` covers and when you need your own callable. The friction of forcing
  every user to import a symbol was judged more likely to push people toward not masking at all.
- **Why `None` is rejected:** it was the pre-0.3.0 default, so it is what a distracted caller or a
  stale example reaches for to silence the error — accepting it would reopen the exact hole.
- **Enforcement is the interpreter's**, via a keyword-only argument with no default, plus mypy
  statically. `resolve_redactor` runs at the capture point *ahead of* the `EVALSHIFT_CAPTURE` gate,
  so a wrong value fails identically whether or not capture is on.
- **Fail-open is untouched.** The `TypeError` is a programming error in the host's own call, raised
  before any span tree exists and before the user's function runs. No new `raise` was added to any
  code path that executes during a captured invocation. D-4a (fail-closed on redactor error) is
  unchanged.
- **`configure(redact=...)` removed**, along with `_Config.redact` and `active_redactor()`. One way
  to set masking: the call site. No process-wide state can change what a given agent records, and
  there is no precedence rule left to document.
- **Verified by:** `tests/test_redaction_required.py` and the handler cases in
  `tests/adapters/test_langchain.py`.

## Phase-6 implementation decisions (confirmed this session)

### D-6 — hygiene is opt-in, fail-open, and self-managing
Three independent firehose controls, all **off by default** (with no knob set, capture behaves
exactly as in Phases 0-5; `active_sink()` returns the bare sink with identity preserved):

- **Sampling decides at agent entry**, not write time. `should_capture_now()` is read at the top of
  `_run_agent`/`_run_agent_async` and both `agent_session*` context managers; an unsampled run is a
  near-pure pass-through (no span tree, no redaction, no serialize). Decision made once per run.
- **Dedup + GC compose as a `HygieneSink`** returned by `active_sink()` only when a knob is set —
  the single write seam in `capture/api.py` is unchanged. Fail-open is **granular inside the
  wrapper**: the outer `fail_open("sink write")` would drop the whole capture, so the dedup check
  and GC are guarded individually — a dedup fault defaults to "not a duplicate, write it"; a GC
  fault is swallowed after the write already succeeded. Neither can suppress the base write.

- **Why these defaults:**
  - *Sampling-on-fault → capture* (not drop): a sampling bug must never silently disable all
    telemetry.
  - *Dedup keyed by `(suite, input_hash)`* where `input_hash = canonical_hash(agent_input)` (the
    prompt). Dedup is **best-effort, per-process, in-memory** — a successful capture can suppress a
    *later identical-input failure*. Accepted for v1 because outcome-aware keying complicates the
    wrapper; flagged as a follow-up since Phase 2 deemed failed runs the highest-value telemetry.
    There is **no** cross-process / cross-run dedup. Marked seen on "base write did not raise" (so
    dedup works for `MemorySink`, whose `write` returns `None`); a `FileSink` disk-full drop still
    marks seen — a minor, documented imperfection.
  - *GC orders by filesystem **mtime**, not envelope `created_at`* — the SDK never reads `created_at`
    back, so one `os.scandir` stat pass (no JSON parse) gives both count-ordering and TTL-age
    cheaply, even at large `max_captures`. Limitation: tools that rewrite mtimes (`cp -p`, rsync,
    tar-extract) can perturb eviction order. `capture_ttl` is in **seconds**.
- **Verified by:** `tests/test_hygiene_{sample,dedup,gc,sink}.py`, `tests/test_capture_sampling.py`,
  `tests/test_capture_hygiene_e2e.py`, and the hygiene fault matrix added to
  `tests/test_capture_failopen.py` + `tests/test_capture_async_failopen.py`.

### D-6a — hygiene defaults ON (bounded), env-configurable (supersedes D-6's "off by default")
D-6 shipped every hygiene knob **off by default**, which meant a host that enabled capture but
never called `configure(...)` grew `.evalshift/captures/` without any bound — one `cap_<hex>.json`
per sampled invocation, forever. This is the single most common way the directory blows up in
practice. So the built-in defaults are now **bounded-but-generous**, and every knob is settable
from the environment (no code change required):

- **New defaults:** `dedup = True` and `max_captures = 200` per suite dir; `capture_ttl` and
  `sample_rate` stay `None` (off). 200 is larger than any realistic single golden suite, so normal
  use never loses recent data, but runaway accumulation is capped. Dedup-on only collapses
  *identical-input* re-runs (the same best-effort per-process key as D-6) — a free win against the
  re-run-the-same-eval churn pattern.
- **Env vars set the default** (read at `_Config` construction and re-read on `reset_config()`):
  `EVALSHIFT_MAX_CAPTURES`, `EVALSHIFT_CAPTURE_TTL`, `EVALSHIFT_DEDUP`, `EVALSHIFT_SAMPLE_RATE`.
  For the numeric knobs, `0` / `none` / `unlimited` means **uncapped** (`None`); a malformed value
  fails open to the built-in default (never crashes capture). Parsing lives in the `_env_*` helpers
  in `config.py`.
- **Precedence:** an explicit `configure(...)` call > env var > built-in default. `configure`'s
  merge semantics are unchanged, so in-code hosts keep full control.
- **Escape hatch:** `EVALSHIFT_MAX_CAPTURES=0` (plus `EVALSHIFT_DEDUP=off` to also drop dedup)
  restores the pre-D-6a unbounded, identity-preserved bare-sink behavior for hosts that genuinely
  want every capture kept.
- **Verified by:** `tests/test_config_env_defaults.py` (parsing, sentinels, fail-open, precedence,
  `reset_config` re-read) plus the updated seam assertions in `tests/test_config.py` /
  `tests/test_configure.py`.

## Phase-8 implementation decisions (confirmed this session)

### D-8 — schema migration: upgrade-on-read, raise-not-fail-open
Phase 8 adds the SDK's first read/deserialize path in `src/evalshift/trace/migrate.py`. Today
there is exactly one schema version (`1.0.0`), so this phase builds the **framework + policy +
reverse-load path** and tests it with *synthetic* migrations (no fake project history). Full
policy in `docs/SCHEMA.md`.

- **Dicts-first, typed reconstruction second.** Migrations are pure `dict -> dict` steps
  (tolerant of unknown fields). `load_capture` parses + migrates to a dict; `load_envelope` then
  reconstructs typed dataclasses (`envelope_from_dict` / `event_from_dict` — the inverse of
  `to_jsonable`). Reconstruction drops unknown event *fields* (forward-minor tolerance) but treats
  an unknown event *type* as a hard error.
- **Raise, don't fail open.** The migrate/load path is **read-side tooling** (the CLI consumes it
  in Phase 9), not the capture hot path. It raises typed `MigrationError` subclasses on
  unreadable/unsupported input — the opposite of `safety.py`'s host-protecting fail-open, which
  applies only while *capturing*.
- **Forward-compat = warn newer-minor, refuse newer-major.** A newer minor/patch (same major) is
  read best-effort with a `warning`; a newer major raises `UnsupportedSchemaVersionError`. Safe
  because the version governs only the envelope and the inner trace is independently CLI-valid;
  minor bumps are additive-only.
- **Refuse an older major with no registered bridge, don't fabricate.** *(Added at schema 2.0.0.)*
  A MAJOR bump normally registers a migration so older captures keep upgrading on read (see
  `docs/SCHEMA.md`'s versioning-policy footnote) — but 2.0.0 deliberately ships none from the 1.x
  major, because deriving its new `tools_offered` field for a pre-2.0.0 capture would assert facts
  that capture never recorded. `upgrade_envelope_dict` tries the registered chain first (so a
  future bump *can* still bridge a major boundary); only when that chain can't reach the target
  *and* the gap spans a major version does it raise `ObsoleteSchemaVersionError` — a distinct,
  actionable message ("re-run your agent to re-capture") instead of the internal
  `NoMigrationPathError` a missing registry edge would otherwise leak. A missing edge *within* a
  major still raises the raw `NoMigrationPathError`, unchanged — that's a registry bug, not an
  obsolete capture.
- **Linear chain, forward-only.** One registered outgoing step per version
  (`register_migration` / `reset_migrations`), walked from source to current; no downgrade path. A
  missing step raises `NoMigrationPathError`. The migrated input is never mutated (a deep copy is
  upgraded).
- **No missing-version guessing.** A capture lacking `schema_version` raises unless a caller opts
  into `default_version=`.
- **Verified by:** `tests/test_migrate.py`, `tests/test_migrate_reconstruct.py`, and
  `tests/conformance/test_migrate_parity.py` (migrated + reloaded inner traces validate against the
  vendored CLI `AgentTrace` — the promotable/replayable proof). The drift guard in
  `tests/conformance/test_parity.py` ties the reconstruction map to `schema.EVENT_TYPES`.

## Per-call toolset capture (confirmed this session)

### D-toolset — per-call toolset capture
`tools` is a **required** keyword, typed `Any`, at every model-call recording point
(`record_model_call`, `capture.model_call`) and every session-establishing point (`capture.agent`,
`capture.agent_session`, `capture.agent_session_async`, `EvalShiftCallbackHandler`):

| passed | effect |
| --- | --- |
| a real toolset (Anthropic / OpenAI / Gemini shape, or a mixed list) | normalised, fingerprinted, written once to a content-addressed sidecar, and stamped onto the `model_call` event as `toolset_ref` + `tools_offered` |
| `[]` | a real, asserted "this call/session had no tools" — a first-class value, not a default |
| `None` (call-level only: `record_model_call` / `capture.model_call`) | defer to the enclosing session's own `tools=` |
| anything unrecognised (any shape `normalize_tools` rejects) | neither field is stamped; logged at `debug`; the capture is left structurally invalid for that event — never raises |
| omitted entirely | `TypeError` (no default in the signature) |

- **Why per-call, not config:** a real agent can switch toolsets mid-run — one process, one suite,
  two toolsets, chosen by an `if` at call time. A config-level or session-only toolset can't express
  that; the switching case is why this exists at all. It is also the concrete failure this closes:
  an agent that calls its model with no tools in production, evaluated against a 52-tool manifest
  built for a *different* agent, inflated its prompt from 779 to 9,342 tokens; every row came back
  `finish_reason=tool_calls, text=""`; every text evaluator skipped; the run measured nothing.
- **Why content-addressed, not inline:** a real toolset schema runs ~47KB against a ~4.5KB average
  capture — inlining would make captures roughly an order of magnitude larger (~11x). A toolset is
  instead written once per distinct fingerprint by `ToolsetSink`
  (`<base>/toolsets/<hex>.json`, content-addressed so a repeat write is a cheap existence check, not
  a second write) and every `model_call` that used it carries only a `toolset_ref` pointer plus the
  cheap, display-only `tools_offered` name list.
- **Why required, not optional — same reasoning as D-4c's `redact=`.** An optional `tools=`
  defaulting to `[]` or `None` would make an un-instrumented call site indistinguishable from a real
  empty-toolset run — exactly the silent, plausible-looking wrong data this plan exists to eliminate.
  Enforcement is the interpreter's, via a keyword-only argument with no default, plus mypy
  statically — identical mechanism to D-4c. Unlike `redact=`, though, an unrecognised *value* never
  raises: `normalize_tools` degrades to `None` and the two fields are simply left unstamped, because
  the failure mode here is "we don't know what tools were offered" (an honest gap the CLI can refuse
  to promote at promotion time) rather than "we are about to write unmasked PII to disk" (a leak
  `redact=` must prevent by raising before any capture work starts).
- **The empty toolset is a real, first-class value**, exactly as in `normalize_tools` /
  `ToolsetSink`. `tools=[]` fingerprints, gets a real sidecar (`"tools": []`), and gets a real
  `toolset_ref` like any other toolset — nothing branches on "no tools" as if it meant "nothing to
  record."
- **`tools=None` means "inherit the session", only at the two call-level entry points.** A session
  (`agent` / `agent_session` / `agent_session_async`) normalises its own `tools=` once at entry and
  stores the `(normalized, fingerprint)` pair in a new `evalshift.capture.state` contextvar
  (`_current_toolset`, exposed as `current_toolset()` / `use_toolset()` — the module's third,
  alongside the pre-existing `_current_tree` / `_current_parent`); a `record_model_call` /
  `capture.model_call` that passes `tools=None` reads it back (`capture/api.py`'s
  `_resolve_call_toolset`). A call's own non-`None` value always wins, even across repeated calls in
  the same session that each choose differently. `None` has no such meaning at the three
  session-establishing points — there is nothing to inherit *from* there — so it is treated like any
  other unrecognised value there (degrades, does not raise, does not mean "inherit"). The LangChain
  adapter does not use this contextvar at all: like `_current_tree`, it keeps its own separate state
  (see its module docstring), and its one constructor-level `tools=` is stamped directly onto every
  `model_call` span it ever opens — there is no per-call override surface in a LangChain callback.
- **No allow-list**, unlike `generation_config`. An `input_schema` is arbitrary user JSON needed in
  full to dispatch the tool; normalisation only recognises or rejects tool *shapes* (Anthropic /
  OpenAI / Gemini), never prunes keys within a schema.
- **Not redacted, like `generation_config` — but for a different mechanical reason**, stated
  precisely so nobody "fixes" it later. `generation_config` is exempt because it lives in
  `span.metadata`, which `redact_tree` (`redaction/base.py`) never walks. `tools_offered` /
  `toolset_ref` are top-level `span.data` fields — the same dict as `input` / `output` — and are
  safe only because `_REDACTABLE_FIELDS["model_call"]` names exactly `("input", "output")`; neither
  toolset field is in that tuple. A future change that widens that tuple wholesale (e.g. to `"*"` or
  a computed set) rather than naming fields individually would silently start redacting these two.
- **No garbage collection for toolsets — verified by reading, not assumed.** `hygiene/gc.py`'s
  `evict` never recurses into subdirectories (its own docstring says so), and its only caller
  (`sinks/hygiene.py`) always passes `<base>/captures/<suite>/` — a sibling of, never an ancestor
  of, `<base>/toolsets/`. Distinct toolsets are few and captures number in the thousands, so an
  evicted toolset would orphan every capture that still references it, not just the one just
  evicted; orphan sweeping (if ever needed) is a CLI-side concern, not this SDK's.
- **No migration.** Schema 2.0.0 (D-8) already registers no edge from the 1.x major for exactly this
  reason — there is no honest value `tools_offered` can take for a capture written before per-call
  toolset capture existed. This decision does not reopen that question; it is the feature D-8's
  schema bump was written in anticipation of.
- **Verified by:** `tests/test_toolset.py` / `tests/test_toolset_sink.py` (Tasks 1/5 — the pure
  normalise/fingerprint/sidecar-write logic this decision builds on), `tests/test_state.py` (the new
  `_current_toolset` contextvar), `tests/test_serialize.py` (the serializer reads the two fields
  straight off `span.data`), `tests/test_capture_toolset.py` (all six entry points end to end:
  required-kwarg `TypeError`, round trip, session inheritance, per-call override, unrecognised-value
  degrade, redaction exemption, and — since a rejected toolset is undiagnosable without it — the
  `debug`-level log line itself), `tests/adapters/test_langchain.py` (the handler's own required
  `tools=` and its stamping on `on_llm_start` / `on_chat_model_start`), `tests/test_hygiene_gc.py`
  / `tests/test_capture_hygiene_e2e.py` (the GC-never-touches-`toolsets/` regression, both
  structurally and end to end through a real eviction pass), and
  `tests/test_capture_concurrent.py` / `tests/test_thread_safety.py` (`_current_toolset` isolation
  under real `asyncio.gather` interleaving and real OS threads — two concurrent sessions, two
  different toolsets, neither one's `tools_offered` ever attributed to the other's call; the same
  contextvar-bleed class the sibling/nested `parent_call_id` tests above it already guard, now
  guarded for toolsets too).

## Open follow-ups (not blocking v1)
- **D1-followup:** unify packaging so CLI + SDK co-install cleanly (CLI-depends-on-SDK, or a single
  dist with a `[cli]` extra).
- **Outcome-aware dedup key** so a success can't suppress a later identical-input failure.
- **Cross-process / on-disk dedup** (the v1 registry is per-process, in-memory).
- **GC throttling** (every-Nth-write or a background thread) if profiling shows latency at large
  `max_captures`; a background thread reintroduces the Phase-5 threading concerns — defer
  deliberately.
- TS SDK (deferred).
- Live sandbox tool execution during replay (spec's opt-in, later).
