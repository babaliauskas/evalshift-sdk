# Redaction & the capture data boundary

EvalShift captures the *inside* of an agent run — tool arguments, tool results, model inputs and
outputs. That payload routinely contains PII (emails, names, API keys). This document defines
**where redaction happens, what it guarantees, and what a written capture may still contain.**

See `docs/DECISIONS.md` D-4 / D-4a / D-4b for the locked decisions behind this.

## TL;DR

- Redaction runs **in-process, before any byte hits disk** (D-4).
- It is **opt-in**: nothing is masked unless you pass a `redact=` callable (D-4b).
- If your redactor **raises, the capture is dropped** — never written half-masked (D-4a). Your
  agent is unaffected.

## How to enable it

A *redactor* is any callable `(value) -> redacted_value`. Ship-with default `default_redactor`
masks emails and common API keys.

Per-agent (wins over global):

```python
from evalshift import capture, default_redactor

@capture.agent(suite="support_agent", redact=default_redactor)
def handle_ticket(query): ...
```

Process-wide:

```python
from evalshift import configure, default_redactor

configure(redact=default_redactor)
```

Custom redactor (full control — recurse however you like, return a redacted copy):

```python
def redact(value):
    if isinstance(value, dict):
        return {k: ("***" if k == "ssn" else redact(v)) for k, v in value.items()}
    if isinstance(value, str):
        return value.replace(SECRET, "[REDACTED]")
    return value

configure(redact=redact)
```

**Precedence:** decorator `redact=` → `configure(redact=...)` → none (verbatim capture).

## Where it runs (the boundary)

Redaction is applied in `build_capture` (`src/evalshift/trace/serialize.py`) by `redact_tree`
(`src/evalshift/redaction/base.py`), **before** the span tree is serialized into trace events and
**before** the `Sink` writes anything. The redactor only ever sees in-memory payloads; it never
sees on-disk bytes.

Redactable fields, per recorded span kind:

| Span kind      | Fields passed to the redactor          |
|----------------|----------------------------------------|
| `tool`         | `arguments`, `result`, `error`         |
| `model_call`   | `input`, `output`                      |
| `retrieval`    | `query`, `documents`                   |
| `guardrail`    | `reason`                               |
| `final_output` | `text`                                 |
| `error`        | `message`                              |

Because redaction runs **before** serialization, the tool replay-fixture key
(`(call_id, input_hash)`, D-1) is computed from the **redacted** arguments — the capture stays
internally consistent.

## Guarantees

- **Fail-closed (D-4a).** A redactor that raises drops the whole capture (no file) and logs one
  debug line. We never write a possibly-unredacted file. The host agent still returns normally.
- **Host-safe.** Redaction errors can never propagate to your agent (`safety.guard` boundary).
- **No mutation of your data.** `default_redactor` returns redacted copies; it does not mutate the
  values your agent passed around. (Custom redactors should do the same.)

## What a written / promoted artifact MAY still contain

Redaction is a masking pass over payload **values**. It does **not** scrub:

- **Structural metadata** — span/tool `name`s, `model_id`, token counts, costs, timestamps,
  `sequence_index`, `call_id`/`parent_call_id`, and the `metadata["evalshift"]` concurrency block.
- **The capture envelope** — `capture_id`, `suite`, `code_version`, `created_at`, and the
  `input_hash` (a one-way SHA-256 of the agent's bound input; the raw input itself is **never**
  stored, so it is not separately redacted).
- **Anything your redactor misses.** `default_redactor` covers emails and a few key formats only
  (precise patterns, to avoid mangling benign text). It is **not** a comprehensive PII scrubber —
  supply a domain-specific redactor when your data has structured secrets.

A **promoted golden case** (Phase 9) is a *copy* of a capture, so it inherits exactly the boundary
above: redacted payloads, intact structure + envelope. Redact at capture time if you do not want a
value to reach a golden suite or an upload.

## Limitations (this phase)

- Opt-in only — no value is masked without an explicit `redact=`.
- `default_redactor` patterns are intentionally conservative (emails, `sk-…`, AWS `AKIA…`,
  `Bearer …`); extend or replace for your domain.
- Sampling / dedup hygiene (Phase 6) compose around redaction but do not change the boundary.
