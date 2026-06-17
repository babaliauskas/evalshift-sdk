# Schema versioning & migration

A written capture (`cap_<id>.json`) carries a `schema_version` in its **envelope**; the inner
`AgentTrace` is the frozen CLI contract and carries no version of its own (D-5b). This document
defines how that version evolves and how an old capture stays readable — promotable and
replayable — under a newer SDK or CLI.

See `docs/DECISIONS.md` D-5 / D-5b (where the version lives) and D-8 (the read/migration policy).

## TL;DR

- `schema_version` lives in the **envelope only**; the inner `AgentTrace` is always CLI-valid.
- **MAJOR** = breaking envelope change (needs a migration); **MINOR** = additive; **PATCH** = no
  field change.
- Reading is **upgrade-on-read**: `load_capture()` / `load_envelope()` migrate old → current.
- A **newer minor/patch** is tolerated (warn + best-effort read); a **newer major** is refused.
- The migrate/load path **raises** typed errors — it does **not** fail open like the capture path.

## The versioned surface

`schema_version` governs the **capture envelope** — the SDK's own wrapper — not the trace inside:

| Envelope key (`schema.ENVELOPE_KEYS`) | Versioned by `schema_version`            |
|---------------------------------------|------------------------------------------|
| `schema_version`                      | the version itself                       |
| `capture_id`, `suite`, `input_hash`, `code_version`, `created_at` | yes — envelope fields |
| `trace`                               | the inner value is the **CLI contract**; the CLI re-validates it independently |

Because the trace is always emitted to the frozen CLI shape (`extra="forbid"`), schema versioning
never risks the trace's CLI-validity — it only governs the envelope around it.

## Semantic-versioning policy

The version is `MAJOR.MINOR.PATCH`.

| Bump  | Meaning                                                            | Migration step? |
|-------|-------------------------------------------------------------------|-----------------|
| MAJOR | a backward-incompatible envelope change — a field removed/renamed/retyped, or a reshape an old reader can't safely interpret | **required** (mechanically transformable) |
| MINOR | a backward-compatible **additive** change — a new optional envelope field with a default | none |
| PATCH | no field-set change — docs, a relaxed constraint, a derived-value fix | none |

## Reading a capture (upgrade-on-read)

```python
from evalshift import load_capture, load_envelope

upgraded = load_capture(text)        # parse JSON + migrate to current -> dict
envelope = load_envelope(text)       # + reconstruct a typed CaptureEnvelope
```

The chain inside `upgrade_envelope_dict`:

1. **detect** the source version (`detect_version`).
2. **apply the forward-compat policy** (below) — may return as-is with a warning, or raise.
3. **run the migration chain** step by step (each step is a pure `dict -> dict`), then re-stamp
   `schema_version` to the target.

`load_envelope` then reconstructs typed dataclasses (the inverse of `to_jsonable`): unknown event
`type` is a hard error; unknown *fields* are dropped tolerantly (forward-minor compatibility).

## Adding a new schema version (for SDK maintainers)

1. Bump `SCHEMA_VERSION` in `src/evalshift/trace/schema.py` and add it to
   `SUPPORTED_SCHEMA_VERSIONS`.
2. If the **trace** contract changed, re-sync the vendored CLI model
   (`tests/conformance/cli_models_vendored.py`) and the `schema.py` field constants — the drift
   guard in `tests/conformance/test_parity.py` enforces parity.
3. For a **MAJOR** bump, `register_migration("<old>", "<new>", fn)` in `trace/migrate.py` so old
   captures upgrade on read. Minor/patch bumps need no step.
4. Update this doc and add a `docs/DECISIONS.md` note.

## Forward-compatibility (reading a newer capture)

| Source version vs. supported | Behavior                                               |
|------------------------------|--------------------------------------------------------|
| older                        | migrate up the registered chain (`NoMigrationPathError` if a step is missing) |
| same                         | read as-is                                             |
| newer **minor/patch**, same major | **warn** (`logging.getLogger("evalshift")`) and read best-effort; unknown fields dropped |
| newer **major**              | refuse — raise `UnsupportedSchemaVersionError`         |

**Why tolerating newer-minor is safe:** minor bumps are additive-only within a major, and the
version governs only the envelope — the inner trace is an independently CLI-valid `AgentTrace`. An
older reader can therefore read a newer-minor file by ignoring fields it doesn't know. A newer
**major** may have removed or reshaped envelope fields this SDK depends on, so reading it would be
unsound — we refuse loudly.

## Guarantees & limits

- **Raise, don't fail open.** Reading is read-side tooling (the CLI consumes it in Phase 9), not
  the capture hot path. It raises typed `MigrationError` subclasses (`UnreadableCaptureError`,
  `MissingSchemaVersionError`, `InvalidSchemaVersionError`, `UnsupportedSchemaVersionError`,
  `NoMigrationPathError`, `UnknownEventTypeError`) — contrast the capture path's fail-open
  `safety.py` boundary, which must never break the host agent.
- **Pure, forward-only migrations.** Steps are `dict -> dict`, do no I/O, and must not mutate their
  input (`upgrade_envelope_dict` migrates a deep copy). The chain is **linear** — one outgoing step
  per version — and there is **no** downgrade path.
- **No missing-version guessing.** A capture without `schema_version` raises unless a caller opts
  into `default_version=` (there are no real pre-version captures).
- **The CLI re-validates the trace.** This SDK read path stays validation-light (stdlib-only,
  pydantic is dev/test only); CLI-validity is enforced by the conformance tests here and by the
  CLI's own loader downstream.
