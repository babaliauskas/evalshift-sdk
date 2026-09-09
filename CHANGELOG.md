# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Trace schema `2.1.0`: `model_call` events carry an optional
  `requested_tool_calls` list of `{name, arguments, call_id}` items — what
  the *model asked to call* in its response, as distinct from `tools_offered`
  (what it was allowed to call) and the `tool_call` events (what the app
  actually ran). `null` means "not recorded", `[]` means "the model requested
  no tools"; see `docs/DECISIONS.md` D-requested.
- `record_model_call(..., requested_tool_calls=[...])` and
  `rec.set_requested_tool_calls([...])` on the `capture.model_call` recorder
  record that list. Both are optional and fail-open: a malformed value is
  dropped with a debug log rather than raised, and each item is normalised to
  exactly `{name, arguments, call_id}` so the capture stays CLI-valid. Unlike
  `tools=`, these arguments are redacted — they are model-generated payload,
  so `requested_tool_calls` joins `input` / `output` in `model_call`'s
  redactable fields.

### Changed

- **Schema `2.0.0` → `2.1.0`** (MINOR, additive). `SUPPORTED_SCHEMA_VERSIONS`
  is now `("2.0.0", "2.1.0")` and a built-in identity migration upgrades a
  2.0.0 capture on read — `requested_tool_calls` stays absent (reading back as
  `None`) rather than being fabricated as `[]`. 1.x captures are still refused
  with `ObsoleteSchemaVersionError`, unchanged.

- Packaging: the EvalShift CLI (`evalshift` 0.14.0+) now depends on this
  package and imports as `evalshift_cli`, so the two install into one
  environment and `pip install evalshift` brings the SDK with it. The
  "separate virtual environments" rule is gone from the README and DOCS.
  No code change; `import evalshift` is, as before, this SDK.

## [0.3.0] - 2026-08-23

First release from the public repository. Compared to 0.2.0 on PyPI:

### Changed

- **Breaking:** `redact=` is now a required keyword on every capture entry
  point. Pass `True` for the built-in `default_redactor`, `False` to capture
  verbatim, or a custom callable (the new public `RedactSetting` type).
  Masking is an explicit choice, never a default.
- **Breaking:** trace schema is now `2.0.0` (adds `toolset_ref` /
  `tools_offered` on `model_call` spans). No migration from 1.x is registered:
  loading a 1.x capture raises `ObsoleteSchemaVersionError` rather than
  silently asserting it ran with no tools offered. Re-capture to upgrade.
- `generation_config` values are sanitized through a shared allow-list with
  JSON coercion, so a non-serializable config value can no longer drop the
  whole capture at sink-write time, and un-allow-listed keys (e.g.
  `system_instruction`) never reach the capture.

### Added

- Toolset capture: the tools an agent was offered on each model call are
  normalized from Anthropic, OpenAI, or Gemini shapes into one canonical
  form, fingerprinted, and written once as a content-addressed sidecar
  (`.evalshift/toolsets/<hex>.json`) referenced by `toolset_ref`.
- `RedactSetting` exported from the package root.
