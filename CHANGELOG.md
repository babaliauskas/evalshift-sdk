# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

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
