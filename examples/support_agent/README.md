# SDK demo: capture → sync → run → push

End-to-end proof of the EvalShift SDK → CLI → server loop. A deterministic agent (`agent.py`,
no real LLM) emits schema-valid captures; `evalshift capture sync` promotes them into a golden
suite and wires it into `evalshift.yaml`; `evalshift run` then replays those cases against two
models and scores the candidate's tool selection against what the capture recorded.

**About API keys.** Step 1 (capture) needs none — `agent.py` calls no model. Steps 3–5 do:
there is **no `--offline` mode and no `--fixtures` flag** on `evalshift run`; it always calls the
real provider. This suite is two examples × two models = four calls, which is cents at
flash-tier pricing. See [Running it without API keys](#running-it-without-api-keys) below if you
want the pipeline without a key.

## Run it

The agent runs against this checkout's SDK (`uv run --project ../..`); the `evalshift` binary
comes from the CLI (`pip install evalshift`, which depends on the released SDK — one
environment can hold both; the split here only keeps the example on the source tree). Set
`EVALSHIFT_DIR` so the agent and the CLI agree on the data dir.

```bash
cd evalshift-sdk/examples/support_agent
export EVALSHIFT_DIR="$PWD/.evalshift"

# 1. capture (this checkout's SDK). Writes .evalshift/captures/support_demo/cap_*.json plus the
#    content-addressed toolset sidecar .evalshift/toolsets/<sha256>.json.
EVALSHIFT_CAPTURE=1 uv run --project ../.. python agent.py

# --- everything below uses the `evalshift` binary from the CLI ---

evalshift capture list support_demo

# 2. promote every capture into .evalshift/suites/support_demo/golden.jsonl and rewrite the
#    managed `suites:` region of evalshift.yaml with the evaluators these captures justify
#    (tool_selection + tool_arguments). Re-runnable; add --force to re-promote.
evalshift capture sync --suite support_demo

# sanity check before spending anything
evalshift validate --suite .evalshift/suites/support_demo/golden.jsonl
evalshift doctor

# 3. run. Needs a real key for the models in evalshift.yaml (GEMINI_API_KEY as shipped; swap
#    defaults.source_model / target_model for a provider you do have). `doctor` above names
#    exactly which key is missing. This is the step that costs money.
export GEMINI_API_KEY=...
evalshift run --suite-name support_demo --yes

# 4. score, analyse, report
evalshift evaluate <run_id>
evalshift analyze <run_id>
evalshift report <run_id>            # --no-insights skips the extra narrative LLM call

# 5. push to the hosted server. The suite comes from the run's own state.json, so no
#    --suite is needed; pass --suite-name support_demo if you want to be explicit.
evalshift push <run_id> \
  --host https://api.evalshift.dev --token <es_token> --project <org>/sdk-demo
```

`evalshift all --suite-name support_demo --to <candidate>` collapses steps 3–4 into one command.

## What this actually measures

Each promoted case pins the tool calls the recorded run made (`expected_tools`) and a
`toolset_ref` naming the toolset the agent's model call was offered. Both models are dispatched
with **that** toolset — resolved from the `.evalshift/toolsets/` sidecar the SDK wrote, not from
any file in this directory — and `tool_selection` / `tool_arguments` score each side against the
capture. A candidate that skips `issue_refund` on the refund ticket shows up as a negative delta
on `routing`.

Everything under `.evalshift/` is generated and gitignored; delete it to start over.

## Running it without API keys

There is no keyless path through the CLI itself. The only mocked harness is the CLI's own test
double, `ReplayClient` in `evalshift-cli/tests/integration/replay_client.py`, which is injected
straight into the orchestrator:

```python
run_orchestrator(..., client=ReplayClient(fixtures_path))
```

Using it outside the test suite is possible but unsupported: it lives in the CLI's test tree
(shipped in the sdist, **not** in the published wheel), so it needs a CLI *source checkout* on
`sys.path` plus a short driver script that loads the config and suite and calls
`run_orchestrator` directly. `evalshift-cli/tests/integration/test_conversation_pipeline.py` is
the worked example, including the JSONL fixture format. Once the run directory exists, the
downstream `evalshift evaluate` / `analyze` / `report` commands work normally.

## Known gap: tool-result fixtures are captured but not replayed

The SDK records every tool's *result* on its `tool_result` event together with the tool's
`call_id` and a `metadata.evalshift.input_hash`, and `build_fixture_table`
(`src/evalshift/trace/serialize.py`) derives the `(call_id, input_hash) -> result` lookup that
`docs/DECISIONS.md` §1 promises to CLI replay.

**Nothing in the CLI reads it yet.** `evalshift run` makes one model call per example and scores
the first tool-emitting round only, so a multi-round agent's later rounds are captured but never
replayed — which is also why `capture sync` defaults to `--rounds first`. Teacher-forced
multi-round replay is Phase 2 of
`evalshift-cli/docs/superpowers/plans/2026-09-08-external-review-response.md`.

## Files
- `agent.py` — SDK-instrumented deterministic agent (`record_model_call` + `@capture.tool`).
- `prompts.py` — the eval's system prompt, read by `evalshift.yaml` via `detection: python_string`.
- `evalshift.yaml` — eval config. The `suites:` region between the `>>> evalshift suites` markers
  is owned by `evalshift capture sync`; hand edits inside it are overwritten.
