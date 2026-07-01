# SDK demo: capture → promote → run → push

End-to-end proof of the EvalShift SDK → CLI → server loop, fully offline (no API keys, no Docker).
A deterministic agent emits captures; the CLI promotes them into a golden suite, runs a migration
eval (source vs target model) against replay fixtures, and pushes the bundle to the hosted server.

The target model (`gemini-3.1-flash-lite-preview`) deliberately **drops `issue_refund`** on the
refund ticket — so the eval surfaces a tool-selection regression.

## Run it

The agent needs the **SDK** venv; the CLI commands need the **CLI** venv (they clash on the
`evalshift` import name, so keep them separate). Set `EVALSHIFT_DIR` so both agree on the data dir.

```bash
cd evalshift-sdk/examples/support_agent
export EVALSHIFT_DIR="$PWD/.evalshift"

# 1. capture (SDK venv) — writes .evalshift/captures/support_demo/cap_*.json
EVALSHIFT_CAPTURE=1 uv run --project ../.. python agent.py

# 2..5. CLI venv (evalshift binary)
evalshift capture list support_demo
evalshift capture promote <cap_id> --as <case> --suite support_demo   # -> .evalshift/suites/support_demo/golden.jsonl
evalshift run --offline --fixtures fixtures.jsonl --suite-name support_demo --yes
evalshift evaluate <run_id> && evalshift analyze <run_id> && evalshift report <run_id>

# 6. push to the hosted server.
#    NOTE: `push` rebuilds the bundle and needs the suite explicitly via --suite
#    (it does not read --suite-name from config like `run` does).
evalshift push <run_id> \
  --suite .evalshift/suites/support_demo/golden.jsonl \
  --host https://evalshift-server.fly.dev --token <es_token> --project <org>/sdk-demo
```

## Files
- `agent.py` — SDK-instrumented deterministic agent (`record_model_call` + `@capture.tool`).
- `prompts.py` / `tools.yaml` — the eval's system prompt + tool catalogue.
- `evalshift.yaml` — eval config; the suite is `source: captured` (produced by promote).
- `fixtures.jsonl` — offline model replays (source passes; target drops the refund tool).
