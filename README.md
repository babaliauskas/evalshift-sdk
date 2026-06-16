# evalshift-sdk

In-process capture SDK for [EvalShift](https://github.com/babaliauskas/evalshift-cli).

Install it inside your agent process to record what the agent does — model calls, tool calls,
retrievals — and write CLI-valid traces to `.evalshift/captures/`. The `evalshift` CLI reads
those captures from disk; the SDK and CLI never call each other.

- **Distribution:** `evalshift-sdk` · **import name:** `evalshift`
- **Runtime deps:** none (stdlib-only)
- **Python:** >= 3.10
- **Capture is off by default** — set `EVALSHIFT_CAPTURE=1` to record.

```python
from evalshift import capture

@capture.agent(suite="support_agent")   # no-op unless EVALSHIFT_CAPTURE=1
def handle_ticket(query): ...
```

> Status: early scaffold. See `IMPLEMENTATION_PLAN.md` for the phased roadmap and
> `docs/DECISIONS.md` for locked design decisions. The public `capture`/`configure` API
> lands in later phases.
