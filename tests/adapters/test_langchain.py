"""Phase 7 — the LangChain adapter callback handler.

Most tests drive ``EvalShiftCallbackHandler`` by calling its callback methods directly with the
kwargs LangChain passes (synthetic, deterministic, no langchain needed at run time). The final
test runs a real ``langchain_core`` chain through the handler (guarded by ``importorskip``) to
prove the real callback signatures match.

The written capture is validated through the vendored pydantic ``AgentTrace`` — the CLI contract —
exactly as the manual-instrumentation conformance tests do.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest

from evalshift import configure
from evalshift.adapters.langchain import EvalShiftCallbackHandler
from tests.conformance.cli_models_vendored import AgentTrace
from tests.conftest import CaptureReader


def _llm_result(text: str, *, input_tokens: int = 0, output_tokens: int = 0) -> SimpleNamespace:
    """A duck-typed ``LLMResult`` carrying one generation + token usage (no langchain import)."""
    message = SimpleNamespace(
        usage_metadata={"input_tokens": input_tokens, "output_tokens": output_tokens}
    )
    generation = SimpleNamespace(text=text, message=message)
    return SimpleNamespace(generations=[[generation]], llm_output=None)


def _events(cap: dict[str, Any]) -> list[dict[str, Any]]:
    return list(cap["trace"]["events"])


def _of_type(cap: dict[str, Any], type_: str) -> list[dict[str, Any]]:
    return [e for e in _events(cap) if e["type"] == type_]


def test_chain_with_tool_and_model_call_is_schema_valid(
    capturing: Path, read_captures: CaptureReader
) -> None:
    handler = EvalShiftCallbackHandler(suite="lc_agent")
    root, tool, mc = uuid4(), uuid4(), uuid4()

    handler.on_chain_start({"name": "chain"}, {"query": "hi"}, run_id=root, parent_run_id=None)
    handler.on_tool_start(
        {"name": "search"}, "hi", run_id=tool, parent_run_id=root, inputs={"q": "hi"}
    )
    handler.on_tool_end({"hits": 1}, run_id=tool, parent_run_id=root)
    handler.on_llm_start({"name": "llm"}, ["answer this"], run_id=mc, parent_run_id=root)
    handler.on_llm_end(
        _llm_result("answer", input_tokens=12, output_tokens=3), run_id=mc, parent_run_id=root
    )
    handler.on_chain_end({"output": "answer"}, run_id=root, parent_run_id=None)

    cap = read_captures("lc_agent")[0]
    AgentTrace.model_validate(cap["trace"])  # raises if not CLI-valid
    assert len(_of_type(cap, "tool_call")) == 1
    assert len(_of_type(cap, "tool_result")) == 1
    assert len(_of_type(cap, "model_call")) == 1
    assert _of_type(cap, "tool_call")[0]["name"] == "search"
    assert _of_type(cap, "model_call")[0]["model_id"] == "llm"
    assert _of_type(cap, "model_call")[0]["input_tokens"] == 12
    assert cap["suite"] == "lc_agent"


def test_nested_tool_parentage(capturing: Path, read_captures: CaptureReader) -> None:
    handler = EvalShiftCallbackHandler(suite="lc_nest")
    root, tool_a, mc, tool_b = uuid4(), uuid4(), uuid4(), uuid4()

    handler.on_chain_start({"name": "c"}, {"q": "x"}, run_id=root, parent_run_id=None)
    handler.on_tool_start({"name": "tool_a"}, "x", run_id=tool_a, parent_run_id=root)
    # model call nested *inside* tool_a
    handler.on_llm_start({"name": "llm"}, ["p"], run_id=mc, parent_run_id=tool_a)
    handler.on_llm_end(_llm_result("o"), run_id=mc, parent_run_id=tool_a)
    handler.on_tool_end("done", run_id=tool_a, parent_run_id=root)
    # sibling tool_b at top level
    handler.on_tool_start({"name": "tool_b"}, "y", run_id=tool_b, parent_run_id=root)
    handler.on_tool_end("done", run_id=tool_b, parent_run_id=root)
    handler.on_chain_end({"output": "ok"}, run_id=root, parent_run_id=None)

    cap = read_captures("lc_nest")[0]
    AgentTrace.model_validate(cap["trace"])
    call_a = next(e for e in _of_type(cap, "tool_call") if e["name"] == "tool_a")
    call_b = next(e for e in _of_type(cap, "tool_call") if e["name"] == "tool_b")
    model = _of_type(cap, "model_call")[0]
    # model call's parent is tool_a's call_id (carried in the evalshift meta block)
    assert model["metadata"]["evalshift"]["parent_call_id"] == call_a["call_id"]
    # sibling tool at top level has no tool parent
    assert call_b["parent_call_id"] is None


def test_tool_error_is_captured(capturing: Path, read_captures: CaptureReader) -> None:
    handler = EvalShiftCallbackHandler(suite="lc_err")
    root, tool = uuid4(), uuid4()

    handler.on_chain_start({"name": "c"}, {"q": "x"}, run_id=root, parent_run_id=None)
    handler.on_tool_start({"name": "boom"}, "x", run_id=tool, parent_run_id=root)
    handler.on_tool_error(ValueError("kaboom"), run_id=tool, parent_run_id=root)
    handler.on_chain_end({"output": ""}, run_id=root, parent_run_id=None)

    cap = read_captures("lc_err")[0]
    AgentTrace.model_validate(cap["trace"])
    result = _of_type(cap, "tool_result")[0]
    assert result["error"] is not None and "kaboom" in result["error"]


def test_chain_error_records_error_event(capturing: Path, read_captures: CaptureReader) -> None:
    handler = EvalShiftCallbackHandler(suite="lc_chain_err")
    root = uuid4()

    handler.on_chain_start({"name": "c"}, {"q": "x"}, run_id=root, parent_run_id=None)
    handler.on_chain_error(RuntimeError("chain blew up"), run_id=root, parent_run_id=None)

    cap = read_captures("lc_chain_err")[0]
    trace = AgentTrace.model_validate(cap["trace"])
    assert any(e.type == "error" for e in trace.events)


def test_gate_off_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, read_captures: CaptureReader
) -> None:
    # EVALSHIFT_DIR set but EVALSHIFT_CAPTURE unset -> gate off, no files.
    monkeypatch.setenv("EVALSHIFT_DIR", str(tmp_path))
    monkeypatch.delenv("EVALSHIFT_CAPTURE", raising=False)
    handler = EvalShiftCallbackHandler(suite="lc_off")
    root, tool = uuid4(), uuid4()
    handler.on_chain_start({"name": "c"}, {"q": "x"}, run_id=root, parent_run_id=None)
    handler.on_tool_start({"name": "t"}, "x", run_id=tool, parent_run_id=root)
    handler.on_tool_end("ok", run_id=tool, parent_run_id=root)
    handler.on_chain_end({"output": "ok"}, run_id=root, parent_run_id=None)
    assert read_captures("lc_off") == []


def test_sampling_zero_writes_nothing(capturing: Path, read_captures: CaptureReader) -> None:
    configure(sample_rate=0.0)
    handler = EvalShiftCallbackHandler(suite="lc_sample")
    root = uuid4()
    handler.on_chain_start({"name": "c"}, {"q": "x"}, run_id=root, parent_run_id=None)
    handler.on_chain_end({"output": "ok"}, run_id=root, parent_run_id=None)
    assert read_captures("lc_sample") == []


def test_sampling_full_writes_one(capturing: Path, read_captures: CaptureReader) -> None:
    configure(sample_rate=1.0)
    handler = EvalShiftCallbackHandler(suite="lc_sample")
    root = uuid4()
    handler.on_chain_start({"name": "c"}, {"q": "x"}, run_id=root, parent_run_id=None)
    handler.on_chain_end({"output": "ok"}, run_id=root, parent_run_id=None)
    assert len(read_captures("lc_sample")) == 1


def test_redactor_fault_drops_capture_without_raising(
    capturing: Path, read_captures: CaptureReader
) -> None:
    def _boom(value: Any) -> Any:
        raise RuntimeError("redactor down")

    handler = EvalShiftCallbackHandler(suite="lc_redact", redact=_boom)
    root = uuid4()
    handler.on_chain_start({"name": "c"}, {"q": "x"}, run_id=root, parent_run_id=None)
    # The terminal callback must not raise even though redaction fails inside finalize.
    handler.on_chain_end({"output": "ok"}, run_id=root, parent_run_id=None)
    assert read_captures("lc_redact") == []  # fail-closed: dropped, not written unredacted


def test_malformed_callback_never_raises(capturing: Path) -> None:
    handler = EvalShiftCallbackHandler(suite="lc_safe")
    root, tool = uuid4(), uuid4()
    handler.on_chain_start({"name": "c"}, {"q": "x"}, run_id=root, parent_run_id=None)
    # serialized=None would blow up an unguarded .get(...); must be swallowed.
    handler.on_tool_start(None, "x", run_id=tool, parent_run_id=root)
    handler.on_tool_end("ok", run_id=tool, parent_run_id=root)
    # an end for an unknown run id must be a no-op, not a KeyError
    handler.on_tool_end("ok", run_id=uuid4(), parent_run_id=root)
    handler.on_chain_end({"output": "ok"}, run_id=root, parent_run_id=None)


def test_provenance_metadata(capturing: Path, read_captures: CaptureReader) -> None:
    handler = EvalShiftCallbackHandler(suite="lc_prov")
    root, tool = uuid4(), uuid4()
    handler.on_chain_start({"name": "c"}, {"q": "x"}, run_id=root, parent_run_id=None)
    handler.on_tool_start({"name": "t"}, "x", run_id=tool, parent_run_id=root)
    handler.on_tool_end("ok", run_id=tool, parent_run_id=root)
    handler.on_chain_end({"output": "ok"}, run_id=root, parent_run_id=None)

    cap = read_captures("lc_prov")[0]
    call = _of_type(cap, "tool_call")[0]
    assert call["metadata"]["adapter"] == "langchain"
    assert call["metadata"]["lc_run_id"] == str(tool)
    # the serializer's own namespace must survive alongside the provenance keys
    assert "evalshift" in call["metadata"]


def test_handler_reuse_across_invocations(capturing: Path, read_captures: CaptureReader) -> None:
    handler = EvalShiftCallbackHandler(suite="lc_reuse")
    for q in ("a", "b"):
        root = uuid4()
        handler.on_chain_start({"name": "c"}, {"q": q}, run_id=root, parent_run_id=None)
        handler.on_chain_end({"output": q}, run_id=root, parent_run_id=None)
    caps = read_captures("lc_reuse")
    assert len(caps) == 2
    ids = {c["capture_id"] for c in caps}
    assert len(ids) == 2  # distinct captures, no state bleed


def test_real_langchain_chain_smoke(capturing: Path, read_captures: CaptureReader) -> None:
    pytest.importorskip("langchain_core")
    from langchain_core.runnables import RunnableLambda
    from langchain_core.tools import tool

    @tool
    def shout(text: str) -> str:
        """Uppercase the text."""
        return text.upper()

    def run(inputs: dict[str, Any]) -> str:
        return str(shout.invoke(inputs["text"]))

    chain = RunnableLambda(run)
    handler = EvalShiftCallbackHandler(suite="lc_smoke")
    result = chain.invoke({"text": "hi"}, config={"callbacks": [handler]})
    assert result == "HI"

    caps = read_captures("lc_smoke")
    assert len(caps) == 1
    AgentTrace.model_validate(caps[0]["trace"])
    assert any(e["name"] == "shout" for e in _of_type(caps[0], "tool_call"))
