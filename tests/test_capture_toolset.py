"""Per-call toolset capture: ``tools=`` is required at every capture point (D-toolset).

Mirrors ``tests/test_redaction_required.py`` (D-4c) for the "required keyword" shape, and
``tests/test_capture_generation_config.py`` for the "lands on the event" shape. Covers the six
entry points named in the implementation plan: ``record_model_call``, ``capture.model_call``,
``capture.agent``, ``capture.agent_session``, ``capture.agent_session_async``, and
``EvalShiftCallbackHandler`` (the LangChain-specific cases live in
``tests/adapters/test_langchain.py`` alongside its other constructor-requirement tests).
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

import pytest

from evalshift import FileSink, MemorySink, capture, configure, record_model_call
from evalshift.capture.toolset import fingerprint_tools, normalize_tools
from tests.conftest import CaptureReader

SEARCH_TOOL: dict[str, Any] = {
    "name": "search_orders",
    "description": "Look up a customer's orders.",
    "input_schema": {
        "type": "object",
        "properties": {"customer_id": {"type": "string"}},
        "required": ["customer_id"],
    },
}
REFUND_TOOL: dict[str, Any] = {
    "name": "issue_refund",
    "description": "Issue a refund on an order.",
    "input_schema": {
        "type": "object",
        "properties": {"order_id": {"type": "string"}},
        "required": ["order_id"],
    },
}
TWO_TOOLS = [SEARCH_TOOL, REFUND_TOOL]


def _model_events(capture_dict: dict[str, Any]) -> list[dict[str, Any]]:
    return [e for e in capture_dict["trace"]["events"] if e["type"] == "model_call"]


def _sidecar_path(base: Path, fingerprint: str) -> Path:
    return base / "toolsets" / f"{fingerprint.removeprefix('sha256:')}.json"


# --- tools= is required: TypeError without it, at every entry point -------------------------


def test_record_model_call_requires_tools() -> None:
    with pytest.raises(TypeError):
        record_model_call(model_id="m")  # type: ignore[call-arg]


def test_capture_model_call_requires_tools() -> None:
    with pytest.raises(TypeError):
        capture.model_call(model_id="m")  # type: ignore[call-arg]


def test_agent_decorator_requires_tools() -> None:
    with pytest.raises(TypeError):
        capture.agent(suite="req", redact=False)  # type: ignore[call-arg]


def test_agent_session_requires_tools() -> None:
    with pytest.raises(TypeError), capture.agent_session(suite="req", redact=False):  # type: ignore[call-arg]
        pass


async def test_agent_session_async_requires_tools() -> None:
    with pytest.raises(TypeError):
        async with capture.agent_session_async(suite="req", redact=False):  # type: ignore[call-arg]
            pass


def test_agent_decorator_requires_tools_even_when_gate_is_off() -> None:
    """Gate-independent, exactly like D-4c's ``redact=``: a missing required kwarg is a
    programming error in every environment, capture on or off."""
    with pytest.raises(TypeError):
        capture.agent(suite="req", redact=False)  # type: ignore[call-arg]


def test_agent_decorator_raises_at_decoration_time_not_call_time() -> None:
    """Forgetting ``tools=`` must fail loudly where the decorator is written, not deep inside a
    request handler the first time the decorated function is actually invoked."""
    with pytest.raises(TypeError):

        @capture.agent(suite="req", redact=False)  # type: ignore[call-arg]
        def _agent() -> None:  # pragma: no cover - never reached, decoration itself raises
            raise AssertionError("must not be called")


# --- round trip: record_model_call --------------------------------------------------------


def test_record_model_call_stamps_tools_offered_and_toolset_ref(
    capturing: Path, read_captures: CaptureReader
) -> None:
    @capture.agent(suite="s", redact=False, tools=[])
    def agent() -> str:
        record_model_call(model_id="m", tools=TWO_TOOLS, input="hi", output="ok")
        return "ok"

    agent()
    [envelope] = read_captures("s")
    [mc] = _model_events(envelope)
    assert mc["tools_offered"] == ["search_orders", "issue_refund"]
    normalized = normalize_tools(TWO_TOOLS)
    assert normalized is not None
    assert mc["toolset_ref"] == fingerprint_tools(normalized)


def test_record_model_call_toolset_ref_sidecar_is_written_to_disk(
    capturing: Path, read_captures: CaptureReader
) -> None:
    @capture.agent(suite="s", redact=False, tools=[])
    def agent() -> str:
        record_model_call(model_id="m", tools=TWO_TOOLS, output="ok")
        return "ok"

    agent()
    [envelope] = read_captures("s")
    [mc] = _model_events(envelope)
    normalized = normalize_tools(TWO_TOOLS)
    assert normalized is not None
    fingerprint = fingerprint_tools(normalized)
    assert mc["toolset_ref"] == fingerprint
    sidecar = _sidecar_path(capturing, fingerprint)
    assert sidecar.exists()
    assert json.loads(sidecar.read_text(encoding="utf-8")) == {
        "schema_version": "1.0.0",
        "fingerprint": fingerprint,
        "tools": normalized,
    }


def test_record_model_call_without_tools_call_writes_no_toolset_fields_by_default(
    capturing: Path, read_captures: CaptureReader
) -> None:
    """Explicit empty own-call assertion (``tools=[]``): a real, valid toolset, so the fields
    still stamp -- an empty *list* of tool names, not an absent one."""

    @capture.agent(suite="s", redact=False, tools=[])
    def agent() -> str:
        record_model_call(model_id="m", tools=[], output="ok")
        return "ok"

    agent()
    [envelope] = read_captures("s")
    [mc] = _model_events(envelope)
    assert mc["tools_offered"] == []
    assert mc["toolset_ref"] == fingerprint_tools([])


# --- round trip: capture.model_call (streaming) --------------------------------------------


def test_streaming_model_call_stamps_tools_offered_and_toolset_ref(
    capturing: Path, read_captures: CaptureReader
) -> None:
    @capture.agent(suite="s", redact=False, tools=[])
    def agent() -> str:
        with capture.model_call(model_id="m", tools=TWO_TOOLS, input="hi") as rec:
            rec.add_text("ok")
        return "ok"

    agent()
    [envelope] = read_captures("s")
    [mc] = _model_events(envelope)
    assert mc["tools_offered"] == ["search_orders", "issue_refund"]
    normalized = normalize_tools(TWO_TOOLS)
    assert normalized is not None
    assert mc["toolset_ref"] == fingerprint_tools(normalized)


# --- normalize_tools returning None: neither field stamped, capture still written ----------


def test_unrecognised_own_tools_leaves_both_fields_unstamped(
    capturing: Path, read_captures: CaptureReader
) -> None:
    """A caller passed *something* for tools, but it matches no recognised shape --
    normalize_tools returns None -- so the capture is left structurally invalid for this event
    (neither field stamped) rather than guessing. The capture is still written: this is a data
    problem, not a host-agent-breaking one (fail-open still applies to bookkeeping)."""

    @capture.agent(suite="s", redact=False, tools=[])
    def agent() -> str:
        record_model_call(model_id="m", tools="not-a-toolset", output="ok")
        return "ok"

    agent()
    [envelope] = read_captures("s")
    [mc] = _model_events(envelope)
    assert mc["toolset_ref"] is None
    assert mc["tools_offered"] is None


def test_unrecognised_own_tools_does_not_fall_back_to_session(
    capturing: Path, read_captures: CaptureReader
) -> None:
    """Own-call authority cuts both ways: a call that explicitly (if badly) asserts its own
    tools must not silently borrow the session's -- only a bare ``None`` defers."""

    @capture.agent(suite="s", redact=False, tools=TWO_TOOLS)
    def agent() -> str:
        record_model_call(model_id="m", tools=42, output="ok")
        return "ok"

    agent()
    [envelope] = read_captures("s")
    [mc] = _model_events(envelope)
    assert mc["toolset_ref"] is None
    assert mc["tools_offered"] is None


def test_unrecognised_tools_logs_at_debug(
    capturing: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """``_normalize_toolset`` is the single funnel both the per-call and the session-establishing
    paths normalise through (see its docstring), so one assertion here covers every entry point:
    a rejected value must be diagnosable from the log without reading source. Mirrors
    ``tests/test_safety.py``'s ``caplog`` pattern for ``fail_open``/``guard``.

    Two independent normalisations happen inside this one ``agent()`` call: the session's own
    ``tools="not-a-session-toolset"`` (rejected when ``agent()`` runs, not at decoration time --
    see ``test_agent_decorator_raises_at_decoration_time_not_call_time`` for why decoration itself
    cannot normalise), and the call's own ``tools=42``. Both must log."""

    @capture.agent(suite="s", redact=False, tools="not-a-session-toolset")
    def agent() -> str:
        record_model_call(model_id="m", tools=42, output="ok")
        return "ok"

    with caplog.at_level(logging.DEBUG, logger="evalshift"):
        agent()

    toolset_records = [r for r in caplog.records if "recognised toolset shape" in r.message]
    assert len(toolset_records) == 2
    # Each message names the shape that was rejected, not just "something went wrong" --
    # useful enough to diagnose a rejected capture without reading source.
    assert any("str" in r.message for r in toolset_records)  # the session's "not-a-..." string
    assert any("int" in r.message for r in toolset_records)  # the call's 42


def test_unrecognised_list_of_tools_logs_its_length_at_debug(
    capturing: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A list value gets a more specific log shape (``list[N]``) than a bare type name -- a
    caller who passed three tools and got one wrong benefits from seeing "list[2]", not "list"."""

    @capture.agent(suite="s", redact=False, tools=[])
    def agent() -> str:
        record_model_call(model_id="m", tools=[SEARCH_TOOL, "garbage"], output="ok")
        return "ok"

    with caplog.at_level(logging.DEBUG, logger="evalshift"):
        agent()

    assert any(
        "recognised toolset shape" in r.message and "list[2]" in r.message for r in caplog.records
    )


def test_unrecognised_tuple_of_tools_logs_its_length_at_debug(
    capturing: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """F4: a tuple gets the same actionable ``tuple[N]`` shape in the log as a list gets
    ``list[N]`` -- not just the bare type name ``"tuple"``."""

    @capture.agent(suite="s", redact=False, tools=[])
    def agent() -> str:
        record_model_call(model_id="m", tools=(SEARCH_TOOL, "garbage"), output="ok")
        return "ok"

    with caplog.at_level(logging.DEBUG, logger="evalshift"):
        agent()

    assert any(
        "recognised toolset shape" in r.message and "tuple[2]" in r.message for r in caplog.records
    )


def test_partially_unrecognised_tool_list_leaves_both_fields_unstamped(
    capturing: Path, read_captures: CaptureReader
) -> None:
    @capture.agent(suite="s", redact=False, tools=[])
    def agent() -> str:
        record_model_call(model_id="m", tools=[SEARCH_TOOL, "garbage"], output="ok")
        return "ok"

    agent()
    [envelope] = read_captures("s")
    [mc] = _model_events(envelope)
    assert mc["toolset_ref"] is None
    assert mc["tools_offered"] is None


# --- session inheritance: tools=None defers to the enclosing session -----------------------


def test_record_model_call_inherits_session_toolset_via_agent_session(
    capturing: Path, read_captures: CaptureReader
) -> None:
    with capture.agent_session(suite="s", redact=False, tools=TWO_TOOLS, agent_input="hi"):
        record_model_call(model_id="m", tools=None, output="ok")

    [envelope] = read_captures("s")
    [mc] = _model_events(envelope)
    assert mc["tools_offered"] == ["search_orders", "issue_refund"]
    normalized = normalize_tools(TWO_TOOLS)
    assert normalized is not None
    assert mc["toolset_ref"] == fingerprint_tools(normalized)


async def test_record_model_call_inherits_session_toolset_via_agent_session_async(
    capturing: Path, read_captures: CaptureReader
) -> None:
    async with capture.agent_session_async(
        suite="s", redact=False, tools=TWO_TOOLS, agent_input="hi"
    ):
        record_model_call(model_id="m", tools=None, output="ok")

    [envelope] = read_captures("s")
    [mc] = _model_events(envelope)
    assert mc["tools_offered"] == ["search_orders", "issue_refund"]


def test_record_model_call_inherits_session_toolset_via_agent_decorator(
    capturing: Path, read_captures: CaptureReader
) -> None:
    @capture.agent(suite="s", redact=False, tools=TWO_TOOLS)
    def agent() -> str:
        record_model_call(model_id="m", tools=None, output="ok")
        return "ok"

    agent()
    [envelope] = read_captures("s")
    [mc] = _model_events(envelope)
    assert mc["tools_offered"] == ["search_orders", "issue_refund"]


def test_streaming_model_call_inherits_session_toolset(
    capturing: Path, read_captures: CaptureReader
) -> None:
    @capture.agent(suite="s", redact=False, tools=TWO_TOOLS)
    def agent() -> str:
        with capture.model_call(model_id="m", tools=None, input="hi") as rec:
            rec.add_text("ok")
        return "ok"

    agent()
    [envelope] = read_captures("s")
    [mc] = _model_events(envelope)
    assert mc["tools_offered"] == ["search_orders", "issue_refund"]


def test_empty_session_toolset_is_inherited_as_a_real_empty_list(
    capturing: Path, read_captures: CaptureReader
) -> None:
    """The session itself asserted "no tools" (``tools=[]``, not the *absence* of a session
    toolset) -- an inheriting call must see a real empty list, not an unstamped pair of fields.
    """

    @capture.agent(suite="s", redact=False, tools=[])
    def agent() -> str:
        record_model_call(model_id="m", tools=None, output="ok")
        return "ok"

    agent()
    [envelope] = read_captures("s")
    [mc] = _model_events(envelope)
    assert mc["tools_offered"] == []
    assert mc["toolset_ref"] == fingerprint_tools([])


def test_record_model_call_tools_none_without_any_session_toolset_leaves_fields_unstamped(
    capturing: Path, read_captures: CaptureReader
) -> None:
    """``tools=None`` inherits nothing when there is nothing to inherit -- this must degrade the
    same as any other unrecognised value, not crash."""

    @capture.agent(suite="s", redact=False, tools="not-a-toolset")
    def agent() -> str:
        record_model_call(model_id="m", tools=None, output="ok")
        return "ok"

    agent()
    [envelope] = read_captures("s")
    [mc] = _model_events(envelope)
    assert mc["toolset_ref"] is None
    assert mc["tools_offered"] is None


# --- per-call authority: a call's own tools overrides the session's ------------------------


def test_record_model_call_own_tools_overrides_session_toolset(
    capturing: Path, read_captures: CaptureReader
) -> None:
    """The switching case the whole plan exists for: one agent, one session, two calls, two
    different toolsets chosen at runtime -- the per-call value always wins."""

    @capture.agent(suite="s", redact=False, tools=[SEARCH_TOOL])
    def agent() -> str:
        record_model_call(model_id="m", tools=[REFUND_TOOL], output="ok")
        return "ok"

    agent()
    [envelope] = read_captures("s")
    [mc] = _model_events(envelope)
    assert mc["tools_offered"] == ["issue_refund"]  # the call's own, not the session's


def test_two_calls_in_one_session_each_keep_their_own_toolset(
    capturing: Path, read_captures: CaptureReader
) -> None:
    @capture.agent(suite="s", redact=False, tools=[SEARCH_TOOL])
    def agent() -> str:
        record_model_call(model_id="m1", tools=None, output="first")  # inherits [SEARCH_TOOL]
        record_model_call(model_id="m2", tools=[REFUND_TOOL], output="second")  # own, overrides
        record_model_call(model_id="m3", tools=None, output="third")  # inherits again
        return "ok"

    agent()
    [envelope] = read_captures("s")
    calls = _model_events(envelope)
    assert [c["tools_offered"] for c in calls] == [
        ["search_orders"],
        ["issue_refund"],
        ["search_orders"],
    ]


def test_streaming_model_call_own_tools_overrides_session_toolset(
    capturing: Path, read_captures: CaptureReader
) -> None:
    @capture.agent(suite="s", redact=False, tools=[SEARCH_TOOL])
    def agent() -> str:
        with capture.model_call(model_id="m", tools=[REFUND_TOOL], input="hi") as rec:
            rec.add_text("ok")
        return "ok"

    agent()
    [envelope] = read_captures("s")
    [mc] = _model_events(envelope)
    assert mc["tools_offered"] == ["issue_refund"]


# --- redaction never touches toolset fields (they live in span.data, but outside the
#     model_call allow-list in redaction/base.py's _REDACTABLE_FIELDS) ----------------------


def test_toolset_fields_survive_redaction(capturing: Path, read_captures: CaptureReader) -> None:
    email_flavoured_tool = {
        "name": "contact_support",
        "description": "Email admin@corp.com if the customer is upset.",
        "input_schema": {"type": "object"},
    }

    @capture.agent(suite="s", redact=True, tools=[])
    def agent() -> str:
        record_model_call(model_id="m", tools=[email_flavoured_tool], output="ok")
        return "ok"

    agent()
    [envelope] = read_captures("s")
    [mc] = _model_events(envelope)
    # tools_offered is names-only, so the description never reaches this field regardless --
    # the real assertion is that stamping happened at all under redact=True.
    assert mc["tools_offered"] == ["contact_support"]
    assert mc["toolset_ref"] is not None
    # And the sidecar itself -- a different file entirely -- keeps the description verbatim,
    # proving the redactor never walked it (ToolsetSink has no redaction hook, by design).
    sidecar = _sidecar_path(capturing, mc["toolset_ref"])
    tools = json.loads(sidecar.read_text(encoding="utf-8"))["tools"]
    assert "admin@corp.com" in tools[0]["description"]


# --- outside any session: no crash, no fields (mirrors record_model_call's existing no-op) --


def test_record_model_call_outside_a_session_is_a_silent_noop(capturing: Path) -> None:
    record_model_call(model_id="m", tools=TWO_TOOLS)  # no active session -> must not raise


def test_streaming_model_call_outside_a_session_is_inert(capturing: Path) -> None:
    rec = capture.model_call(model_id="m", tools=TWO_TOOLS)
    with rec:  # no active session -> inert, no span, no write, no crash
        rec.add_text("ok")


# --- F1: a raising fingerprint_tools must degrade like an unrecognised shape, never drop the
#     whole event. A normalised toolset's input_schema is arbitrary user JSON passed through
#     unvalidated (D-toolset: "no allow-list"); a nested value json.dumps cannot serialise (a
#     set, here) makes fingerprint_tools raise TypeError. That must land exactly like
#     normalize_tools already returning None: neither field stamped, capture still written. ----

RAISING_SCHEMA_TOOL: dict[str, Any] = {
    "name": "bad_tool",
    "description": "desc",
    # A set nested inside a dict passes normalize_tools/_coerce_schema untouched (only the
    # top-level shape is coerced) but is not JSON-serialisable -- json.dumps inside
    # fingerprint_tools raises TypeError on it.
    "input_schema": {"type": "object", "properties": {"weird": {1, 2, 3}}},
}


def test_record_model_call_raising_fingerprint_leaves_event_and_fields_intact(
    capturing: Path, read_captures: CaptureReader
) -> None:
    @capture.agent(suite="s", redact=False, tools=[])
    def agent() -> str:
        record_model_call(
            model_id="m",
            tools=[RAISING_SCHEMA_TOOL],
            input="hi",
            output="ok",
            input_tokens=5,
            output_tokens=7,
        )
        return "ok"

    agent()
    [envelope] = read_captures("s")
    [mc] = _model_events(envelope)
    # Degrades exactly like an unrecognised shape: neither toolset field stamped...
    assert mc["toolset_ref"] is None
    assert mc["tools_offered"] is None
    # ...but the event itself -- and its unrelated data -- survives intact (this is the bug:
    # pre-fix, the whole event was dropped, "events": [] on the written capture).
    assert mc["input"] == "hi"
    assert mc["output"] == "ok"
    assert mc["input_tokens"] == 5
    assert mc["output_tokens"] == 7


def test_streaming_model_call_raising_fingerprint_leaves_event_intact(
    capturing: Path, read_captures: CaptureReader
) -> None:
    """Same degrade, at the other of the two unguarded call sites (_ModelCallRecorder._open)."""

    @capture.agent(suite="s", redact=False, tools=[])
    def agent() -> str:
        with capture.model_call(model_id="m", tools=[RAISING_SCHEMA_TOOL], input="hi") as rec:
            rec.add_text("streamed")
        return "ok"

    agent()
    [envelope] = read_captures("s")
    [mc] = _model_events(envelope)
    assert mc["toolset_ref"] is None
    assert mc["tools_offered"] is None
    assert mc["input"] == "hi"
    assert mc["output"] == "streamed"


def test_session_toolset_raising_fingerprint_leaves_session_toolset_unset(
    capturing: Path, read_captures: CaptureReader
) -> None:
    """The same guard covers the session-establishing path (``_normalize_toolset`` called from
    ``_run_agent`` et al.) -- a raising fingerprint there must degrade to "no session toolset"
    rather than blowing up session open, exactly as an unrecognised session ``tools=`` already
    does (see
    ``test_record_model_call_tools_none_without_any_session_toolset_leaves_fields_unstamped``).
    """

    @capture.agent(suite="s", redact=False, tools=[RAISING_SCHEMA_TOOL])
    def agent() -> str:
        record_model_call(model_id="m", tools=None, output="ok")
        return "ok"

    agent()
    [envelope] = read_captures("s")
    [mc] = _model_events(envelope)
    assert mc["toolset_ref"] is None
    assert mc["tools_offered"] is None
    assert mc["output"] == "ok"  # the event survives regardless


def test_raising_fingerprint_logs_at_debug(
    capturing: Path, caplog: pytest.LogCaptureFixture
) -> None:
    @capture.agent(suite="s", redact=False, tools=[])
    def agent() -> str:
        record_model_call(model_id="m", tools=[RAISING_SCHEMA_TOOL], output="ok")
        return "ok"

    with caplog.at_level(logging.DEBUG, logger="evalshift"):
        agent()

    assert any("fingerprint" in r.message.lower() for r in caplog.records)


# --- F3 hardening: a value nested in input_schema that raises on copy.deepcopy itself (not just
#     on json.dumps) must degrade the same way, not propagate out of normalize_tools -- which the
#     same two per-call sites call with no wrapping guard of their own -- and reproduce F1's
#     whole-event-drop bug one step earlier in the pipeline. See
#     test_coerce_schema_degrades_to_empty_dict_when_deepcopy_itself_raises (test_toolset.py) for
#     the pure-function-level version of this regression. ---------------------------------------

UNCOPYABLE_SCHEMA_TOOL: dict[str, Any] = {
    "name": "lock_tool",
    "description": "desc",
    "input_schema": {"type": "object", "properties": {"lock": threading.Lock()}},
}


def test_record_model_call_uncopyable_schema_leaves_event_intact(
    capturing: Path, read_captures: CaptureReader
) -> None:
    @capture.agent(suite="s", redact=False, tools=[])
    def agent() -> str:
        record_model_call(model_id="m", tools=[UNCOPYABLE_SCHEMA_TOOL], input="hi", output="ok")
        return "ok"

    agent()
    [envelope] = read_captures("s")
    [mc] = _model_events(envelope)
    # normalize_tools degrades the one bad schema to {} rather than raising, so the toolset
    # still normalises and fingerprints -- unlike F1's fully-unrecognised-shape case, this event
    # keeps both toolset fields; only the one tool's schema is empty.
    assert mc["tools_offered"] == ["lock_tool"]
    assert mc["toolset_ref"] is not None
    assert mc["input"] == "hi"
    assert mc["output"] == "ok"


# --- F2: the sidecar must land under whatever base the configured sink uses, not always the
#     EVALSHIFT_DIR/CWD default -- otherwise a capture written to `configure(sink=FileSink(base=
#     X))`'s X carries a toolset_ref pointing at a sidecar that was actually written somewhere
#     else entirely, and the CLI refuses to promote it (BUNDLE_SPEC.md: "sidecar is missing"). --


def test_toolset_sidecar_follows_a_configured_filesink_base(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The core regression: EVALSHIFT_DIR is deliberately left unset here (and would point
    elsewhere than `configured_base` if it mattered) -- proving the sidecar's location comes
    from the configured sink, not env/CWD resolution."""
    monkeypatch.setenv("EVALSHIFT_CAPTURE", "1")
    monkeypatch.delenv("EVALSHIFT_DIR", raising=False)
    configured_base = tmp_path / "configured"
    configure(sink=FileSink(base=configured_base))

    @capture.agent(suite="s", redact=False, tools=[])
    def agent() -> str:
        record_model_call(model_id="m", tools=TWO_TOOLS, output="ok")
        return "ok"

    agent()

    [capture_file] = (configured_base / "captures" / "s").glob("*.json")
    envelope = json.loads(capture_file.read_text(encoding="utf-8"))
    [mc] = _model_events(envelope)
    fingerprint = mc["toolset_ref"]
    assert fingerprint is not None
    sidecar = _sidecar_path(configured_base, fingerprint)
    assert sidecar.exists()
    # And nothing was written under the (unset) default -- no stray .evalshift anywhere near cwd.
    assert not (tmp_path / ".evalshift").exists()


def test_toolset_sidecar_follows_configured_filesink_base_even_under_hygiene_wrapping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`active_sink()` transparently wraps the configured sink in a HygieneSink whenever a
    hygiene knob is set (dedup is on by default) -- the base-matching fix must see through that
    wrapper, not just the bare-sink case the previous test exercises."""
    monkeypatch.setenv("EVALSHIFT_CAPTURE", "1")
    monkeypatch.delenv("EVALSHIFT_DIR", raising=False)
    configured_base = tmp_path / "configured"
    configure(sink=FileSink(base=configured_base), dedup=True)  # explicit: force HygieneSink

    @capture.agent(suite="s", redact=False, tools=[])
    def agent() -> str:
        record_model_call(model_id="m", tools=TWO_TOOLS, output="ok", input="unique-input")
        return "ok"

    agent()

    [capture_file] = (configured_base / "captures" / "s").glob("*.json")
    envelope = json.loads(capture_file.read_text(encoding="utf-8"))
    [mc] = _model_events(envelope)
    fingerprint = mc["toolset_ref"]
    assert fingerprint is not None
    assert _sidecar_path(configured_base, fingerprint).exists()


def test_toolset_sidecar_falls_back_to_evalshift_dir_when_sink_has_no_base(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Deliberate, documented decision for a non-file sink (MemorySink is DOCS.md's own
    recommendation for read-only filesystems / Lambda): it has no on-disk base of its own for
    a sidecar to match, so the sidecar keeps using the ordinary EVALSHIFT_DIR/CWD resolution --
    it must not crash, and it must not silently resolve to nowhere. A host that genuinely runs
    on a read-only filesystem still needs EVALSHIFT_DIR pointed at a writable mount for captures
    to stay promotable (see DOCS.md's MemorySink / FAQ sections)."""
    monkeypatch.setenv("EVALSHIFT_CAPTURE", "1")
    monkeypatch.setenv("EVALSHIFT_DIR", str(tmp_path))
    sink = MemorySink()
    configure(sink=sink)

    @capture.agent(suite="s", redact=False, tools=[])
    def agent() -> str:
        record_model_call(model_id="m", tools=TWO_TOOLS, output="ok")
        return "ok"

    agent()

    [_envelope] = sink.flush()  # exactly one capture was buffered; content unused below
    fingerprint = fingerprint_tools(normalize_tools(TWO_TOOLS) or [])
    assert _sidecar_path(tmp_path, fingerprint).exists()
