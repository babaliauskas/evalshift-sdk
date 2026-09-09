"""Recording what the *model asked for*: ``requested_tool_calls`` (D-requested, schema 2.1.0).

Covers the two model-call recording entry points -- ``record_model_call`` and the
``capture.model_call`` streaming recorder (sync and async) -- plus normalisation, the fail-open
degrade for a malformed value, and the redaction pass-through that makes this field payload --
unlike ``tools_offered`` / ``toolset_ref``, which are config (``tests/test_capture_toolset.py``).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest

from evalshift import capture, record_model_call
from evalshift.redaction import default_redactor
from tests.conftest import CaptureReader

SEARCH_CALL: dict[str, Any] = {
    "name": "search_orders",
    "arguments": {"customer_id": "customer_42"},
    "call_id": "toolu_01",
}


def _model_events(capture_dict: dict[str, Any]) -> list[dict[str, Any]]:
    return [e for e in capture_dict["trace"]["events"] if e["type"] == "model_call"]


def _one_model_event(read_captures: CaptureReader, suite: str = "s") -> dict[str, Any]:
    [envelope] = read_captures(suite)
    [model_call] = _model_events(envelope)
    return model_call


# --- record_model_call: the value lands on the event ------------------------------------------


def test_record_model_call_records_requested_tool_calls(
    capturing: Path, read_captures: CaptureReader
) -> None:
    @capture.agent(suite="s", redact=False, tools=[])
    def agent() -> None:
        record_model_call(model_id="m", tools=[], output="", requested_tool_calls=[SEARCH_CALL])

    agent()
    assert _one_model_event(read_captures)["requested_tool_calls"] == [SEARCH_CALL]


def test_record_model_call_omitting_requested_tool_calls_leaves_field_none(
    capturing: Path, read_captures: CaptureReader
) -> None:
    @capture.agent(suite="s", redact=False, tools=[])
    def agent() -> None:
        record_model_call(model_id="m", tools=[], output="ok")

    agent()
    assert _one_model_event(read_captures)["requested_tool_calls"] is None


def test_record_model_call_explicit_none_leaves_field_none(
    capturing: Path, read_captures: CaptureReader
) -> None:
    @capture.agent(suite="s", redact=False, tools=[])
    def agent() -> None:
        record_model_call(model_id="m", tools=[], output="ok", requested_tool_calls=None)

    agent()
    assert _one_model_event(read_captures)["requested_tool_calls"] is None


def test_record_model_call_empty_list_records_an_asserted_empty_list(
    capturing: Path, read_captures: CaptureReader
) -> None:
    """``[]`` is a real value -- "the model requested no tools" -- and must not collapse to the
    ``None`` that means "not recorded". The CLI's fallback to executed calls turns on exactly
    this distinction (D-requested)."""

    @capture.agent(suite="s", redact=False, tools=[])
    def agent() -> None:
        record_model_call(model_id="m", tools=[], output="ok", requested_tool_calls=[])

    agent()
    assert _one_model_event(read_captures)["requested_tool_calls"] == []


# --- normalisation: exactly {name, arguments, call_id} ----------------------------------------


def test_items_are_normalised_to_exactly_the_three_contract_keys(
    capturing: Path, read_captures: CaptureReader
) -> None:
    """The CLI's ``RequestedToolCall`` is ``extra="forbid"``, so a provider-shaped item's extra
    keys (``type``, ``index``, ``id``, ...) must be dropped before the capture is written."""

    @capture.agent(suite="s", redact=False, tools=[])
    def agent() -> None:
        record_model_call(
            model_id="m",
            tools=[],
            requested_tool_calls=[
                {
                    "type": "tool_use",
                    "index": 0,
                    "name": "search_orders",
                    "arguments": {"customer_id": "c42"},
                    "call_id": "toolu_01",
                }
            ],
        )

    agent()
    [call] = _one_model_event(read_captures)["requested_tool_calls"]
    assert call == {
        "name": "search_orders",
        "arguments": {"customer_id": "c42"},
        "call_id": "toolu_01",
    }


def test_missing_arguments_and_call_id_get_contract_defaults(
    capturing: Path, read_captures: CaptureReader
) -> None:
    @capture.agent(suite="s", redact=False, tools=[])
    def agent() -> None:
        record_model_call(model_id="m", tools=[], requested_tool_calls=[{"name": "ping"}])

    agent()
    [call] = _one_model_event(read_captures)["requested_tool_calls"]
    assert call == {"name": "ping", "arguments": {}, "call_id": None}


def test_non_dict_arguments_degrade_to_empty_dict(
    capturing: Path, read_captures: CaptureReader
) -> None:
    """``arguments`` is typed ``dict`` on the CLI side; a provider string (unparsed JSON, say)
    would fail validation, so it degrades to ``{}`` rather than taking the event down."""

    @capture.agent(suite="s", redact=False, tools=[])
    def agent() -> None:
        record_model_call(
            model_id="m", tools=[], requested_tool_calls=[{"name": "ping", "arguments": "{}"}]
        )

    agent()
    [call] = _one_model_event(read_captures)["requested_tool_calls"]
    assert call["arguments"] == {}


def test_non_string_call_id_is_coerced_to_string(
    capturing: Path, read_captures: CaptureReader
) -> None:
    @capture.agent(suite="s", redact=False, tools=[])
    def agent() -> None:
        record_model_call(
            model_id="m", tools=[], requested_tool_calls=[{"name": "ping", "call_id": 7}]
        )

    agent()
    [call] = _one_model_event(read_captures)["requested_tool_calls"]
    assert call["call_id"] == "7"


def test_unusable_items_are_dropped_and_usable_ones_kept(
    capturing: Path, read_captures: CaptureReader
) -> None:
    @capture.agent(suite="s", redact=False, tools=[])
    def agent() -> None:
        record_model_call(
            model_id="m",
            tools=[],
            requested_tool_calls=["nope", {"no_name": 1}, {"name": ""}, SEARCH_CALL],
        )

    agent()
    assert _one_model_event(read_captures)["requested_tool_calls"] == [SEARCH_CALL]


# --- fail-open: a malformed value is dropped, never raised ------------------------------------


@pytest.mark.parametrize("value", ["not-a-list", 42, {"name": "ping"}, object()])
def test_non_list_value_is_dropped_fail_open(
    capturing: Path, read_captures: CaptureReader, value: Any
) -> None:
    """A bad value leaves the field unrecorded (``None``) and never breaks the host agent -- the
    model_call event, and the whole capture, are still written."""

    @capture.agent(suite="s", redact=False, tools=[])
    def agent() -> str:
        record_model_call(model_id="m", tools=[], output="ok", requested_tool_calls=value)
        return "host result"

    assert agent() == "host result"
    model_call = _one_model_event(read_captures)
    assert model_call["requested_tool_calls"] is None
    assert model_call["output"] == "ok"


def test_list_with_no_usable_items_records_nothing_rather_than_empty(
    capturing: Path, read_captures: CaptureReader
) -> None:
    """All items unusable is "we could not read what the model asked for", not "it asked for
    nothing" -- so the field stays ``None``, never ``[]``."""

    @capture.agent(suite="s", redact=False, tools=[])
    def agent() -> None:
        record_model_call(model_id="m", tools=[], requested_tool_calls=[1, 2, 3])

    agent()
    assert _one_model_event(read_captures)["requested_tool_calls"] is None


def test_dropped_value_is_logged_at_debug(
    capturing: Path, read_captures: CaptureReader, caplog: pytest.LogCaptureFixture
) -> None:
    @capture.agent(suite="s", redact=False, tools=[])
    def agent() -> None:
        record_model_call(model_id="m", tools=[], requested_tool_calls="not-a-list")

    with caplog.at_level(logging.DEBUG, logger="evalshift"):
        agent()
    assert any("requested_tool_calls" in record.message for record in caplog.records)


# --- capture.model_call recorder (sync + async) ------------------------------------------------


def test_streaming_recorder_records_requested_tool_calls(
    capturing: Path, read_captures: CaptureReader
) -> None:
    @capture.agent(suite="s", redact=False, tools=[])
    def agent() -> None:
        with capture.model_call(model_id="m", tools=[], input="hi") as rec:
            rec.add_text("ok")
            rec.set_requested_tool_calls([SEARCH_CALL])

    agent()
    model_call = _one_model_event(read_captures)
    assert model_call["requested_tool_calls"] == [SEARCH_CALL]
    assert model_call["output"] == "ok"


async def test_async_streaming_recorder_records_requested_tool_calls(
    capturing: Path, read_captures: CaptureReader
) -> None:
    @capture.agent(suite="s", redact=False, tools=[])
    async def agent() -> None:
        async with capture.model_call(model_id="m", tools=[], input="hi") as rec:
            rec.add_text("ok")
            rec.set_requested_tool_calls([SEARCH_CALL])

    await agent()
    assert _one_model_event(read_captures)["requested_tool_calls"] == [SEARCH_CALL]


def test_streaming_recorder_without_the_call_leaves_field_none(
    capturing: Path, read_captures: CaptureReader
) -> None:
    @capture.agent(suite="s", redact=False, tools=[])
    def agent() -> None:
        with capture.model_call(model_id="m", tools=[], input="hi") as rec:
            rec.add_text("ok")

    agent()
    assert _one_model_event(read_captures)["requested_tool_calls"] is None


def test_streaming_recorder_set_before_entering_still_lands(
    capturing: Path, read_captures: CaptureReader
) -> None:
    """The recorder is constructed before the ``with``; a value set on it beforehand must be
    stamped onto the span when it opens, not silently lost."""

    @capture.agent(suite="s", redact=False, tools=[])
    def agent() -> None:
        rec = capture.model_call(model_id="m", tools=[], input="hi")
        rec.set_requested_tool_calls([SEARCH_CALL])
        with rec:
            rec.add_text("ok")

    agent()
    assert _one_model_event(read_captures)["requested_tool_calls"] == [SEARCH_CALL]


def test_streaming_recorder_last_write_wins(capturing: Path, read_captures: CaptureReader) -> None:
    @capture.agent(suite="s", redact=False, tools=[])
    def agent() -> None:
        with capture.model_call(model_id="m", tools=[], input="hi") as rec:
            rec.set_requested_tool_calls([{"name": "first"}])
            rec.set_requested_tool_calls([SEARCH_CALL])

    agent()
    assert _one_model_event(read_captures)["requested_tool_calls"] == [SEARCH_CALL]


def test_streaming_recorder_malformed_value_is_dropped_fail_open(
    capturing: Path, read_captures: CaptureReader
) -> None:
    @capture.agent(suite="s", redact=False, tools=[])
    def agent() -> str:
        with capture.model_call(model_id="m", tools=[], input="hi") as rec:
            rec.set_requested_tool_calls("not-a-list")
            rec.add_text("ok")
        return "host result"

    assert agent() == "host result"
    model_call = _one_model_event(read_captures)
    assert model_call["requested_tool_calls"] is None
    assert model_call["output"] == "ok"


def test_streaming_recorder_outside_a_session_is_inert() -> None:
    """No active session -> the recorder is a no-op; setting a value must not raise."""
    with capture.model_call(model_id="m", tools=[], input="hi") as rec:
        rec.set_requested_tool_calls([SEARCH_CALL])


# --- redaction: requested arguments are payload, and are masked (D-4c / D-requested) -----------


def test_redactor_masks_requested_tool_call_arguments(
    capturing: Path, read_captures: CaptureReader
) -> None:
    @capture.agent(suite="s", redact=default_redactor, tools=[])
    def agent() -> None:
        record_model_call(
            model_id="m",
            tools=[],
            requested_tool_calls=[
                {"name": "email_customer", "arguments": {"to": "bob@corp.com"}, "call_id": "t1"}
            ],
        )

    agent()
    [call] = _one_model_event(read_captures)["requested_tool_calls"]
    assert "bob@corp.com" not in str(call)
    assert "[REDACTED_EMAIL]" in call["arguments"]["to"]
    assert call["name"] == "email_customer"  # the tool name is structural, never masked


def test_streaming_recorder_requested_tool_call_arguments_are_redacted(
    capturing: Path, read_captures: CaptureReader
) -> None:
    @capture.agent(suite="s", redact=default_redactor, tools=[])
    def agent() -> None:
        with capture.model_call(model_id="m", tools=[], input="hi") as rec:
            rec.set_requested_tool_calls(
                [{"name": "email_customer", "arguments": {"to": "bob@corp.com"}}]
            )

    agent()
    [call] = _one_model_event(read_captures)["requested_tool_calls"]
    assert "bob@corp.com" not in str(call)
