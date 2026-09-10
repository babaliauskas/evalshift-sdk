"""generation_config lands in model_call metadata via every entry point."""

from __future__ import annotations

from typing import Any

from evalshift.capture.api import capture, record_model_call
from tests.conftest import CaptureReader


def _model_events(capture_dict: dict[str, Any]) -> list[dict[str, Any]]:
    return [e for e in capture_dict["trace"]["events"] if e["type"] == "model_call"]


def test_record_model_call_stamps_generation_config(
    capturing: Any, read_captures: CaptureReader
) -> None:
    cfg = {
        "temperature": 0.0,
        "response_mime_type": "application/json",
        "response_schema": {"type": "object"},
    }

    @capture.agent(suite="s", redact=False, tools=[])
    def agent() -> str:
        record_model_call(model_id="m", input="hi", output="ok", generation_config=cfg, tools=[])
        return "ok"

    agent()
    [envelope] = read_captures("s")
    [mc] = _model_events(envelope)
    assert mc["metadata"]["generation_config"] == cfg
    assert "evalshift" in mc["metadata"]  # the timing block still rides alongside


def test_record_model_call_without_generation_config_has_no_key(
    capturing: Any, read_captures: CaptureReader
) -> None:
    @capture.agent(suite="s", redact=False, tools=[])
    def agent() -> str:
        record_model_call(model_id="m", output="ok", tools=[])
        return "ok"

    agent()
    [envelope] = read_captures("s")
    [mc] = _model_events(envelope)
    assert "generation_config" not in mc["metadata"]


def test_model_call_recorder_constructor_config(
    capturing: Any, read_captures: CaptureReader
) -> None:
    cfg = {"response_mime_type": "application/json"}

    @capture.agent(suite="s", redact=False, tools=[])
    def agent() -> str:
        with capture.model_call(model_id="m", input="hi", generation_config=cfg, tools=[]) as rec:
            rec.add_text("ok")
        return "ok"

    agent()
    [envelope] = read_captures("s")
    [mc] = _model_events(envelope)
    assert mc["metadata"]["generation_config"] == cfg


def test_set_generation_config_mid_stream_last_write_wins(
    capturing: Any, read_captures: CaptureReader
) -> None:
    @capture.agent(suite="s", redact=False, tools=[])
    def agent() -> str:
        with capture.model_call(
            model_id="m", input="hi", generation_config={"temperature": 1.0}, tools=[]
        ) as rec:
            rec.add_text("ok")
            rec.set_generation_config({"temperature": 0.0})
        return "ok"

    agent()
    [envelope] = read_captures("s")
    [mc] = _model_events(envelope)
    assert mc["metadata"]["generation_config"] == {"temperature": 0.0}


def test_set_generation_config_inert_without_session() -> None:
    rec = capture.model_call(model_id="m", tools=[])
    rec.set_generation_config({"temperature": 0.0})  # must not raise outside a session


def test_record_model_call_drops_non_dict_generation_config(
    capturing: Any, read_captures: CaptureReader
) -> None:
    @capture.agent(suite="s", redact=False, tools=[])
    def agent() -> str:
        record_model_call(
            model_id="m",
            output="ok",
            generation_config="not a dict",  # type: ignore[arg-type]
            tools=[],
        )
        return "ok"

    agent()
    [envelope] = read_captures("s")
    [mc] = _model_events(envelope)
    assert "generation_config" not in mc["metadata"]


class _Schema:
    """Stand-in for a google-genai ``response_schema`` class (not JSON-serializable)."""

    def __repr__(self) -> str:
        return "<_Schema>"


def test_non_jsonable_value_still_writes_the_capture(
    capturing: Any, read_captures: CaptureReader
) -> None:
    @capture.agent(suite="s", redact=False, tools=[])
    def agent() -> str:
        record_model_call(
            model_id="m", output="ok", generation_config={"response_schema": _Schema()}, tools=[]
        )
        return "ok"

    agent()
    envelopes = read_captures("s")
    assert len(envelopes) == 1  # a non-JSON-able value must not silently drop the capture
    [mc] = _model_events(envelopes[0])
    assert mc["metadata"]["generation_config"] == {"response_schema": "<_Schema>"}


def test_non_jsonable_value_via_model_call_constructor_still_writes(
    capturing: Any, read_captures: CaptureReader
) -> None:
    @capture.agent(suite="s", redact=False, tools=[])
    def agent() -> str:
        with capture.model_call(
            model_id="m", input="hi", generation_config={"response_schema": _Schema()}, tools=[]
        ) as rec:
            rec.add_text("ok")
        return "ok"

    agent()
    envelopes = read_captures("s")
    assert len(envelopes) == 1
    [mc] = _model_events(envelopes[0])
    assert mc["metadata"]["generation_config"] == {"response_schema": "<_Schema>"}


def test_non_jsonable_value_via_set_generation_config_still_writes(
    capturing: Any, read_captures: CaptureReader
) -> None:
    @capture.agent(suite="s", redact=False, tools=[])
    def agent() -> str:
        with capture.model_call(model_id="m", input="hi", tools=[]) as rec:
            rec.add_text("ok")
            rec.set_generation_config({"response_schema": _Schema()})
        return "ok"

    agent()
    envelopes = read_captures("s")
    assert len(envelopes) == 1
    [mc] = _model_events(envelopes[0])
    assert mc["metadata"]["generation_config"] == {"response_schema": "<_Schema>"}


def test_non_allowlisted_keys_are_dropped(capturing: Any, read_captures: CaptureReader) -> None:
    @capture.agent(suite="s", redact=False, tools=[])
    def agent() -> str:
        record_model_call(
            model_id="m",
            output="ok",
            generation_config={
                "temperature": 0.0,
                "system_instruction": "You are Bob, api key sk-secret",
                "safety_settings": [{"category": "HARM", "threshold": "NONE"}],
            },
            tools=[],
        )
        return "ok"

    agent()
    [envelope] = read_captures("s")
    [mc] = _model_events(envelope)
    assert mc["metadata"]["generation_config"] == {"temperature": 0.0}


def test_nested_allowlisted_value_stays_real_json(
    capturing: Any, read_captures: CaptureReader
) -> None:
    schema = {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]}

    @capture.agent(suite="s", redact=False, tools=[])
    def agent() -> str:
        record_model_call(
            model_id="m", output="ok", generation_config={"response_schema": schema}, tools=[]
        )
        return "ok"

    agent()
    [envelope] = read_captures("s")
    [mc] = _model_events(envelope)
    assert mc["metadata"]["generation_config"]["response_schema"] == schema


def test_all_non_allowlisted_records_no_generation_config_key(
    capturing: Any, read_captures: CaptureReader
) -> None:
    @capture.agent(suite="s", redact=False, tools=[])
    def agent() -> str:
        record_model_call(
            model_id="m",
            output="ok",
            generation_config={"system_instruction": "secret", "candidate_count": 2},
            tools=[],
        )
        return "ok"

    agent()
    [envelope] = read_captures("s")
    [mc] = _model_events(envelope)
    assert "generation_config" not in mc["metadata"]


class _Boom:
    """A value whose ``__str__`` raises — the pathological framework object."""

    def __str__(self) -> str:  # pragma: no cover - exercised via jsonable
        raise RuntimeError("boom")

    __repr__ = __str__


def test_recorder_constructor_is_fail_open_on_unstringable_value(
    capturing: Any, read_captures: CaptureReader
) -> None:
    @capture.agent(suite="s", redact=False, tools=[])
    def agent() -> str:
        with capture.model_call(
            model_id="m", input="i", generation_config={"response_schema": _Boom()}, tools=[]
        ) as rec:
            rec.add_text("ok")
        return "ok"

    agent()  # must not raise into the host
    [envelope] = read_captures("s")
    [mc] = _model_events(envelope)
    assert "generation_config" not in mc["metadata"]


def test_recorder_constructor_is_fail_open_on_self_referential_config(
    capturing: Any, read_captures: CaptureReader
) -> None:
    schema: dict[str, Any] = {}
    schema["self"] = schema  # jsonable recurses -> RecursionError

    @capture.agent(suite="s", redact=False, tools=[])
    def agent() -> str:
        with capture.model_call(
            model_id="m", input="i", generation_config={"response_schema": schema}, tools=[]
        ) as rec:
            rec.add_text("ok")
        return "ok"

    agent()  # must not raise into the host
    [envelope] = read_captures("s")
    [mc] = _model_events(envelope)
    assert "generation_config" not in mc["metadata"]


# --- Phase 4 (review #8): tool-use constraints ride along with the other generation params ---


def test_tool_use_constraints_land_on_the_event(
    capturing: Any, read_captures: CaptureReader
) -> None:
    cfg = {
        "tool_choice": {"type": "tool", "name": "search"},
        "parallel_tool_calls": False,
        "temperature": 0.0,
    }

    @capture.agent(suite="s", redact=False, tools=[])
    def agent() -> str:
        record_model_call(model_id="m", output="ok", generation_config=cfg, tools=[])
        return "ok"

    agent()
    [envelope] = read_captures("s")
    [mc] = _model_events(envelope)
    assert mc["metadata"]["generation_config"] == cfg


class _ToolConfig:
    """Duck-typed stand-in for ``google.genai.types.ToolConfig`` (never imported — D-deps)."""

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        return {"function_calling_config": {"mode": "ANY", "allowed_function_names": ["search"]}}


def test_gemini_tool_config_object_lands_as_a_dict(
    capturing: Any, read_captures: CaptureReader
) -> None:
    @capture.agent(suite="s", redact=False, tools=[])
    def agent() -> str:
        record_model_call(
            model_id="m",
            output="ok",
            generation_config={"tool_config": _ToolConfig()},
            tools=[],
        )
        return "ok"

    agent()
    [envelope] = read_captures("s")
    [mc] = _model_events(envelope)
    assert mc["metadata"]["generation_config"] == {
        "tool_config": {
            "function_calling_config": {"mode": "ANY", "allowed_function_names": ["search"]}
        }
    }
