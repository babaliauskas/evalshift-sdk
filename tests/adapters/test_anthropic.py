"""The Anthropic client wrapper (`adapters/anthropic.py`).

Exercised through a synthetic client whose attribute paths mirror ``anthropic.Anthropic`` /
``AsyncAnthropic`` (``client.messages.create`` / ``.stream``), with responses and stream events
built from the real ``anthropic`` pydantic types so the shapes are the ones production hands us.
One ``importorskip``-guarded test drives a real ``anthropic.Anthropic`` over a stub transport.
"""

from __future__ import annotations

import importlib
import json
import sys
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import pytest

from evalshift import capture
from evalshift.adapters import anthropic as adapter
from evalshift.adapters._wrap import AsyncStreamProxy, StreamProxy, unwrap
from evalshift.adapters.anthropic import wrap_anthropic
from tests.conftest import CaptureReader

anthropic = pytest.importorskip("anthropic")
from anthropic.types import (  # noqa: E402
    Message,
    RawContentBlockDeltaEvent,
    RawContentBlockStartEvent,
    RawContentBlockStopEvent,
    RawMessageDeltaEvent,
    RawMessageStartEvent,
    RawMessageStopEvent,
)


def _model_calls(capture_dict: dict[str, Any]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = capture_dict["trace"]["events"]
    return [e for e in events if e["type"] == "model_call"]


# --- realistic payloads -----------------------------------------------------------------------

TOOLS = [
    {
        "name": "search",
        "description": "Search the web",
        "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}},
    }
]

TEXT_MESSAGE = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "model": "claude-opus-5",
    "content": [{"type": "text", "text": "Hello "}, {"type": "text", "text": "there"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 12, "output_tokens": 5},
}

TOOL_MESSAGE = {
    **TEXT_MESSAGE,
    "content": [
        {"type": "text", "text": "Let me look."},
        {"type": "tool_use", "id": "toolu_1", "name": "search", "input": {"q": "evalshift"}},
    ],
    "stop_reason": "tool_use",
}


def _message(payload: dict[str, Any]) -> Message:
    return Message.model_validate(payload)


def _events(*, tool_use: bool) -> list[Any]:
    start = RawMessageStartEvent.model_validate(
        {"type": "message_start", "message": {**TEXT_MESSAGE, "content": [], "stop_reason": None}}
    )
    events: list[Any] = [
        start,
        RawContentBlockStartEvent.model_validate(
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            }
        ),
        RawContentBlockDeltaEvent.model_validate(
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "Hel"},
            }
        ),
        RawContentBlockDeltaEvent.model_validate(
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "lo"},
            }
        ),
        RawContentBlockStopEvent.model_validate({"type": "content_block_stop", "index": 0}),
    ]
    if tool_use:
        events += [
            RawContentBlockStartEvent.model_validate(
                {
                    "type": "content_block_start",
                    "index": 1,
                    "content_block": {
                        "type": "tool_use",
                        "id": "toolu_s",
                        "name": "search",
                        "input": {},
                    },
                }
            ),
            RawContentBlockDeltaEvent.model_validate(
                {
                    "type": "content_block_delta",
                    "index": 1,
                    "delta": {"type": "input_json_delta", "partial_json": '{"q": "ev'},
                }
            ),
            RawContentBlockDeltaEvent.model_validate(
                {
                    "type": "content_block_delta",
                    "index": 1,
                    "delta": {"type": "input_json_delta", "partial_json": 'alshift"}'},
                }
            ),
            RawContentBlockStopEvent.model_validate({"type": "content_block_stop", "index": 1}),
        ]
    events += [
        RawMessageDeltaEvent.model_validate(
            {
                "type": "message_delta",
                "delta": {
                    "stop_reason": "tool_use" if tool_use else "end_turn",
                    "stop_sequence": None,
                },
                "usage": {"output_tokens": 9},
            }
        ),
        RawMessageStopEvent.model_validate({"type": "message_stop"}),
    ]
    return events


# --- a synthetic client with anthropic's attribute paths ------------------------------------


class _Stream:
    def __init__(self, events: list[Any]) -> None:
        self._events = events
        self.closed = False

    def __iter__(self) -> Iterator[Any]:
        yield from self._events

    def __enter__(self) -> _Stream:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self.closed = True


class _AsyncStream:
    def __init__(self, events: list[Any]) -> None:
        self._events = events

    async def __aiter__(self) -> AsyncIterator[Any]:
        for event in self._events:
            yield event

    async def close(self) -> None:
        pass


class _MessageStream:
    """Stand-in for ``anthropic.MessageStream``: consumable, then ``get_final_message()``."""

    def __init__(self, final: Any, *, fail: bool = False) -> None:
        self._final = final
        self._fail = fail
        self.consumed = False
        self.final_calls = 0
        self.closed = False

    @property
    def text_stream(self) -> Iterator[str]:
        self.consumed = True
        yield "Hello "
        yield "there"

    def get_final_message(self) -> Any:
        self.final_calls += 1
        if self._fail:
            raise RuntimeError("stream errored")
        return self._final

    @property
    def current_message_snapshot(self) -> Any:
        return _message({**TEXT_MESSAGE, "content": [{"type": "text", "text": "partial"}]})

    def close(self) -> None:
        self.closed = True


class _Manager:
    def __init__(self, stream: _MessageStream) -> None:
        self._stream = stream
        self.request_id = "req_x"

    def __enter__(self) -> _MessageStream:
        return self._stream

    def __exit__(self, *exc: object) -> None:
        self._stream.close()


class _AsyncMessageStream(_MessageStream):
    async def get_final_message(self) -> Any:
        return super().get_final_message()


class _AsyncManager:
    def __init__(self, stream: _AsyncMessageStream) -> None:
        self._stream = stream

    async def __aenter__(self) -> _AsyncMessageStream:
        return self._stream

    async def __aexit__(self, *exc: object) -> None:
        self._stream.close()


class _Messages:
    def __init__(self, response: Any = None, events: list[Any] | None = None) -> None:
        self.response = response if response is not None else _message(TEXT_MESSAGE)
        self.events = events or _events(tool_use=False)
        self.calls: list[dict[str, Any]] = []
        self.last_stream: _MessageStream | None = None

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if kwargs.get("fail"):
            raise RuntimeError("provider down")
        if kwargs.get("stream") is True:
            return _Stream(self.events)
        return self.response

    def stream(self, **kwargs: Any) -> _Manager:
        self.calls.append(kwargs)
        self.last_stream = _MessageStream(self.response, fail=bool(kwargs.get("fail")))
        return _Manager(self.last_stream)

    def count_tokens(self, **kwargs: Any) -> str:
        return "untouched"


class _AsyncMessages(_Messages):
    # Mirrors anthropic 1.x: ``AsyncMessages.create`` is a plain ``def`` returning a coroutine.
    def create(self, **kwargs: Any) -> Any:
        return self._create(**kwargs)

    async def _create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if kwargs.get("stream") is True:
            return _AsyncStream(self.events)
        return self.response

    def stream(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        self.last_stream = _AsyncMessageStream(self.response, fail=bool(kwargs.get("fail")))
        return _AsyncManager(self.last_stream)


class _Client:
    def __init__(self, messages: _Messages | None = None) -> None:
        self.messages = messages or _Messages()
        self.api_key = "sk-test"

    def __enter__(self) -> _Client:
        return self

    def __exit__(self, *exc: object) -> None:
        pass


USER = [{"role": "user", "content": "hi"}]


# --- proxying ---------------------------------------------------------------------------------


def test_unrelated_attributes_forward(capturing: Path, read_captures: CaptureReader) -> None:
    client = _Client()
    proxy = wrap_anthropic(client)
    assert proxy.api_key == "sk-test"
    assert proxy.messages.count_tokens(model="m") == "untouched"
    assert unwrap(proxy) is client
    with proxy as entered:
        assert unwrap(entered) is client
    assert "wrapper" in repr(proxy)


def test_outside_session_is_inert(capturing: Path, read_captures: CaptureReader) -> None:
    proxy = wrap_anthropic(_Client())
    response = proxy.messages.create(model="claude-opus-5", max_tokens=8, messages=USER)
    assert response.content[0].text == "Hello "
    with proxy.messages.stream(model="claude-opus-5", max_tokens=8, messages=USER) as stream:
        assert "".join(stream.text_stream) == "Hello there"
    assert read_captures("anthropic") == []


# --- messages.create --------------------------------------------------------------------------


def test_sync_create_records(capturing: Path, read_captures: CaptureReader) -> None:
    client = _Client()
    proxy = wrap_anthropic(client)

    @capture.agent(suite="anthropic", redact=False, tools=[])
    def agent() -> Any:
        return proxy.messages.create(
            model="claude-opus-5",
            max_tokens=64,
            temperature=0.3,
            system="Be brief.",
            messages=USER,
            tools=TOOLS,
        )

    response = agent()
    assert response.content[1].text == "there"
    assert client.messages.calls[0]["tools"] is TOOLS  # kwargs reach the client untouched

    (call,) = _model_calls(read_captures("anthropic")[0])
    assert call["model_id"] == "claude-opus-5"
    assert call["input"] == [{"role": "system", "content": "Be brief."}, *USER]
    assert call["output"] == "Hello there"
    assert call["input_tokens"] == 12
    assert call["output_tokens"] == 5
    assert call["latency_ms"] >= 0
    assert call["tools_offered"] == ["search"]
    assert call["requested_tool_calls"] == []
    assert call["metadata"]["generation_config"] == {"temperature": 0.3, "max_tokens": 64}


def test_create_with_tool_use_records_requested_calls(
    capturing: Path, read_captures: CaptureReader
) -> None:
    proxy = wrap_anthropic(_Client(_Messages(response=_message(TOOL_MESSAGE))))

    @capture.agent(suite="anthropic", redact=False, tools=[])
    def agent() -> None:
        proxy.messages.create(model="claude-opus-5", max_tokens=8, messages=USER, tools=TOOLS)

    agent()
    (call,) = _model_calls(read_captures("anthropic")[0])
    assert call["output"] == "Let me look."
    assert call["requested_tool_calls"] == [
        {"name": "search", "arguments": {"q": "evalshift"}, "call_id": "toolu_1"}
    ]


def test_dict_response_works_too(capturing: Path, read_captures: CaptureReader) -> None:
    proxy = wrap_anthropic(_Client(_Messages(response=TOOL_MESSAGE)))

    @capture.agent(suite="anthropic", redact=False, tools=[])
    def agent() -> None:
        proxy.messages.create(model="claude-opus-5", max_tokens=8, messages=USER)

    agent()
    (call,) = _model_calls(read_captures("anthropic")[0])
    assert call["output"] == "Let me look."
    assert call["requested_tool_calls"][0]["call_id"] == "toolu_1"


def test_no_tools_kwarg_asserts_empty_toolset(
    capturing: Path, read_captures: CaptureReader
) -> None:
    proxy = wrap_anthropic(_Client())

    @capture.agent(suite="anthropic", redact=False, tools=TOOLS)
    def agent() -> None:
        proxy.messages.create(model="claude-opus-5", max_tokens=8, messages=USER)

    agent()
    (call,) = _model_calls(read_captures("anthropic")[0])
    assert call["tools_offered"] == []


def test_system_blocks_and_absent_system(capturing: Path, read_captures: CaptureReader) -> None:
    proxy = wrap_anthropic(_Client())
    blocks = [{"type": "text", "text": "Be brief.", "cache_control": {"type": "ephemeral"}}]

    @capture.agent(suite="anthropic", redact=False, tools=[])
    def agent() -> None:
        proxy.messages.create(model="claude-opus-5", max_tokens=8, system=blocks, messages=USER)
        proxy.messages.create(model="claude-opus-5", max_tokens=8, messages=USER)

    agent()
    first, second = _model_calls(read_captures("anthropic")[0])
    assert first["input"] == [{"role": "system", "content": blocks}, *USER]
    assert second["input"] == USER


def test_unexpected_response_shape_is_fail_open(
    capturing: Path, read_captures: CaptureReader
) -> None:
    proxy = wrap_anthropic(_Client(_Messages(response="not a message")))

    @capture.agent(suite="anthropic", redact=False, tools=[])
    def agent() -> Any:
        return proxy.messages.create(model="claude-opus-5", max_tokens=8, messages=USER)

    assert agent() == "not a message"
    assert _model_calls(read_captures("anthropic")[0]) == []


def test_provider_error_propagates_and_records_nothing(
    capturing: Path, read_captures: CaptureReader
) -> None:
    proxy = wrap_anthropic(_Client())

    @capture.agent(suite="anthropic", redact=False, tools=[])
    def agent() -> None:
        proxy.messages.create(model="claude-opus-5", max_tokens=8, messages=USER, fail=True)

    with pytest.raises(RuntimeError, match="provider down"):
        agent()
    assert _model_calls(read_captures("anthropic")[0]) == []


async def test_async_create_records(capturing: Path, read_captures: CaptureReader) -> None:
    proxy = wrap_anthropic(_Client(_AsyncMessages(response=_message(TOOL_MESSAGE))))

    @capture.agent(suite="anthropic", redact=False, tools=[])
    async def agent() -> Any:
        return await proxy.messages.create(
            model="claude-opus-5", max_tokens=8, messages=USER, tools=TOOLS
        )

    response = await agent()
    assert response.stop_reason == "tool_use"
    (call,) = _model_calls(read_captures("anthropic")[0])
    assert call["output"] == "Let me look."
    assert call["input_tokens"] == 12
    assert call["requested_tool_calls"][0]["name"] == "search"


# --- messages.create(stream=True) -------------------------------------------------------------


def test_sync_stream_records_text_and_usage(capturing: Path, read_captures: CaptureReader) -> None:
    proxy = wrap_anthropic(_Client())

    @capture.agent(suite="anthropic", redact=False, tools=[])
    def agent() -> list[str]:
        stream = proxy.messages.create(
            model="claude-opus-5", max_tokens=8, messages=USER, stream=True
        )
        assert isinstance(stream, StreamProxy)
        with stream:
            return [event.type for event in stream]

    kinds = agent()
    assert kinds[0] == "message_start" and kinds[-1] == "message_stop"
    (call,) = _model_calls(read_captures("anthropic")[0])
    assert call["output"] == "Hello"
    assert call["input_tokens"] == 12
    assert call["output_tokens"] == 9
    assert call["requested_tool_calls"] == []


def test_sync_stream_assembles_tool_use(capturing: Path, read_captures: CaptureReader) -> None:
    proxy = wrap_anthropic(_Client(_Messages(events=_events(tool_use=True))))

    @capture.agent(suite="anthropic", redact=False, tools=[])
    def agent() -> None:
        for _ in proxy.messages.create(
            model="claude-opus-5", max_tokens=8, messages=USER, tools=TOOLS, stream=True
        ):
            pass

    agent()
    (call,) = _model_calls(read_captures("anthropic")[0])
    assert call["output"] == "Hello"
    assert call["requested_tool_calls"] == [
        {"name": "search", "arguments": {"q": "evalshift"}, "call_id": "toolu_s"}
    ]


async def test_async_stream_assembles_tool_use(
    capturing: Path, read_captures: CaptureReader
) -> None:
    proxy = wrap_anthropic(_Client(_AsyncMessages(events=_events(tool_use=True))))

    @capture.agent(suite="anthropic", redact=False, tools=[])
    async def agent() -> int:
        stream = await proxy.messages.create(
            model="claude-opus-5", max_tokens=8, messages=USER, stream=True
        )
        assert isinstance(stream, AsyncStreamProxy)
        return len([event async for event in stream])

    assert await agent() == 11  # message_start, 4 text, 4 tool_use, message_delta/stop
    (call,) = _model_calls(read_captures("anthropic")[0])
    assert call["output"] == "Hello"
    assert call["output_tokens"] == 9
    assert call["requested_tool_calls"][0]["call_id"] == "toolu_s"


# --- messages.stream() ------------------------------------------------------------------------


def test_stream_helper_records_on_exit(capturing: Path, read_captures: CaptureReader) -> None:
    messages = _Messages(response=_message(TOOL_MESSAGE))
    proxy = wrap_anthropic(_Client(messages))

    @capture.agent(suite="anthropic", redact=False, tools=[])
    def agent() -> str:
        manager = proxy.messages.stream(
            model="claude-opus-5", max_tokens=8, system="Be brief.", messages=USER, tools=TOOLS
        )
        assert manager.request_id == "req_x"  # forwarded from the real manager
        with manager as stream:
            assert isinstance(stream, _MessageStream)  # the SDK's own stream, unwrapped
            text = "".join(stream.text_stream)
            assert read_captures("anthropic") == []  # nothing recorded until exit
        return text

    assert agent() == "Hello there"
    assert messages.last_stream is not None
    assert messages.last_stream.final_calls == 1
    assert messages.last_stream.closed
    (call,) = _model_calls(read_captures("anthropic")[0])
    assert call["input"] == [{"role": "system", "content": "Be brief."}, *USER]
    assert call["output"] == "Let me look."
    assert call["input_tokens"] == 12
    assert call["tools_offered"] == ["search"]
    assert call["requested_tool_calls"] == [
        {"name": "search", "arguments": {"q": "evalshift"}, "call_id": "toolu_1"}
    ]
    assert call["metadata"]["generation_config"] == {"max_tokens": 8}


def test_stream_helper_records_nothing_when_final_message_fails(
    capturing: Path, read_captures: CaptureReader
) -> None:
    proxy = wrap_anthropic(_Client())

    @capture.agent(suite="anthropic", redact=False, tools=[])
    def agent() -> None:
        with proxy.messages.stream(model="claude-opus-5", max_tokens=8, messages=USER, fail=True):
            pass

    agent()
    assert _model_calls(read_captures("anthropic")[0]) == []


def test_stream_helper_records_snapshot_when_body_raises(
    capturing: Path, read_captures: CaptureReader
) -> None:
    messages = _Messages()
    proxy = wrap_anthropic(_Client(messages))

    @capture.agent(suite="anthropic", redact=False, tools=[])
    def agent() -> None:
        with proxy.messages.stream(model="claude-opus-5", max_tokens=8, messages=USER):
            raise ValueError("user code failed")

    with pytest.raises(ValueError, match="user code failed"):
        agent()
    assert messages.last_stream is not None
    assert messages.last_stream.final_calls == 0  # no network read on the error path
    (call,) = _model_calls(read_captures("anthropic")[0])
    assert call["output"] == "partial"


async def test_async_stream_helper_records_on_exit(
    capturing: Path, read_captures: CaptureReader
) -> None:
    messages = _AsyncMessages()
    proxy = wrap_anthropic(_Client(messages))

    @capture.agent(suite="anthropic", redact=False, tools=[])
    async def agent() -> str:
        async with proxy.messages.stream(model="claude-opus-5", max_tokens=8, messages=USER) as s:
            assert isinstance(s, _AsyncMessageStream)
            return "".join(s.text_stream)

    assert await agent() == "Hello there"
    assert messages.last_stream is not None
    assert messages.last_stream.closed
    (call,) = _model_calls(read_captures("anthropic")[0])
    assert call["output"] == "Hello there"
    assert call["output_tokens"] == 5
    assert call["requested_tool_calls"] == []


# --- packaging --------------------------------------------------------------------------------


def test_module_imports_without_anthropic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "anthropic", None)
    reloaded = importlib.reload(adapter)
    assert callable(reloaded.wrap_anthropic)
    monkeypatch.undo()
    importlib.reload(adapter)


# --- the real client over a stub transport ----------------------------------------------------


def test_real_client_smoke(capturing: Path, read_captures: CaptureReader) -> None:
    httpx2 = pytest.importorskip("httpx2")
    seen: list[dict[str, Any]] = []

    def handler(request: Any) -> Any:
        seen.append(json.loads(request.content))
        return httpx2.Response(200, json=TOOL_MESSAGE)

    client = anthropic.Anthropic(
        api_key="test", http_client=httpx2.Client(transport=httpx2.MockTransport(handler))
    )
    proxy = wrap_anthropic(client)

    @capture.agent(suite="anthropic", redact=False, tools=[])
    def agent() -> Any:
        return proxy.messages.create(
            model="claude-opus-5", max_tokens=64, system="Be brief.", messages=USER, tools=TOOLS
        )

    response = agent()
    assert isinstance(response, Message)
    assert seen[0]["system"] == "Be brief."
    (call,) = _model_calls(read_captures("anthropic")[0])
    assert call["model_id"] == "claude-opus-5"
    assert call["input"] == [{"role": "system", "content": "Be brief."}, *USER]
    assert call["output"] == "Let me look."
    assert call["input_tokens"] == 12
    assert call["tools_offered"] == ["search"]
    assert call["requested_tool_calls"] == [
        {"name": "search", "arguments": {"q": "evalshift"}, "call_id": "toolu_1"}
    ]
