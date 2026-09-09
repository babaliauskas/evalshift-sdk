"""The shared wrapper base (`adapters/_wrap.py`): proxying, timing, streaming, fail-open.

Exercised through a synthetic client so no provider SDK is needed. The per-provider modules
only add ``describe`` / ``complete`` / ``on_chunk`` functions on top of what is pinned here.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import pytest

from evalshift import capture
from evalshift.adapters._wrap import (
    AsyncStreamProxy,
    CallSpec,
    ClientProxy,
    Completion,
    Instrumentation,
    StreamProxy,
    StreamState,
    unwrap,
)
from tests.conftest import CaptureReader


def _model_calls(capture_dict: dict[str, Any]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = capture_dict["trace"]["events"]
    return [e for e in events if e["type"] == "model_call"]


# --- a synthetic provider ---------------------------------------------------------------------


class _Stream:
    """A provider-style sync stream: iterable, context manager, closeable."""

    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        self._chunks = chunks
        self.closed = False
        self.entered = False

    def __iter__(self) -> Iterator[dict[str, Any]]:
        yield from self._chunks

    def __enter__(self) -> _Stream:
        self.entered = True
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self.closed = True


class _AsyncStream:
    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        self._chunks = chunks
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[dict[str, Any]]:
        for chunk in self._chunks:
            yield chunk

    async def aclose(self) -> None:
        self.closed = True


class _Completions:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if kwargs.get("fail"):
            raise RuntimeError("provider down")
        if kwargs.get("stream"):
            return _Stream([{"text": "a", "usage": None}, {"text": "b", "usage": (3, 2)}])
        return {
            "text": "hello",
            "usage": (10, 4),
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {"id": "c1", "function": {"name": "search", "arguments": '{"q": "x"}'}}
                        ]
                    }
                }
            ],
        }

    async def acreate(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            return _AsyncStream([{"text": "x", "usage": None}, {"text": "y", "usage": (1, 1)}])
        return {"text": "async hello", "usage": (2, 2), "choices": [{"message": {}}]}

    def other(self) -> str:
        return "untouched"


class _Chat:
    def __init__(self) -> None:
        self.completions = _Completions()


class _Client:
    def __init__(self) -> None:
        self.chat = _Chat()
        self.entered = False
        self.exited = False

    def __enter__(self) -> _Client:
        self.entered = True
        return self

    def __exit__(self, *exc: object) -> None:
        self.exited = True

    def ping(self) -> str:
        return "pong"


def _describe(kwargs: Any) -> CallSpec:
    return CallSpec(
        model_id=kwargs["model"],
        tools=kwargs.get("tools", []),
        input=kwargs.get("messages"),
        generation_config=dict(kwargs),
    )


def _complete(response: Any) -> Completion:
    tokens = response["usage"]
    return Completion(output=response["text"], input_tokens=tokens[0], output_tokens=tokens[1])


def _on_chunk(chunk: Any, state: StreamState) -> None:
    state.text.append(chunk["text"])
    if chunk["usage"] is not None:
        state.input_tokens, state.output_tokens = chunk["usage"]


_INST = Instrumentation(
    describe=_describe,
    complete=_complete,
    is_stream=lambda kwargs, response: bool(kwargs.get("stream")),
    on_chunk=_on_chunk,
)


def _wrap(client: _Client) -> ClientProxy:
    return ClientProxy(client, {"chat": {"completions": {"create": _INST, "acreate": _INST}}})


# --- proxying ---------------------------------------------------------------------------------


def test_proxy_forwards_everything_else() -> None:
    client = _Client()
    proxy = _wrap(client)
    assert proxy.ping() == "pong"
    assert proxy.chat.completions.other() == "untouched"
    assert unwrap(proxy) is client
    assert unwrap(client) is client
    assert "ping" in dir(proxy)
    assert "wrapper" in repr(proxy)


def test_proxy_forwards_context_manager() -> None:
    client = _Client()
    with _wrap(client) as proxy:
        assert isinstance(proxy, ClientProxy)
        assert client.entered
    assert client.exited


def test_missing_attribute_raises_attribute_error() -> None:
    with pytest.raises(AttributeError):
        _ = _wrap(_Client()).nope


# --- recording --------------------------------------------------------------------------------


def test_sync_call_records_one_model_call(capturing: Path, read_captures: CaptureReader) -> None:
    client = _Client()
    proxy = _wrap(client)
    tools = [{"type": "function", "function": {"name": "search", "parameters": {}}}]

    @capture.agent(suite="wrap", redact=False, tools=[])
    def agent() -> Any:
        return proxy.chat.completions.create(
            model="m-1", messages=[{"role": "user", "content": "hi"}], tools=tools, temperature=0.2
        )

    response = agent()
    assert response["text"] == "hello"
    assert client.chat.completions.calls[0]["model"] == "m-1"

    (call,) = _model_calls(read_captures("wrap")[0])
    assert call["model_id"] == "m-1"
    assert call["input"] == [{"role": "user", "content": "hi"}]
    assert call["output"] == "hello"
    assert call["input_tokens"] == 10
    assert call["output_tokens"] == 4
    assert call["cost_usd"] == 0
    assert call["latency_ms"] >= 0
    assert call["tools_offered"] == ["search"]
    assert call["requested_tool_calls"] == [
        {"name": "search", "arguments": {"q": "x"}, "call_id": "c1"}
    ]
    assert call["metadata"]["generation_config"] == {"temperature": 0.2}


def test_no_tools_kwarg_asserts_empty_toolset(
    capturing: Path, read_captures: CaptureReader
) -> None:
    proxy = _wrap(_Client())

    @capture.agent(suite="wrap", redact=False, tools=[{"name": "session_tool", "input_schema": {}}])
    def agent() -> None:
        proxy.chat.completions.create(model="m", messages=[])

    agent()
    (call,) = _model_calls(read_captures("wrap")[0])
    assert call["tools_offered"] == []


def test_outside_session_is_inert(capturing: Path, read_captures: CaptureReader) -> None:
    proxy = _wrap(_Client())
    assert proxy.chat.completions.create(model="m", messages=[])["text"] == "hello"
    assert read_captures("wrap") == []


def test_provider_error_propagates_and_records_nothing(
    capturing: Path, read_captures: CaptureReader
) -> None:
    proxy = _wrap(_Client())

    @capture.agent(suite="wrap", redact=False, tools=[])
    def agent() -> None:
        proxy.chat.completions.create(model="m", messages=[], fail=True)

    with pytest.raises(RuntimeError, match="provider down"):
        agent()
    assert _model_calls(read_captures("wrap")[0]) == []


def test_bookkeeping_fault_is_fail_open(capturing: Path, read_captures: CaptureReader) -> None:
    def explode(_: Any) -> CallSpec:
        raise ValueError("bad describe")

    broken = Instrumentation(describe=explode, complete=_complete)
    proxy = ClientProxy(_Client(), {"chat": {"completions": {"create": broken}}})

    @capture.agent(suite="wrap", redact=False, tools=[])
    def agent() -> Any:
        return proxy.chat.completions.create(model="m", messages=[])

    assert agent()["text"] == "hello"
    assert _model_calls(read_captures("wrap")[0]) == []


def test_complete_fault_is_fail_open(capturing: Path, read_captures: CaptureReader) -> None:
    def explode(_: Any) -> Completion:
        raise ValueError("bad complete")

    broken = Instrumentation(describe=_describe, complete=explode)
    proxy = ClientProxy(_Client(), {"chat": {"completions": {"create": broken}}})

    @capture.agent(suite="wrap", redact=False, tools=[])
    def agent() -> Any:
        return proxy.chat.completions.create(model="m", messages=[])

    assert agent()["text"] == "hello"
    assert _model_calls(read_captures("wrap")[0]) == []


def test_describe_none_skips_recording(capturing: Path, read_captures: CaptureReader) -> None:
    skip = Instrumentation(describe=lambda kwargs: None, complete=_complete)
    proxy = ClientProxy(_Client(), {"chat": {"completions": {"create": skip}}})

    @capture.agent(suite="wrap", redact=False, tools=[])
    def agent() -> None:
        proxy.chat.completions.create(model="m", messages=[])

    agent()
    assert _model_calls(read_captures("wrap")[0]) == []


def test_explicit_requested_tool_calls_win_over_extraction(
    capturing: Path, read_captures: CaptureReader
) -> None:
    explicit = Instrumentation(
        describe=_describe,
        complete=lambda response: Completion(output="o", requested_tool_calls=[]),
    )
    proxy = ClientProxy(_Client(), {"chat": {"completions": {"create": explicit}}})

    @capture.agent(suite="wrap", redact=False, tools=[])
    def agent() -> None:
        proxy.chat.completions.create(model="m", messages=[])

    agent()
    (call,) = _model_calls(read_captures("wrap")[0])
    assert call["requested_tool_calls"] == []


async def test_async_call_records(capturing: Path, read_captures: CaptureReader) -> None:
    proxy = _wrap(_Client())

    @capture.agent(suite="wrap", redact=False, tools=[])
    async def agent() -> Any:
        return await proxy.chat.completions.acreate(model="m-async", messages=[])

    assert (await agent())["text"] == "async hello"
    (call,) = _model_calls(read_captures("wrap")[0])
    assert call["model_id"] == "m-async"
    assert call["output"] == "async hello"
    assert call["input_tokens"] == 2


# --- streaming --------------------------------------------------------------------------------


def test_sync_stream_records_on_exhaustion(capturing: Path, read_captures: CaptureReader) -> None:
    proxy = _wrap(_Client())

    @capture.agent(suite="wrap", redact=False, tools=[])
    def agent() -> list[str]:
        stream = proxy.chat.completions.create(model="m", messages=[], stream=True)
        assert isinstance(stream, StreamProxy)
        return [chunk["text"] for chunk in stream]

    assert agent() == ["a", "b"]
    (call,) = _model_calls(read_captures("wrap")[0])
    assert call["output"] == "ab"
    assert call["input_tokens"] == 3
    assert call["output_tokens"] == 2


def test_sync_stream_records_once_when_closed_early(
    capturing: Path, read_captures: CaptureReader
) -> None:
    proxy = _wrap(_Client())

    @capture.agent(suite="wrap", redact=False, tools=[])
    def agent() -> None:
        with proxy.chat.completions.create(model="m", messages=[], stream=True) as stream:
            for chunk in stream:
                if chunk["text"] == "a":
                    break
        stream.close()  # a second finish must not record twice
        assert unwrap_stream(stream).closed
        assert unwrap_stream(stream).entered

    agent()
    (call,) = _model_calls(read_captures("wrap")[0])
    assert call["output"] == "a"
    assert call["output_tokens"] == 0  # usage only arrives on the final chunk


def unwrap_stream(stream: Any) -> Any:
    return stream._stream


def test_sync_stream_records_partial_output_on_error(
    capturing: Path, read_captures: CaptureReader
) -> None:
    class _Boom(_Stream):
        def __iter__(self) -> Iterator[dict[str, Any]]:
            yield {"text": "partial", "usage": None}
            raise ConnectionError("dropped")

    inst = Instrumentation(
        describe=_describe, complete=_complete, is_stream=lambda k, r: True, on_chunk=_on_chunk
    )

    class _C:
        def create(self, **kwargs: Any) -> Any:
            return _Boom([])

    proxy = ClientProxy(_C(), {"create": inst})

    @capture.agent(suite="wrap", redact=False, tools=[])
    def agent() -> None:
        for _ in proxy.create(model="m"):
            pass

    with pytest.raises(ConnectionError):
        agent()
    (call,) = _model_calls(read_captures("wrap")[0])
    assert call["output"] == "partial"


def test_stream_end_hook_and_forwarding(capturing: Path, read_captures: CaptureReader) -> None:
    def at_end(state: StreamState) -> None:
        state.requested_tool_calls = [{"name": "t", "arguments": {}, "call_id": None}]

    inst = Instrumentation(
        describe=_describe,
        complete=_complete,
        is_stream=lambda k, r: True,
        on_chunk=_on_chunk,
        on_stream_end=at_end,
    )

    class _C:
        def create(self, **kwargs: Any) -> Any:
            stream = _Stream([{"text": "z", "usage": (1, 1)}])
            return stream

    proxy = ClientProxy(_C(), {"create": inst})

    @capture.agent(suite="wrap", redact=False, tools=[])
    def agent() -> None:
        stream = proxy.create(model="m")
        assert stream.closed is False  # attribute forwarded from the inner stream
        list(stream)

    agent()
    (call,) = _model_calls(read_captures("wrap")[0])
    assert call["requested_tool_calls"] == [{"name": "t", "arguments": {}, "call_id": None}]


async def test_async_stream_records(capturing: Path, read_captures: CaptureReader) -> None:
    proxy = _wrap(_Client())

    @capture.agent(suite="wrap", redact=False, tools=[])
    async def agent() -> list[str]:
        stream = await proxy.chat.completions.acreate(model="m", messages=[], stream=True)
        assert isinstance(stream, AsyncStreamProxy)
        out = [chunk["text"] async for chunk in stream]
        await stream.aclose()
        return out

    assert await agent() == ["x", "y"]
    (call,) = _model_calls(read_captures("wrap")[0])
    assert call["output"] == "xy"
    assert call["output_tokens"] == 1


def test_stream_outside_session_is_plain_passthrough(capturing: Path) -> None:
    proxy = _wrap(_Client())
    stream = proxy.chat.completions.create(model="m", messages=[], stream=True)
    assert [c["text"] for c in stream] == ["a", "b"]
