"""The OpenAI wrapper (`adapters/openai.py`): Chat Completions and Responses, sync/async/stream.

Responses are built from the real ``openai`` package's pydantic types so the attribute paths the
wrapper duck-types are the ones a live client returns; the *client* is synthetic so no test needs
the network. Two smoke tests at the bottom run the real ``openai.OpenAI`` / ``AsyncOpenAI`` over
a stub HTTP transport (guarded by ``importorskip``: the transport package is the one thing that
varies across ``openai`` releases).
"""

from __future__ import annotations

import functools
import json
import subprocess
import sys
from collections.abc import AsyncIterator, Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.responses import Response, ResponseCompletedEvent, ResponseTextDeltaEvent

from evalshift import capture
from evalshift.adapters._wrap import AsyncStreamProxy, StreamProxy, unwrap
from evalshift.adapters.openai import wrap_openai
from tests.conftest import CaptureReader


def _model_calls(capture_dict: dict[str, Any]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = capture_dict["trace"]["events"]
    return [e for e in events if e["type"] == "model_call"]


def _only_call(read_captures: CaptureReader, suite: str = "oai") -> dict[str, Any]:
    (call,) = _model_calls(read_captures(suite)[0])
    return call


# --- canned provider payloads ------------------------------------------------------------------

_USAGE = {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15}

CHAT_TEXT: dict[str, Any] = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "created": 1,
    "model": "gpt-4o-2024-08-06",
    "choices": [
        {
            "index": 0,
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": "Hello!"},
        }
    ],
    "usage": _USAGE,
}

CHAT_TOOL: dict[str, Any] = {
    "id": "chatcmpl-2",
    "object": "chat.completion",
    "created": 1,
    "model": "gpt-4o-2024-08-06",
    "choices": [
        {
            "index": 0,
            "finish_reason": "tool_calls",
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "search", "arguments": '{"q": "x"}'},
                    }
                ],
            },
        }
    ],
    "usage": _USAGE,
}

CHAT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search the web",
            "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
        },
    }
]


def _chunk(delta: dict[str, Any], **extra: Any) -> ChatCompletionChunk:
    choices = extra.pop("choices", [{"index": 0, "delta": delta, "finish_reason": None}])
    return ChatCompletionChunk.model_validate(
        {
            "id": "chatcmpl-s",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-4o-2024-08-06",
            "choices": choices,
            **extra,
        }
    )


def _text_chunks(*, with_usage: bool) -> list[ChatCompletionChunk]:
    chunks = [
        _chunk({"role": "assistant", "content": ""}),
        _chunk({"content": "Hel"}),
        _chunk({"content": "lo"}),
        _chunk({}, choices=[{"index": 0, "delta": {}, "finish_reason": "stop"}]),
    ]
    if with_usage:
        chunks.append(_chunk({}, choices=[], usage=_USAGE))
    return chunks


def _tool_delta(index: int, **function: Any) -> dict[str, Any]:
    call: dict[str, Any] = {"index": index, "function": function}
    if "id" in function:
        call["id"] = function.pop("id")
        call["type"] = "function"
    return {"tool_calls": [call]}


def _tool_chunks() -> list[ChatCompletionChunk]:
    return [
        _chunk({"role": "assistant", "content": None}),
        _chunk(_tool_delta(0, id="call_1", name="search", arguments="")),
        _chunk(_tool_delta(0, arguments='{"q":')),
        _chunk(_tool_delta(0, arguments=' "x"}')),
        _chunk(_tool_delta(1, id="call_2", name="lookup", arguments="{}")),
        _chunk({}, choices=[{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]),
        _chunk({}, choices=[], usage=_USAGE),
    ]


RESPONSE: dict[str, Any] = {
    "id": "resp_1",
    "object": "response",
    "created_at": 1.0,
    "model": "gpt-4.1",
    "output": [
        {
            "type": "message",
            "id": "msg_1",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "hello from responses", "annotations": []}],
        },
        {
            "type": "function_call",
            "id": "fc_1",
            "call_id": "call_9",
            "name": "search",
            "arguments": '{"q": "y"}',
            "status": "completed",
        },
    ],
    "parallel_tool_calls": True,
    "tool_choice": "auto",
    "tools": [],
    "usage": {
        "input_tokens": 7,
        "output_tokens": 4,
        "total_tokens": 11,
        "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
        "output_tokens_details": {"reasoning_tokens": 0},
    },
}

RESPONSES_TOOLS = [
    {
        "type": "function",
        "name": "search",
        "description": "Search the web",
        "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
        "strict": True,
    }
]


def _response_events(*, completed: bool = True) -> list[Any]:
    def delta(text: str, seq: int) -> ResponseTextDeltaEvent:
        return ResponseTextDeltaEvent.model_validate(
            {
                "type": "response.output_text.delta",
                "delta": text,
                "item_id": "msg_1",
                "output_index": 0,
                "content_index": 0,
                "sequence_number": seq,
                "logprobs": [],
            }
        )

    events: list[Any] = [delta("hello ", 1), delta("from responses", 2)]
    if completed:
        events.append(
            ResponseCompletedEvent.model_validate(
                {"type": "response.completed", "sequence_number": 3, "response": RESPONSE}
            )
        )
    return events


# --- a synthetic client with the real attribute paths -----------------------------------------


class _Stream:
    def __init__(self, chunks: list[Any]) -> None:
        self._chunks = chunks
        self.closed = False

    def __iter__(self) -> Iterator[Any]:
        yield from self._chunks

    def __enter__(self) -> _Stream:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self.closed = True


class _AsyncStream:
    def __init__(self, chunks: list[Any]) -> None:
        self._chunks = chunks
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[Any]:
        for chunk in self._chunks:
            yield chunk

    async def close(self) -> None:
        self.closed = True


Responder = Callable[[dict[str, Any]], Any]


class _Endpoint:
    def __init__(self, respond: Responder) -> None:
        self._respond = respond
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self._respond(kwargs)


class _AsyncEndpoint(_Endpoint):
    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self._respond(kwargs)


class _Namespace:
    def __init__(self, **attrs: Any) -> None:
        for name, value in attrs.items():
            setattr(self, name, value)


def _client(
    chat: Responder = lambda kwargs: ChatCompletion.model_validate(CHAT_TEXT),
    responses: Responder = lambda kwargs: Response.model_validate(RESPONSE),
    *,
    is_async: bool = False,
) -> Any:
    endpoint = _AsyncEndpoint if is_async else _Endpoint
    return _Namespace(
        chat=_Namespace(completions=endpoint(chat)),
        responses=endpoint(responses),
        embeddings=_Namespace(create=lambda **kwargs: "embedding"),
        api_key="sk-test",
    )


def _stream_of(chunks: list[Any]) -> Responder:
    return lambda kwargs: _Stream(chunks)


def _async_stream_of(chunks: list[Any]) -> Responder:
    return lambda kwargs: _AsyncStream(chunks)


MESSAGES = [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}]


# --- chat completions: non-streaming ----------------------------------------------------------


def test_chat_create_records_one_model_call(capturing: Path, read_captures: CaptureReader) -> None:
    raw = _client()
    client = wrap_openai(raw)

    @capture.agent(suite="oai", redact=False, tools=[])
    def agent() -> Any:
        return client.chat.completions.create(
            model="gpt-4o", messages=MESSAGES, tools=CHAT_TOOLS, temperature=0.1, seed=7
        )

    response = agent()
    assert response.choices[0].message.content == "Hello!"
    assert raw.chat.completions.calls[0]["seed"] == 7

    call = _only_call(read_captures)
    assert call["model_id"] == "gpt-4o"
    assert call["input"] == MESSAGES
    assert call["output"] == "Hello!"
    assert call["input_tokens"] == 12
    assert call["output_tokens"] == 3
    assert call["latency_ms"] >= 0
    assert call["tools_offered"] == ["search"]
    assert call["requested_tool_calls"] == []
    assert call["metadata"]["generation_config"] == {"temperature": 0.1}


async def test_chat_create_async(capturing: Path, read_captures: CaptureReader) -> None:
    client = wrap_openai(_client(is_async=True))

    @capture.agent(suite="oai", redact=False, tools=[])
    async def agent() -> Any:
        return await client.chat.completions.create(model="gpt-4o", messages=MESSAGES)

    assert (await agent()).choices[0].message.content == "Hello!"
    call = _only_call(read_captures)
    assert call["output"] == "Hello!"
    assert call["input_tokens"] == 12


def test_chat_tool_calls_are_extracted(capturing: Path, read_captures: CaptureReader) -> None:
    client = wrap_openai(_client(chat=lambda kwargs: ChatCompletion.model_validate(CHAT_TOOL)))

    @capture.agent(suite="oai", redact=False, tools=[])
    def agent() -> None:
        client.chat.completions.create(model="gpt-4o", messages=MESSAGES, tools=CHAT_TOOLS)

    agent()
    call = _only_call(read_captures)
    assert call["output"] == ""
    assert call["requested_tool_calls"] == [
        {"name": "search", "arguments": {"q": "x"}, "call_id": "call_1"}
    ]


def test_no_tools_kwarg_asserts_empty_toolset(
    capturing: Path, read_captures: CaptureReader
) -> None:
    client = wrap_openai(_client())

    @capture.agent(suite="oai", redact=False, tools=[{"name": "session_tool", "input_schema": {}}])
    def agent() -> None:
        client.chat.completions.create(model="gpt-4o", messages=MESSAGES)

    agent()
    assert _only_call(read_captures)["tools_offered"] == []


def test_plain_dict_response_is_recorded(capturing: Path, read_captures: CaptureReader) -> None:
    """A user's mock returning the JSON dict (not a pydantic object) records the same call."""
    client = wrap_openai(_client(chat=lambda kwargs: CHAT_TOOL))

    @capture.agent(suite="oai", redact=False, tools=[])
    def agent() -> None:
        client.chat.completions.create(model="gpt-4o", messages=MESSAGES)

    agent()
    call = _only_call(read_captures)
    assert call["input_tokens"] == 12
    assert call["requested_tool_calls"][0]["name"] == "search"


def test_unexpected_shape_returns_to_caller_and_records_nothing(
    capturing: Path, read_captures: CaptureReader
) -> None:
    """Pinned: a response with no ``choices`` list is handed back untouched and not recorded.

    Recording it would assert "the model said nothing, for zero tokens" on no evidence.
    """
    weird = {"object": "something.else"}
    client = wrap_openai(_client(chat=lambda kwargs: weird))

    @capture.agent(suite="oai", redact=False, tools=[])
    def agent() -> Any:
        return client.chat.completions.create(model="gpt-4o", messages=MESSAGES)

    assert agent() is weird
    assert _model_calls(read_captures("oai")[0]) == []


def test_outside_session_is_inert(capturing: Path, read_captures: CaptureReader) -> None:
    client = wrap_openai(_client())
    response = client.chat.completions.create(model="gpt-4o", messages=MESSAGES)
    assert response.choices[0].message.content == "Hello!"
    assert read_captures("oai") == []


def test_unrelated_attributes_are_forwarded() -> None:
    raw = _client()
    client = wrap_openai(raw)
    assert client.embeddings.create(model="e", input="x") == "embedding"
    assert client.api_key == "sk-test"
    assert unwrap(client) is raw


# --- chat completions: streaming --------------------------------------------------------------


def test_chat_stream_records_text_and_final_usage(
    capturing: Path, read_captures: CaptureReader
) -> None:
    client = wrap_openai(_client(chat=_stream_of(_text_chunks(with_usage=True))))

    @capture.agent(suite="oai", redact=False, tools=[])
    def agent() -> str:
        stream = client.chat.completions.create(
            model="gpt-4o",
            messages=MESSAGES,
            stream=True,
            stream_options={"include_usage": True},
        )
        assert isinstance(stream, StreamProxy)
        assert stream.closed is False  # forwarded from the inner stream
        return "".join(c.choices[0].delta.content or "" for c in stream if c.choices)

    assert agent() == "Hello"
    call = _only_call(read_captures)
    assert call["output"] == "Hello"
    assert call["input_tokens"] == 12
    assert call["output_tokens"] == 3
    assert call["requested_tool_calls"] == []


def test_chat_stream_without_include_usage_records_zero_tokens(
    capturing: Path, read_captures: CaptureReader
) -> None:
    client = wrap_openai(_client(chat=_stream_of(_text_chunks(with_usage=False))))

    @capture.agent(suite="oai", redact=False, tools=[])
    def agent() -> None:
        with client.chat.completions.create(model="gpt-4o", messages=MESSAGES, stream=True) as s:
            for _ in s:
                pass

    agent()
    call = _only_call(read_captures)
    assert call["output"] == "Hello"
    assert call["input_tokens"] == 0
    assert call["output_tokens"] == 0


def test_chat_stream_assembles_tool_calls_from_deltas(
    capturing: Path, read_captures: CaptureReader
) -> None:
    client = wrap_openai(_client(chat=_stream_of(_tool_chunks())))

    @capture.agent(suite="oai", redact=False, tools=[])
    def agent() -> None:
        for _ in client.chat.completions.create(
            model="gpt-4o", messages=MESSAGES, tools=CHAT_TOOLS, stream=True
        ):
            pass

    agent()
    call = _only_call(read_captures)
    assert call["output"] == ""
    assert call["output_tokens"] == 3
    assert call["requested_tool_calls"] == [
        {"name": "search", "arguments": {"q": "x"}, "call_id": "call_1"},
        {"name": "lookup", "arguments": {}, "call_id": "call_2"},
    ]


def test_chat_stream_legacy_function_call_delta(
    capturing: Path, read_captures: CaptureReader
) -> None:
    chunks = [
        _chunk({"role": "assistant", "function_call": {"name": "search", "arguments": ""}}),
        _chunk({"function_call": {"arguments": '{"q": "z"}'}}),
    ]
    client = wrap_openai(_client(chat=_stream_of(chunks)))

    @capture.agent(suite="oai", redact=False, tools=[])
    def agent() -> None:
        list(client.chat.completions.create(model="gpt-4o", messages=MESSAGES, stream=True))

    agent()
    assert _only_call(read_captures)["requested_tool_calls"] == [
        {"name": "search", "arguments": {"q": "z"}, "call_id": None}
    ]


async def test_chat_stream_async(capturing: Path, read_captures: CaptureReader) -> None:
    client = wrap_openai(
        _client(chat=_async_stream_of(_text_chunks(with_usage=True)), is_async=True)
    )

    @capture.agent(suite="oai", redact=False, tools=[])
    async def agent() -> str:
        stream = await client.chat.completions.create(
            model="gpt-4o", messages=MESSAGES, stream=True
        )
        assert isinstance(stream, AsyncStreamProxy)
        parts = [c.choices[0].delta.content or "" async for c in stream if c.choices]
        await stream.aclose()
        return "".join(parts)

    assert await agent() == "Hello"
    call = _only_call(read_captures)
    assert call["output"] == "Hello"
    assert call["output_tokens"] == 3


# --- responses API ----------------------------------------------------------------------------


def test_responses_create_records_messages_style_input(
    capturing: Path, read_captures: CaptureReader
) -> None:
    client = wrap_openai(_client())

    @capture.agent(suite="oai", redact=False, tools=[])
    def agent() -> Any:
        return client.responses.create(
            model="gpt-4.1",
            instructions="be brief",
            input="hi",
            tools=RESPONSES_TOOLS,
            temperature=0.5,
        )

    assert agent().output_text == "hello from responses"
    call = _only_call(read_captures)
    assert call["model_id"] == "gpt-4.1"
    # ``instructions`` + a string ``input`` become the messages list the CLI already splits
    # into system prompt / current turn.
    assert call["input"] == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hi"},
    ]
    assert call["output"] == "hello from responses"
    assert call["input_tokens"] == 7
    assert call["output_tokens"] == 4
    assert call["tools_offered"] == ["search"]
    assert call["requested_tool_calls"] == [
        {"name": "search", "arguments": {"q": "y"}, "call_id": "call_9"}
    ]
    assert call["metadata"]["generation_config"] == {"temperature": 0.5}

    # The flat Responses tool shape was translated so the schema (and strict) survive.
    toolset_file = next((capturing / "toolsets").glob("*.json"))
    (tool,) = json.loads(toolset_file.read_text(encoding="utf-8"))["tools"]
    assert tool["name"] == "search"
    assert tool["input_schema"] == RESPONSES_TOOLS[0]["parameters"]
    assert tool["strict"] is True


def test_responses_list_input_is_passed_through(
    capturing: Path, read_captures: CaptureReader
) -> None:
    client = wrap_openai(_client())
    items = [
        {"role": "user", "content": "first"},
        {"type": "function_call_output", "call_id": "call_0", "output": "42"},
    ]

    @capture.agent(suite="oai", redact=False, tools=[])
    def agent() -> None:
        client.responses.create(model="gpt-4.1", input=items)

    agent()
    call = _only_call(read_captures)
    assert call["input"] == items  # no instructions -> no system message prepended
    assert call["tools_offered"] == []


def test_responses_dict_response_output_text_is_joined(
    capturing: Path, read_captures: CaptureReader
) -> None:
    client = wrap_openai(_client(responses=lambda kwargs: RESPONSE))

    @capture.agent(suite="oai", redact=False, tools=[])
    def agent() -> None:
        client.responses.create(model="gpt-4.1", input="hi")

    agent()
    call = _only_call(read_captures)
    assert call["output"] == "hello from responses"
    assert call["input_tokens"] == 7


def test_responses_stream_records_on_completed_event(
    capturing: Path, read_captures: CaptureReader
) -> None:
    client = wrap_openai(_client(responses=_stream_of(_response_events())))

    @capture.agent(suite="oai", redact=False, tools=[])
    def agent() -> list[str]:
        stream = client.responses.create(model="gpt-4.1", input="hi", stream=True)
        return [e.type for e in stream]

    assert agent() == [
        "response.output_text.delta",
        "response.output_text.delta",
        "response.completed",
    ]
    call = _only_call(read_captures)
    assert call["output"] == "hello from responses"
    assert call["input_tokens"] == 7
    assert call["output_tokens"] == 4
    assert call["requested_tool_calls"] == [
        {"name": "search", "arguments": {"q": "y"}, "call_id": "call_9"}
    ]


def test_responses_stream_without_completed_event_records_no_requested_calls(
    capturing: Path, read_captures: CaptureReader
) -> None:
    client = wrap_openai(_client(responses=_stream_of(_response_events(completed=False))))

    @capture.agent(suite="oai", redact=False, tools=[])
    def agent() -> None:
        list(client.responses.create(model="gpt-4.1", input="hi", stream=True))

    agent()
    call = _only_call(read_captures)
    assert call["output"] == "hello from responses"
    assert call["output_tokens"] == 0
    assert call["requested_tool_calls"] is None  # honest "not recorded", never ``[]``


async def test_responses_stream_async(capturing: Path, read_captures: CaptureReader) -> None:
    client = wrap_openai(_client(responses=_async_stream_of(_response_events()), is_async=True))

    @capture.agent(suite="oai", redact=False, tools=[])
    async def agent() -> None:
        stream = await client.responses.create(model="gpt-4.1", input="hi", stream=True)
        async for _ in stream:
            pass

    await agent()
    assert _only_call(read_captures)["output"] == "hello from responses"


# --- packaging --------------------------------------------------------------------------------


def test_module_imports_without_openai_installed() -> None:
    """The module never imports ``openai`` (D-deps): it duck-types everything."""
    source = Path(wrap_openai.__code__.co_filename).read_text(encoding="utf-8")
    assert not any(
        line.startswith(("import openai", "from openai")) for line in source.splitlines()
    )
    probe = (
        "import sys; sys.modules['openai'] = None; "
        "import evalshift.adapters.openai as m; assert callable(m.wrap_openai)"
    )
    subprocess.run([sys.executable, "-c", probe], check=True)


def test_sync_wrapper_around_coroutine_function_takes_async_path(
    capturing: Path, read_captures: CaptureReader
) -> None:
    """``openai``'s ``@required_args`` is a plain ``functools.wraps`` wrapper around the async
    ``create``; the wrapper must still see through it and await the call."""

    def required_args(fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return fn(*args, **kwargs)

        return wrapper

    class _Completions:
        @required_args
        async def create(self, **kwargs: Any) -> Any:
            return ChatCompletion.model_validate(CHAT_TEXT)

    raw: Any = _Namespace(chat=_Namespace(completions=_Completions()))
    client = wrap_openai(raw)

    @capture.agent(suite="oai", redact=False, tools=[])
    async def agent() -> Any:
        return await client.chat.completions.create(model="gpt-4o", messages=MESSAGES)

    import asyncio

    assert asyncio.run(agent()).choices[0].message.content == "Hello!"
    assert _only_call(read_captures)["output"] == "Hello!"


# --- real client over a stub transport ---------------------------------------------------------


def _stub_transport_module() -> Any:
    """The HTTP package the installed ``openai`` drives (``httpx2`` since 3.x, ``httpx`` before)."""
    import openai._base_client as base

    if getattr(base, "httpx2", None) is not None:
        return pytest.importorskip("httpx2")
    return pytest.importorskip("httpx")


def test_real_openai_client_smoke(capturing: Path, read_captures: CaptureReader) -> None:
    openai = pytest.importorskip("openai")
    http = _stub_transport_module()
    seen: list[Any] = []

    def handler(request: Any) -> Any:
        seen.append(json.loads(request.content))
        return http.Response(200, json=CHAT_TEXT)

    raw = openai.OpenAI(
        api_key="test", http_client=http.Client(transport=http.MockTransport(handler))
    )
    client = wrap_openai(raw)

    @capture.agent(suite="oai", redact=False, tools=[])
    def agent() -> str:
        response = client.chat.completions.create(
            model="gpt-4o", messages=MESSAGES, tools=CHAT_TOOLS, temperature=0
        )
        return str(response.choices[0].message.content)

    assert agent() == "Hello!"
    assert seen[0]["model"] == "gpt-4o"
    call = _only_call(read_captures)
    assert call["model_id"] == "gpt-4o"
    assert call["input"] == MESSAGES
    assert call["output"] == "Hello!"
    assert call["input_tokens"] == 12
    assert call["tools_offered"] == ["search"]
    assert call["requested_tool_calls"] == []
    assert call["metadata"]["generation_config"] == {"temperature": 0}


async def test_real_async_openai_client_smoke(
    capturing: Path, read_captures: CaptureReader
) -> None:
    openai = pytest.importorskip("openai")
    http = _stub_transport_module()

    def handler(request: Any) -> Any:
        return http.Response(200, json=CHAT_TOOL)

    raw = openai.AsyncOpenAI(
        api_key="test", http_client=http.AsyncClient(transport=http.MockTransport(handler))
    )
    client = wrap_openai(raw)

    @capture.agent(suite="oai", redact=False, tools=[])
    async def agent() -> Any:
        return await client.chat.completions.create(model="gpt-4o", messages=MESSAGES)

    response = await agent()
    assert response.choices[0].message.tool_calls[0].function.name == "search"
    call = _only_call(read_captures)
    assert call["requested_tool_calls"] == [
        {"name": "search", "arguments": {"q": "x"}, "call_id": "call_1"}
    ]
