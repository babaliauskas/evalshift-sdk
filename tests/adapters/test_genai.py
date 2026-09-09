"""The Google GenAI wrapper (`adapters/genai.py`): ``models.generate_content`` and its stream /
async forms.

Responses and configs are built from the real ``google.genai.types`` pydantic classes so the
attribute paths the wrapper duck-types are the ones a live client returns; the *client* is
synthetic (``models`` / ``aio.models`` attribute paths) so no test needs the network. One smoke
test at the bottom runs the real ``google.genai.Client`` against a local ``http.server`` stub.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from collections.abc import AsyncIterator, Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, ClassVar

import pytest
from google.genai import types

from evalshift import capture
from evalshift.adapters._wrap import AsyncStreamProxy, StreamProxy, unwrap
from evalshift.adapters.genai import wrap_genai
from tests.conftest import CaptureReader

SUITE = "genai"


def _model_calls(capture_dict: dict[str, Any]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = capture_dict["trace"]["events"]
    return [e for e in events if e["type"] == "model_call"]


def _only_call(read_captures: CaptureReader) -> dict[str, Any]:
    (call,) = _model_calls(read_captures(SUITE)[0])
    return call


# --- canned provider payloads ------------------------------------------------------------------

USAGE = types.GenerateContentResponseUsageMetadata(
    prompt_token_count=12, candidates_token_count=3, total_token_count=15
)

WEATHER_TOOL = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="get_weather",
            description="Current weather",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={"city": types.Schema(type=types.Type.STRING)},
                required=["city"],
            ),
        )
    ]
)


def _response(
    *parts: types.Part, usage: types.GenerateContentResponseUsageMetadata | None = USAGE
) -> types.GenerateContentResponse:
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(role="model", parts=list(parts)),
                finish_reason=types.FinishReason.STOP,
            )
        ],
        usage_metadata=usage,
    )


TEXT_RESPONSE = _response(types.Part(text="Hello!"))
CALL_RESPONSE = _response(
    types.Part(function_call=types.FunctionCall(name="get_weather", args={"city": "Paris"}))
)


def _text_chunks() -> list[types.GenerateContentResponse]:
    return [
        _response(types.Part(text="Hel"), usage=None),
        _response(types.Part(text="lo"), usage=None),
        _response(types.Part(text="!"), usage=USAGE),
    ]


def _call_chunks() -> list[types.GenerateContentResponse]:
    return [
        _response(
            types.Part(
                function_call=types.FunctionCall(
                    id="fc-1", name="get_weather", args={"city": "Paris"}
                )
            ),
            usage=None,
        ),
        _response(
            types.Part(function_call=types.FunctionCall(name="get_weather", args={"city": "Oslo"})),
            usage=USAGE,
        ),
    ]


# --- synthetic client ---------------------------------------------------------------------------


class _Namespace:
    def __init__(self, **attrs: Any) -> None:
        for name, value in attrs.items():
            setattr(self, name, value)


class _Models:
    """Stands in for ``client.models``; records the kwargs each call received."""

    def __init__(self, response: Any = TEXT_RESPONSE, chunks: list[Any] | None = None) -> None:
        self.response = response
        self.chunks = chunks if chunks is not None else _text_chunks()
        self.calls: list[dict[str, Any]] = []

    def generate_content(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self.response

    def generate_content_stream(self, **kwargs: Any) -> Iterator[Any]:
        self.calls.append(kwargs)
        yield from self.chunks

    def embed_content(self, **kwargs: Any) -> str:
        return "embedding"


class _AsyncModels(_Models):
    async def generate_content(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self.response

    async def generate_content_stream(  # type: ignore[override]
        self, **kwargs: Any
    ) -> AsyncIterator[Any]:
        self.calls.append(kwargs)

        async def gen() -> AsyncIterator[Any]:
            for chunk in self.chunks:
                yield chunk

        return gen()


def _client(models: _Models | None = None, aio: _AsyncModels | None = None) -> Any:
    raw = _Namespace(
        models=models or _Models(),
        aio=_Namespace(models=aio or _AsyncModels()),
        chats=_Namespace(create=lambda **kw: "chat"),
        files=_Namespace(upload=lambda **kw: "file"),
        vertexai=False,
    )
    return wrap_genai(raw)


def _run(client: Any, **kwargs: Any) -> Any:
    kwargs.setdefault("model", "gemini-2.0-flash")
    kwargs.setdefault("contents", "hi")

    @capture.agent(suite=SUITE, redact=False, tools=[])
    def agent() -> Any:
        return client.models.generate_content(**kwargs)

    return agent()


# --- non-streaming ------------------------------------------------------------------------------


def test_generate_content_records_one_model_call(
    capturing: Path, read_captures: CaptureReader
) -> None:
    client = _client()
    response = _run(
        client,
        contents="hi",
        config=types.GenerateContentConfig(temperature=0.0, tools=[WEATHER_TOOL]),
    )
    assert response is TEXT_RESPONSE
    call = _only_call(read_captures)
    assert call["model_id"] == "gemini-2.0-flash"
    assert call["input"] == [{"role": "user", "content": "hi"}]
    assert call["output"] == "Hello!"
    assert call["input_tokens"] == 12
    assert call["output_tokens"] == 3
    assert call["latency_ms"] >= 0
    assert call["tools_offered"] == ["get_weather"]
    assert call["requested_tool_calls"] == []
    assert call["metadata"]["generation_config"] == {"temperature": 0.0}


async def test_generate_content_async(capturing: Path, read_captures: CaptureReader) -> None:
    client = _client()

    @capture.agent(suite=SUITE, redact=False, tools=[])
    async def agent() -> Any:
        return await client.aio.models.generate_content(
            model="gemini-2.0-flash", contents="hi", config={"temperature": 0.2}
        )

    assert (await agent()) is TEXT_RESPONSE
    call = _only_call(read_captures)
    assert call["output"] == "Hello!"
    assert call["input_tokens"] == 12
    assert call["tools_offered"] == []
    assert call["metadata"]["generation_config"] == {"temperature": 0.2}


def test_function_calls_are_extracted(capturing: Path, read_captures: CaptureReader) -> None:
    client = _client(_Models(response=CALL_RESPONSE))
    _run(client, config=types.GenerateContentConfig(tools=[WEATHER_TOOL]))
    call = _only_call(read_captures)
    assert call["output"] == ""
    assert call["requested_tool_calls"] == [
        {"name": "get_weather", "arguments": {"city": "Paris"}, "call_id": None}
    ]


def test_no_tools_asserts_empty_toolset(capturing: Path, read_captures: CaptureReader) -> None:
    _run(_client())
    call = _only_call(read_captures)
    assert call["tools_offered"] == []
    assert "generation_config" not in call["metadata"]


def test_config_as_dict(capturing: Path, read_captures: CaptureReader) -> None:
    _run(
        _client(),
        config={
            "temperature": 0.5,
            "system_instruction": "Be brief.",
            "tools": [
                {
                    "function_declarations": [
                        {
                            "name": "lookup",
                            "description": "Look it up",
                            "parameters": {"type": "OBJECT", "properties": {}},
                        }
                    ]
                }
            ],
        },
    )
    call = _only_call(read_captures)
    assert call["tools_offered"] == ["lookup"]
    assert call["input"] == [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "hi"},
    ]
    assert call["metadata"]["generation_config"] == {"temperature": 0.5}


def test_generation_config_keeps_gemini_spellings(
    capturing: Path, read_captures: CaptureReader
) -> None:
    _run(
        _client(),
        config=types.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=64,
            response_mime_type="application/json",
            tools=[WEATHER_TOOL],
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode=types.FunctionCallingConfigMode.ANY,
                    allowed_function_names=["get_weather"],
                )
            ),
        ),
    )
    call = _only_call(read_captures)
    assert call["metadata"]["generation_config"] == {
        "temperature": 0.1,
        "max_output_tokens": 64,
        "response_mime_type": "application/json",
        "tool_config": {
            "function_calling_config": {"mode": "ANY", "allowed_function_names": ["get_weather"]}
        },
    }


def test_thought_parts_are_not_output(capturing: Path, read_captures: CaptureReader) -> None:
    thinking = _response(types.Part(text="hmm", thought=True), types.Part(text="answer"))
    _run(_client(_Models(response=thinking)))
    assert _only_call(read_captures)["output"] == "answer"


def test_plain_dict_response_is_recorded(capturing: Path, read_captures: CaptureReader) -> None:
    response = {
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [
                        {"text": "Sure."},
                        {"function_call": {"name": "get_weather", "args": {"city": "Rome"}}},
                    ],
                }
            }
        ],
        "usage_metadata": {"prompt_token_count": 5, "candidates_token_count": 2},
    }
    _run(_client(_Models(response=response)))
    call = _only_call(read_captures)
    assert call["output"] == "Sure."
    assert call["input_tokens"] == 5
    assert call["output_tokens"] == 2
    assert call["requested_tool_calls"] == [
        {"name": "get_weather", "arguments": {"city": "Rome"}, "call_id": None}
    ]


def test_unexpected_shape_returns_to_caller_and_records_nothing(
    capturing: Path, read_captures: CaptureReader
) -> None:
    odd = object()
    assert _run(_client(_Models(response=odd))) is odd
    assert _model_calls(read_captures(SUITE)[0]) == []


def test_exploding_config_still_records_the_call(
    capturing: Path, read_captures: CaptureReader
) -> None:
    class Boom:
        tools: ClassVar[list[Any]] = [WEATHER_TOOL]
        system_instruction = None

        def model_dump(self, **kwargs: Any) -> Any:
            raise RuntimeError("no dump for you")

    _run(_client(), config=Boom())
    call = _only_call(read_captures)
    assert call["output"] == "Hello!"
    assert call["tools_offered"] == ["get_weather"]
    assert "generation_config" not in call["metadata"]


def test_outside_session_is_inert(capturing: Path, read_captures: CaptureReader) -> None:
    client = _client()
    assert client.models.generate_content(model="gemini-2.0-flash", contents="hi") is TEXT_RESPONSE
    assert read_captures(SUITE) == []


def test_unrelated_attributes_are_forwarded() -> None:
    models = _Models()
    client = _client(models)
    assert client.chats.create(model="m") == "chat"
    assert client.files.upload(file="x") == "file"
    assert client.models.embed_content(contents="x") == "embedding"
    assert client.vertexai is False
    assert unwrap(client).models is models
    assert models.calls == []


def test_calls_reach_the_real_method_with_kwargs_intact() -> None:
    models = _Models()
    client = _client(models)
    config = types.GenerateContentConfig(temperature=0.3)
    client.models.generate_content(model="m", contents=["a", "b"], config=config)
    assert models.calls == [{"model": "m", "contents": ["a", "b"], "config": config}]


# --- input shapes -------------------------------------------------------------------------------


def test_content_objects_become_messages_with_tool_calls_and_results(
    capturing: Path, read_captures: CaptureReader
) -> None:
    history = [
        types.Content(role="user", parts=[types.Part(text="Weather in Paris?")]),
        types.Content(
            role="model",
            parts=[
                types.Part(text="Checking."),
                types.Part(
                    function_call=types.FunctionCall(
                        id="fc-1", name="get_weather", args={"city": "Paris"}
                    )
                ),
            ],
        ),
        types.Content(
            role="user",
            parts=[
                types.Part(
                    function_response=types.FunctionResponse(
                        id="fc-1", name="get_weather", response={"temp_c": 21}
                    )
                )
            ],
        ),
        types.Content(role="user", parts=[types.Part(text="And Oslo?")]),
    ]
    _run(
        _client(),
        contents=history,
        config=types.GenerateContentConfig(
            system_instruction=types.Content(parts=[types.Part(text="You are terse.")]),
            tools=[WEATHER_TOOL],
        ),
    )
    assert _only_call(read_captures)["input"] == [
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "Weather in Paris?"},
        {
            "role": "model",
            "content": "Checking.",
            "tool_calls": [{"id": "fc-1", "name": "get_weather", "arguments": {"city": "Paris"}}],
        },
        {
            "role": "tool",
            "content": '{"temp_c": 21}',
            "tool_call_id": "fc-1",
            "name": "get_weather",
        },
        {"role": "user", "content": "And Oslo?"},
    ]


def test_loose_parts_fold_into_one_user_message(
    capturing: Path, read_captures: CaptureReader
) -> None:
    """A list of strings / ``Part`` objects is one user turn, as the SDK's ``t_contents`` groups
    them; a system instruction given as a list of strings is one system message."""
    _run(
        _client(),
        contents=["Summarise:", types.Part(text="the text"), {"text": "please"}],
        config={"system_instruction": ["Rule one.", "Rule two."]},
    )
    assert _only_call(read_captures)["input"] == [
        {"role": "system", "content": "Rule one.\nRule two."},
        {"role": "user", "content": "Summarise:\nthe text\nplease"},
    ]


def test_content_dicts_are_read_like_objects(capturing: Path, read_captures: CaptureReader) -> None:
    _run(
        _client(),
        contents=[
            {"role": "user", "parts": [{"text": "hi"}]},
            {"role": "model", "parts": [{"text": "hello"}]},
            {"role": "user", "parts": [{"text": "bye"}]},
        ],
    )
    assert _only_call(read_captures)["input"] == [
        {"role": "user", "content": "hi"},
        {"role": "model", "content": "hello"},
        {"role": "user", "content": "bye"},
    ]


# --- tools --------------------------------------------------------------------------------------


def test_python_callable_tools_are_declared_as_json_schema(
    capturing: Path, read_captures: CaptureReader
) -> None:
    def add(a: int, b: int) -> int:
        """Add two ints."""
        return a + b

    afc_response = _response(types.Part(text="3"))
    afc_response.automatic_function_calling_history = [
        types.Content(role="user", parts=[types.Part(text="1+2?")])
    ]
    _run(
        _client(_Models(response=afc_response)),
        contents="1+2?",
        config=types.GenerateContentConfig(tools=[add, WEATHER_TOOL]),
    )
    call = _only_call(read_captures)
    assert call["tools_offered"] == ["add", "get_weather"]
    assert call["requested_tool_calls"] == []  # the SDK ran the call itself; the final turn is text
    toolset_file = next((capturing / "toolsets").glob("*.json"))
    toolset = json.loads(toolset_file.read_text(encoding="utf-8"))
    (add_tool,) = [t for t in toolset["tools"] if t["name"] == "add"]
    assert add_tool["description"] == "Add two ints."
    assert add_tool["input_schema"]["type"] == "object"
    assert set(add_tool["input_schema"]["properties"]) == {"a", "b"}


def test_json_schema_declarations_keep_their_schema(
    capturing: Path, read_captures: CaptureReader
) -> None:
    tool = types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="lookup",
                parameters_json_schema={"type": "object", "properties": {"q": {"type": "string"}}},
            )
        ]
    )
    _run(_client(), config=types.GenerateContentConfig(tools=[tool]))
    assert _only_call(read_captures)["tools_offered"] == ["lookup"]
    toolset = json.loads(next((capturing / "toolsets").glob("*.json")).read_text(encoding="utf-8"))
    assert toolset["tools"][0]["input_schema"] == {
        "type": "object",
        "properties": {"q": {"type": "string"}},
    }


def test_builtin_tools_do_not_normalise(capturing: Path, read_captures: CaptureReader) -> None:
    """A ``google_search`` Tool has no function declarations: the toolset is not stamped at all
    (never guessed as ``[]``), exactly as ``record_model_call`` treats any unrecognised value."""
    _run(
        _client(),
        config=types.GenerateContentConfig(tools=[types.Tool(google_search=types.GoogleSearch())]),
    )
    call = _only_call(read_captures)
    assert call["tools_offered"] is None
    assert call["toolset_ref"] is None
    assert call["output"] == "Hello!"


# --- streaming ----------------------------------------------------------------------------------


def test_stream_records_text_and_last_usage(capturing: Path, read_captures: CaptureReader) -> None:
    client = _client()

    @capture.agent(suite=SUITE, redact=False, tools=[])
    def agent() -> list[str]:
        stream = client.models.generate_content_stream(model="gemini-2.0-flash", contents="hi")
        assert isinstance(stream, StreamProxy)
        return [chunk.text for chunk in stream]

    assert agent() == ["Hel", "lo", "!"]
    call = _only_call(read_captures)
    assert call["output"] == "Hello!"
    assert call["input_tokens"] == 12
    assert call["output_tokens"] == 3
    assert call["requested_tool_calls"] == []
    assert call["tools_offered"] == []


def test_stream_collects_function_calls_across_chunks(
    capturing: Path, read_captures: CaptureReader
) -> None:
    client = _client(_Models(chunks=_call_chunks()))

    @capture.agent(suite=SUITE, redact=False, tools=[])
    def agent() -> int:
        stream = client.models.generate_content_stream(
            model="gemini-2.0-flash",
            contents="hi",
            config=types.GenerateContentConfig(tools=[WEATHER_TOOL]),
        )
        return len(list(stream))

    assert agent() == 2
    call = _only_call(read_captures)
    assert call["output"] == ""
    assert call["requested_tool_calls"] == [
        {"name": "get_weather", "arguments": {"city": "Paris"}, "call_id": "fc-1"},
        {"name": "get_weather", "arguments": {"city": "Oslo"}, "call_id": None},
    ]
    assert call["output_tokens"] == 3


def test_stream_that_fails_midway_records_what_arrived(
    capturing: Path, read_captures: CaptureReader
) -> None:
    class _Failing(_Models):
        def generate_content_stream(self, **kwargs: Any) -> Iterator[Any]:
            yield _response(types.Part(text="par"), usage=None)
            raise ConnectionError("dropped")

    client = _client(_Failing())

    @capture.agent(suite=SUITE, redact=False, tools=[])
    def agent() -> None:
        for _ in client.models.generate_content_stream(model="gemini-2.0-flash", contents="hi"):
            pass

    with pytest.raises(ConnectionError):
        agent()
    call = _only_call(read_captures)
    assert call["output"] == "par"
    assert call["input_tokens"] == 0


async def test_stream_async(capturing: Path, read_captures: CaptureReader) -> None:
    client = _client(aio=_AsyncModels(chunks=_call_chunks()))

    @capture.agent(suite=SUITE, redact=False, tools=[])
    async def agent() -> int:
        stream = await client.aio.models.generate_content_stream(
            model="gemini-2.0-flash",
            contents="hi",
            config=types.GenerateContentConfig(tools=[WEATHER_TOOL]),
        )
        assert isinstance(stream, AsyncStreamProxy)
        return len([chunk async for chunk in stream])

    assert await agent() == 2
    call = _only_call(read_captures)
    assert call["input_tokens"] == 12
    assert [c["arguments"]["city"] for c in call["requested_tool_calls"]] == ["Paris", "Oslo"]


# --- packaging ----------------------------------------------------------------------------------


def test_module_imports_without_google_genai_installed() -> None:
    """``google.genai`` is only ever imported lazily and guarded (D-deps): the module itself has
    no top-level import of it, and it loads when the package is absent."""
    source = Path(wrap_genai.__code__.co_filename).read_text(encoding="utf-8")
    top_level_imports = [
        line for line in source.splitlines() if line.startswith(("import google", "from google"))
    ]
    assert top_level_imports == []
    probe = (
        "import sys; sys.modules['google'] = None; sys.modules['google.genai'] = None; "
        "import evalshift.adapters.genai as m; assert callable(m.wrap_genai)"
    )
    subprocess.run([sys.executable, "-c", probe], check=True)


def test_callable_tools_without_google_genai_are_left_to_normalise(
    capturing: Path, read_captures: CaptureReader, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without the SDK to declare a callable, the raw callable stays in the list and the shared
    normaliser refuses the toolset (not stamped) -- the call itself is still recorded."""
    monkeypatch.setitem(sys.modules, "google.genai", None)

    def add(a: int) -> int:
        return a

    _run(_client(), config={"tools": [add]})
    call = _only_call(read_captures)
    assert call["tools_offered"] is None
    assert call["toolset_ref"] is None
    assert call["output"] == "Hello!"


# --- real client over a local stub -------------------------------------------------------------

CANNED_GENERATE_CONTENT = {
    "candidates": [
        {
            "content": {
                "role": "model",
                "parts": [{"functionCall": {"name": "get_weather", "args": {"city": "Paris"}}}],
            },
            "finishReason": "STOP",
            "index": 0,
        }
    ],
    "usageMetadata": {"promptTokenCount": 12, "candidatesTokenCount": 3, "totalTokenCount": 15},
    "modelVersion": "gemini-2.0-flash",
}


class _StubHandler(BaseHTTPRequestHandler):
    seen: ClassVar[list[tuple[str, dict[str, Any]]]] = []

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        type(self).seen.append((self.path, json.loads(self.rfile.read(length))))
        body = json.dumps(CANNED_GENERATE_CONTENT).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: Any) -> None:
        pass


@pytest.fixture
def stub_server() -> Iterator[str]:
    _StubHandler.seen = []
    server = HTTPServer(("127.0.0.1", 0), _StubHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


def test_real_genai_client_smoke(
    capturing: Path, read_captures: CaptureReader, stub_server: str
) -> None:
    genai = pytest.importorskip("google.genai")
    raw = genai.Client(api_key="test", http_options={"base_url": stub_server})
    client = wrap_genai(raw)

    @capture.agent(suite=SUITE, redact=False, tools=[])
    def agent() -> Any:
        return client.models.generate_content(
            model="gemini-2.0-flash",
            contents="Weather in Paris?",
            config=types.GenerateContentConfig(
                temperature=0,
                system_instruction="Be terse.",
                tools=[WEATHER_TOOL],
                automatic_function_calling={"disable": True},
            ),
        )

    response = agent()
    assert response.function_calls[0].name == "get_weather"
    path, body = _StubHandler.seen[0]
    assert path.endswith("/models/gemini-2.0-flash:generateContent")
    assert body["contents"] == [{"parts": [{"text": "Weather in Paris?"}], "role": "user"}]
    call = _only_call(read_captures)
    assert call["model_id"] == "gemini-2.0-flash"
    assert call["input"] == [
        {"role": "system", "content": "Be terse."},
        {"role": "user", "content": "Weather in Paris?"},
    ]
    assert call["output"] == ""
    assert call["input_tokens"] == 12
    assert call["output_tokens"] == 3
    assert call["tools_offered"] == ["get_weather"]
    assert call["requested_tool_calls"] == [
        {"name": "get_weather", "arguments": {"city": "Paris"}, "call_id": None}
    ]
    assert call["metadata"]["generation_config"] == {"temperature": 0.0}

    # ``chats`` is forwarded, not intercepted: a Chat drives the *unwrapped* ``models`` it was
    # created from, so its turn reaches the server but records no ``model_call``.
    @capture.agent(suite="genai-chat", redact=False, tools=[])
    def chat_agent() -> Any:
        return client.chats.create(model="gemini-2.0-flash").send_message("hi").function_calls

    assert chat_agent()[0].name == "get_weather"
    assert len(_StubHandler.seen) == 2
    assert _model_calls(read_captures("genai-chat")[0]) == []
