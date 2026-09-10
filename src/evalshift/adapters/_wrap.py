"""Shared machinery for the provider client wrappers (D-wrappers).

The provider wrappers (``adapters/openai.py``, ``adapters/anthropic.py``, ``adapters/genai.py``)
are thin proxies over a client object the user already constructed. Everything that is *not*
provider-specific lives here, so each wrapper is only a description of one provider's shapes:

* :class:`ClientProxy` -- forwards every attribute to the wrapped client and replaces a handful
  of named methods (``chat.completions.create`` ...) with instrumented versions.
* :func:`instrument` -- the instrumented version of one method: read a :class:`CallSpec` from the
  kwargs, time the real call, then record one ``model_call`` through
  :func:`evalshift.capture.api.record_model_call`. Sync and async methods are both handled.
* :class:`StreamProxy` / :class:`AsyncStreamProxy` -- when the response is a stream, the
  provider's ``on_chunk`` folds each chunk into a :class:`StreamState` and one ``model_call`` is
  recorded when the stream ends (exhausted, closed, or failed).

**Fail-open is sacred.** The user's real call is never guarded: its exceptions propagate exactly
as they would without the wrapper. Every piece of SDK bookkeeping around it runs under
:func:`evalshift.safety.guard` / :func:`~evalshift.safety.fail_open`, so a wrapper fault degrades
to "this call was not recorded", never to a broken client. Outside an active capture session the
wrapper is inert: ``record_model_call`` is a no-op there and no stream state is kept.

Nothing here imports a provider SDK (D-deps): the wrappers pass in plain callables that know the
provider's kwargs and response shapes; this module only duck-types what they hand back.
"""

from __future__ import annotations

import functools
import inspect
import time
from collections.abc import AsyncIterator, Callable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

from evalshift import safety
from evalshift.capture import api
from evalshift.capture.requested import extract_requested_tool_calls


@dataclass(frozen=True)
class CallSpec:
    """What one provider call *asked for*, read from its kwargs before the request is sent.

    ``tools`` follows :func:`~evalshift.capture.api.record_model_call`'s contract, with one
    wrapper-specific rule: a wrapper sees exactly what the provider was sent, so it always
    asserts the per-call value -- the ``tools`` kwarg when present, ``[]`` when absent -- and
    never ``None`` (inherit the session's). ``generation_config`` may be the raw kwargs dict:
    ``record_model_call`` allow-lists and JSON-coerces it.
    """

    model_id: str
    tools: Any = None
    input: Any = None
    generation_config: dict[str, Any] | None = None


@dataclass(frozen=True)
class Completion:
    """What a finished (non-streaming) response contained.

    ``requested_tool_calls`` left as ``None`` means "let the base extract them": the recorder
    then runs :func:`~evalshift.capture.requested.extract_requested_tool_calls` over the raw
    response. Pass ``[]`` to assert the model asked for nothing.
    """

    output: Any = None
    input_tokens: int = 0
    output_tokens: int = 0
    requested_tool_calls: Any = None


@dataclass
class StreamState:
    """Mutable accumulator a provider's ``on_chunk`` fills in as a stream is consumed.

    ``text`` parts are joined into the recorded output. ``scratch`` is free for the provider
    (e.g. partial tool-call argument deltas keyed by index) and is never recorded.
    """

    text: list[str] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    requested_tool_calls: Any = None
    scratch: dict[str, Any] = field(default_factory=dict)

    def completion(self) -> Completion:
        return Completion(
            output="".join(self.text),
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            requested_tool_calls=self.requested_tool_calls,
        )


@dataclass(frozen=True)
class Instrumentation:
    """How one provider method is captured. Every callable is run fail-open by the base.

    Attributes:
        describe: kwargs -> :class:`CallSpec`, or ``None`` to leave this call unrecorded.
        complete: non-streaming response -> :class:`Completion`, or ``None`` to record nothing.
        is_stream: ``(kwargs, response)`` -> whether ``response`` is a stream to be proxied.
        on_chunk: fold one stream chunk into the :class:`StreamState`.
        on_stream_end: optional last pass over the state (assemble tool calls from deltas ...).
    """

    describe: Callable[[Mapping[str, Any]], CallSpec | None]
    complete: Callable[[Any], Completion | None]
    is_stream: Callable[[Mapping[str, Any], Any], bool] = lambda kwargs, response: False
    on_chunk: Callable[[Any, StreamState], None] = lambda chunk, state: None
    on_stream_end: Callable[[StreamState], None] | None = None


def _latency_ms(start: float) -> int:
    return max(0, int((time.perf_counter() - start) * 1000))


def record(
    spec: CallSpec, completion: Completion, *, latency_ms: int | None, raw_response: Any = None
) -> None:
    """Record one ``model_call`` for ``spec`` + ``completion``. Fail-open; inert outside a session.

    ``raw_response`` is consulted only when ``completion.requested_tool_calls`` is ``None``.
    """
    with safety.fail_open("wrapper record model call"):
        requested = completion.requested_tool_calls
        if requested is None and raw_response is not None:
            requested = extract_requested_tool_calls(raw_response)
        api.record_model_call(
            model_id=spec.model_id,
            tools=spec.tools,
            input=spec.input,
            output=completion.output,
            requested_tool_calls=requested,
            input_tokens=completion.input_tokens,
            output_tokens=completion.output_tokens,
            latency_ms=latency_ms,
            generation_config=spec.generation_config,
        )


def _record_response(spec: CallSpec, response: Any, inst: Instrumentation, start: float) -> None:
    latency = _latency_ms(start)
    completion = safety.guard("wrapper complete", lambda: inst.complete(response))
    if completion is not None:
        record(spec, completion, latency_ms=latency, raw_response=response)


class _StreamBase:
    """State shared by the sync and async stream proxies (attribute forwarding, one-shot finish)."""

    __slots__ = ("_done", "_inst", "_spec", "_start", "_state", "_stream")

    def __init__(self, stream: Any, spec: CallSpec, inst: Instrumentation, start: float) -> None:
        self._stream = stream
        self._spec = spec
        self._inst = inst
        self._start = start
        self._state = StreamState()
        self._done = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)

    def __repr__(self) -> str:
        return f"<evalshift stream wrapper of {self._stream!r}>"

    def _feed(self, chunk: Any) -> None:
        with safety.fail_open("wrapper stream chunk"):
            self._inst.on_chunk(chunk, self._state)

    def _finish(self) -> None:
        """Record the accumulated stream exactly once (exhausted, closed, or failed)."""
        if self._done:
            return
        self._done = True
        latency = _latency_ms(self._start)
        with safety.fail_open("wrapper stream end"):
            if self._inst.on_stream_end is not None:
                self._inst.on_stream_end(self._state)
        record(self._spec, self._state.completion(), latency_ms=latency)


class StreamProxy(_StreamBase):
    """Iterable proxy over a provider's sync stream; records once the stream ends.

    Supports ``for chunk in stream``, ``with stream:`` (forwarded to the inner object when it
    is a context manager) and ``stream.close()``. A stream the caller abandons without
    exhausting, closing, or erroring records nothing -- there is no final chunk to take usage
    from, and no hook to know the caller is done.
    """

    __slots__ = ()

    def __iter__(self) -> Iterator[Any]:
        try:
            for chunk in self._stream:
                self._feed(chunk)
                yield chunk
        except BaseException:
            self._finish()
            raise
        self._finish()

    def __enter__(self) -> StreamProxy:
        enter = getattr(self._stream, "__enter__", None)
        if callable(enter):
            enter()
        return self

    def __exit__(self, *exc: object) -> Any:
        self._finish()
        leave = getattr(self._stream, "__exit__", None)
        return leave(*exc) if callable(leave) else None

    def close(self) -> None:
        self._finish()
        close = getattr(self._stream, "close", None)
        if callable(close):
            close()


class AsyncStreamProxy(_StreamBase):
    """Async-iterable proxy over a provider's async stream; see :class:`StreamProxy`."""

    __slots__ = ()

    async def __aiter__(self) -> AsyncIterator[Any]:
        try:
            async for chunk in self._stream:
                self._feed(chunk)
                yield chunk
        except BaseException:
            self._finish()
            raise
        self._finish()

    async def __aenter__(self) -> AsyncStreamProxy:
        enter = getattr(self._stream, "__aenter__", None)
        if callable(enter):
            await enter()
        return self

    async def __aexit__(self, *exc: object) -> Any:
        self._finish()
        leave = getattr(self._stream, "__aexit__", None)
        return await leave(*exc) if callable(leave) else None

    async def aclose(self) -> None:
        self._finish()
        close = getattr(self._stream, "aclose", None) or getattr(self._stream, "close", None)
        if callable(close):
            result = close()
            if inspect.isawaitable(result):
                await result


def instrument(original: Callable[..., Any], inst: Instrumentation) -> Callable[..., Any]:
    """Return ``original`` wrapped so that each call records one ``model_call``.

    The real call is never guarded and its return value is handed back unchanged, except that a
    streaming response is returned wrapped in a :class:`StreamProxy` / :class:`AsyncStreamProxy`
    (which forwards every attribute of the original stream). Picks the async variant when
    ``original`` is a coroutine function, or when a plain function hands back an awaitable
    (anthropic's ``AsyncMessages.create`` is a ``def`` returning a coroutine): the awaitable is
    then returned as a coroutine that records once it resolves, so ``await`` works unchanged.
    """

    async def resolve(
        awaitable: Any, spec: CallSpec | None, kw: dict[str, Any], start: float
    ) -> Any:
        response = await awaitable
        if spec is None:
            return response
        if safety.guard("wrapper is_stream", lambda: inst.is_stream(kw, response)):
            return AsyncStreamProxy(response, spec, inst, start)
        _record_response(spec, response, inst, start)
        return response

    # ``inspect.unwrap`` sees through ``functools.wraps`` decorators: the Stainless-generated
    # SDKs (openai, anthropic) wrap their async ``create`` in a *sync* ``@required_args``
    # wrapper, which ``iscoroutinefunction`` alone reports as sync. A plain ``def`` that hands
    # back an awaitable anyway is caught below in ``wrapper``.
    if inspect.iscoroutinefunction(inspect.unwrap(original)):

        @functools.wraps(original)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            spec = safety.guard("wrapper describe", lambda: inst.describe(kwargs))
            start = time.perf_counter()
            return await resolve(original(*args, **kwargs), spec, kwargs, start)

        return async_wrapper

    @functools.wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        spec = safety.guard("wrapper describe", lambda: inst.describe(kwargs))
        start = time.perf_counter()
        response = original(*args, **kwargs)
        if inspect.isawaitable(response):
            return resolve(response, spec, kwargs, start)
        if spec is None:
            return response
        if safety.guard("wrapper is_stream", lambda: inst.is_stream(kwargs, response)):
            return StreamProxy(response, spec, inst, start)
        _record_response(spec, response, inst, start)
        return response

    return wrapper


#: ``name -> Instrumentation`` replaces that method; ``name -> Overrides`` descends into that
#: attribute (a namespace such as ``client.chat``) with its own overrides.
Overrides = Mapping[str, "Instrumentation | Mapping[str, Any]"]


class ClientProxy:
    """Attribute-forwarding proxy over a client, with a few named methods instrumented.

    Every attribute not named in ``overrides`` is fetched from the wrapped object untouched, so
    the proxy is a drop-in for the client in ordinary use (``with client:`` and
    ``async with client:`` are forwarded too). It is **not** an instance of the client's class:
    code that does ``isinstance(client, OpenAI)`` should use :func:`unwrap`.
    """

    __slots__ = ("_overrides", "_target")

    def __init__(self, target: Any, overrides: Overrides) -> None:
        self._target = target
        self._overrides = overrides

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._target, name)
        override = self._overrides.get(name)
        if override is None:
            return attr
        if isinstance(override, Instrumentation):
            if not callable(attr):
                return attr
            return instrument(attr, override)
        return ClientProxy(attr, override)

    def __dir__(self) -> list[str]:
        return dir(self._target)

    def __repr__(self) -> str:
        return f"<evalshift wrapper of {self._target!r}>"

    def __enter__(self) -> ClientProxy:
        enter = getattr(self._target, "__enter__", None)
        if callable(enter):
            enter()
        return self

    def __exit__(self, *exc: object) -> Any:
        leave = getattr(self._target, "__exit__", None)
        return leave(*exc) if callable(leave) else None

    async def __aenter__(self) -> ClientProxy:
        enter = getattr(self._target, "__aenter__", None)
        if callable(enter):
            await enter()
        return self

    async def __aexit__(self, *exc: object) -> Any:
        leave = getattr(self._target, "__aexit__", None)
        return await leave(*exc) if callable(leave) else None


def unwrap(client: Any) -> Any:
    """Return the real client behind a :class:`ClientProxy` (or ``client`` itself if not one)."""
    while isinstance(client, ClientProxy):
        client = client._target
    return client


__all__ = [
    "AsyncStreamProxy",
    "CallSpec",
    "ClientProxy",
    "Completion",
    "Instrumentation",
    "Overrides",
    "StreamProxy",
    "StreamState",
    "instrument",
    "record",
    "unwrap",
]
