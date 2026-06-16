"""The public capture surface: ``@capture.agent``, ``@capture.tool``, ``record_model_call``.

Recording is **off unless** ``EVALSHIFT_CAPTURE`` is set (the :mod:`evalshift.config` gate). When
off, every entry point is a thin pass-through. When on, an ``@capture.agent`` invocation builds a
:class:`~evalshift.capture.span.SpanTree`, records the tool/model spans made beneath it, and
writes one capture file via :func:`evalshift.config.active_sink`.

**Fail-open is sacred.** The user's own function call is the only statement not wrapped by the
:mod:`evalshift.safety` guards; its real return value and exception always propagate. When the
user function raises, an ``error`` event is recorded and the (partial) capture is still written
before the original exception is re-raised — failed runs are the highest-value telemetry.

Stdlib only (D-deps).
"""

from __future__ import annotations

import functools
import inspect
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
from typing import Any, ParamSpec, TypeVar, overload

from evalshift import config, safety
from evalshift.capture import state
from evalshift.capture.span import SpanTree
from evalshift.redaction import Redactor
from evalshift.trace.serialize import build_capture

P = ParamSpec("P")
R = TypeVar("R")


def _now() -> float:
    """Wall-clock seconds for span timing (patched in tests for determinism)."""
    return time.time()


def _new_capture_id() -> str:
    return f"cap_{uuid.uuid4().hex}"


def _new_call_id() -> str:
    return f"call_{uuid.uuid4().hex}"


def _bind(fn: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    """Best-effort ``{param: value}`` view of a call; never raises (for hashing + tool args)."""
    try:
        bound = inspect.signature(fn).bind(*args, **kwargs)
        bound.apply_defaults()
        return dict(bound.arguments)
    except (TypeError, ValueError):
        return {"args": list(args), "kwargs": dict(kwargs)}


def _record_error(tree: SpanTree, exc: BaseException) -> None:
    with safety.fail_open("record error event"):
        ts = _now()
        span = tree.open_span(
            "error",
            span_id=f"err_{uuid.uuid4().hex}",
            start_ts=ts,
            parent_call_id=state.current_parent(),
            data={"message": str(exc), "category": type(exc).__name__},
        )
        tree.close_span(span, end_ts=ts)


def _finalize(
    tree: SpanTree,
    *,
    suite: str,
    agent_input: Any,
    capture_id: str,
    code_version: str,
    redact: Redactor | None,
) -> None:
    # Decorator-level redact wins; else fall back to the process-wide configure(redact=...).
    redactor = redact if redact is not None else config.active_redactor()
    envelope = safety.guard(
        "build capture",
        lambda: build_capture(
            tree,
            suite=suite,
            agent_input=agent_input,
            capture_id=capture_id,
            code_version=code_version,
            redact=redactor,
        ),
    )
    if envelope is None:
        return
    with safety.fail_open("sink write"):
        config.active_sink().write(envelope)


def record_model_call(
    *,
    model_id: str,
    input: Any = None,
    output: Any = None,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cost_usd: float = 0.0,
    latency_ms: int | None = None,
) -> None:
    """Record a completed model call into the active capture session. No-op outside one."""
    tree = state.current_tree()
    if tree is None:
        return
    with safety.fail_open("record model call"):
        data: dict[str, Any] = {
            "model_id": model_id,
            "input": input,
            "output": output,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": cost_usd,
        }
        if latency_ms is not None:
            data["latency_ms"] = latency_ms
        ts = _now()
        span = tree.open_span(
            "model_call",
            span_id=f"mc_{uuid.uuid4().hex}",
            start_ts=ts,
            parent_call_id=state.current_parent(),
            data=data,
        )
        tree.close_span(span, end_ts=ts)


def _run_tool(
    fn: Callable[P, R],
    tool_name: str,
    tree: SpanTree,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> R:
    call_id = _new_call_id()
    arguments = safety.guard("bind tool args", lambda: _bind(fn, args, kwargs)) or {}
    parent = state.current_parent()
    start = _now()
    span = safety.guard(
        "open tool span",
        lambda: tree.open_span(
            "tool",
            span_id=call_id,
            start_ts=start,
            parent_call_id=parent,
            data={"name": tool_name, "arguments": arguments},
        ),
    )
    parent_cm = state.use_parent(call_id) if span is not None else nullcontext()
    with parent_cm:
        try:
            result = fn(*args, **kwargs)
        except BaseException as exc:
            if span is not None:
                with safety.fail_open("close tool span (error)"):
                    tree.close_span(span, end_ts=_now(), result=None, error=str(exc))
            raise
        else:
            if span is not None:
                with safety.fail_open("close tool span"):
                    tree.close_span(span, end_ts=_now(), result=result)
            return result


def _run_agent(
    fn: Callable[P, R],
    *,
    suite: str,
    code_version: str,
    redact: Redactor | None,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> R:
    tree = safety.guard("open session", SpanTree)
    if tree is None:
        return fn(*args, **kwargs)  # bookkeeping failed -> transparent pass-through
    agent_input = safety.guard("derive agent input", lambda: _bind(fn, args, kwargs))
    capture_id = _new_capture_id()
    with state.use_tree(tree):
        try:
            result = fn(*args, **kwargs)
        except BaseException as exc:
            _record_error(tree, exc)
            _finalize(
                tree,
                suite=suite,
                agent_input=agent_input,
                capture_id=capture_id,
                code_version=code_version,
                redact=redact,
            )
            raise
        else:
            _finalize(
                tree,
                suite=suite,
                agent_input=agent_input,
                capture_id=capture_id,
                code_version=code_version,
                redact=redact,
            )
            return result


class _Capture:
    """Public capture facade; the module exposes a single instance named ``capture``."""

    def agent(
        self,
        *,
        suite: str,
        redact: Redactor | None = None,  # masks payloads before any byte hits disk (D-4)
        code_version: str = "",
    ) -> Callable[[Callable[P, R]], Callable[P, R]]:
        """Decorator that captures one agent invocation (no-op unless the gate is on).

        ``redact`` (a ``(value) -> value`` callable) masks tool/model payloads in-process before
        serialization; it overrides any process-wide ``configure(redact=...)``. If it raises, the
        capture is dropped rather than written unredacted — the host agent is never affected.
        """

        def decorate(fn: Callable[P, R]) -> Callable[P, R]:
            @functools.wraps(fn)
            def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                if not config.is_capture_enabled():
                    return fn(*args, **kwargs)
                return _run_agent(
                    fn,
                    suite=suite,
                    code_version=code_version,
                    redact=redact,
                    args=args,
                    kwargs=kwargs,
                )

            return wrapper

        return decorate

    @contextmanager
    def agent_session(
        self,
        *,
        suite: str,
        code_version: str = "",
        agent_input: Any = None,
        redact: Redactor | None = None,
    ) -> Iterator[SpanTree | None]:
        """Context-manager form of :meth:`agent` for inline/manual instrumentation."""
        if not config.is_capture_enabled():
            yield None
            return
        tree = safety.guard("open session", SpanTree)
        if tree is None:
            yield None
            return
        capture_id = _new_capture_id()
        with state.use_tree(tree):
            try:
                yield tree
            except BaseException as exc:
                _record_error(tree, exc)
                _finalize(
                    tree,
                    suite=suite,
                    agent_input=agent_input,
                    capture_id=capture_id,
                    code_version=code_version,
                    redact=redact,
                )
                raise
            else:
                _finalize(
                    tree,
                    suite=suite,
                    agent_input=agent_input,
                    capture_id=capture_id,
                    code_version=code_version,
                    redact=redact,
                )

    @overload
    def tool(self, fn: Callable[P, R]) -> Callable[P, R]: ...

    @overload
    def tool(self, *, name: str | None = None) -> Callable[[Callable[P, R]], Callable[P, R]]: ...

    def tool(self, fn: Callable[..., Any] | None = None, *, name: str | None = None) -> Any:
        """Decorator that records a tool span (no-op when no agent session is active).

        Supports both ``@capture.tool`` and ``@capture.tool(name="...")``.
        """

        def decorate(target: Callable[P, R]) -> Callable[P, R]:
            tool_name = name or target.__name__

            @functools.wraps(target)
            def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                tree = state.current_tree()
                if tree is None:
                    return target(*args, **kwargs)
                return _run_tool(target, tool_name, tree, args, kwargs)

            return wrapper

        return decorate if fn is None else decorate(fn)


capture = _Capture()


__all__ = ["capture", "record_model_call"]
