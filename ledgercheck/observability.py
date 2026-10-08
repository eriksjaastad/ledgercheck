"""Pipeline tracing: one trace per run, one span per step. Off without keys.

``run_pipeline`` wraps intake, policy and approval in spans of one trace and
hands them to a ``Tracer``. Each span records the step name, its latency in
milliseconds, an error (the exception's type name, if the step raised) and a
``Usage`` slot for tokens and cost. That slot stays ``None`` offline: the
fixture path makes no LLM call, so there is nothing to count.

Default: off, nothing sent
--------------------------
``connections.tracer()`` (what ``run_pipeline`` uses when it is given no
``tracer``) returns ``NullTracer`` unless both ``LANGFUSE_PUBLIC_KEY`` and
``LANGFUSE_SECRET_KEY`` are set and non-blank (not empty or whitespace only),
in the environment or in a local ``.env`` (see ``ledgercheck.connections``).
``NullTracer`` does nothing: no I/O, no network, no import of the
``langfuse`` SDK. With one key set and the other missing or blank, tracing
stays off.

Enabling Langfuse later
-----------------------
1. Create a project in Langfuse (free or paid) and copy its API keys.
2. Install the optional extra: ``pip install 'ledgercheck[langfuse]'``.
   ``dependencies`` stays empty; the SDK only comes in through this extra.
3. Set ``LANGFUSE_PUBLIC_KEY`` and ``LANGFUSE_SECRET_KEY``. Optionally set
   ``LANGFUSE_HOST`` to a self-hosted or regional Langfuse URL; unset or blank
   means the SDK's default cloud host.

Every pipeline run then becomes a trace named ``ledgercheck.pipeline``
(metadata: case id, run id, outcome) with spans ``intake``, ``policy`` and
``approval``. ``LangfuseTracer`` wraps a client built by
``ledgercheck.connections`` and targets the v3 SDK API (``start_span``,
``update``, ``update_trace``, ``end``); the SDK batches and sends in the
background and flushes at exit. Spans carry usage in their metadata until a
live LLM step reports it as a Langfuse generation.

Keys set, SDK missing: ``connections.tracer()`` raises ``LangfuseUnavailable``
(so does every ``run_pipeline`` call that uses the default tracer). Setting
the keys asks for traces, so a silent fallback would hide a misconfiguration.
Install the extra or unset the keys.

Tracer errors are not swallowed: an exception from a tracer propagates out of
``run_pipeline``. Tracers never see or change stage outputs.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from decimal import Decimal
from types import MappingProxyType
from typing import Any, Iterator, Mapping, Protocol, runtime_checkable

PIPELINE_TRACE = "ledgercheck.pipeline"


class LangfuseUnavailable(RuntimeError):
    """Langfuse keys are set but the ``langfuse`` SDK is not installed."""


@dataclass(frozen=True, slots=True)
class Usage:
    """Token usage and cost of one step; any part may be unknown (``None``)."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: Decimal | None = None


@dataclass(frozen=True, slots=True)
class SpanRecord:
    """A finished step: name, latency, usage (``None`` offline), error type name."""

    name: str
    latency_ms: float
    usage: Usage | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class TraceRecord:
    """A finished run: its spans in the order they ended.

    A nested span ends, so is listed, before the span around it. The
    pipeline's steps run one after another, so there this is also start order.
    """

    name: str
    metadata: Mapping[str, str]
    spans: tuple[SpanRecord, ...]
    latency_ms: float
    error: str | None = None


@runtime_checkable
class Tracer(Protocol):
    """Receives trace and span lifecycle events; handles are tracer-defined.

    ``start_trace`` / ``start_span`` return an opaque handle that is passed
    back to the matching ``end_*`` call along with the finished record.
    Timing lives in ``trace_run`` / ``RunTrace.span``, not in tracers.
    """

    def start_trace(self, name: str, metadata: Mapping[str, str]) -> Any: ...

    def start_span(self, trace: Any, name: str) -> Any: ...

    def end_span(self, span: Any, record: SpanRecord) -> None: ...

    def end_trace(self, trace: Any, record: TraceRecord) -> None: ...


class NullTracer:
    """The default tracer: every method does nothing and returns ``None``."""

    def start_trace(self, name: str, metadata: Mapping[str, str]) -> None:
        return None

    def start_span(self, trace: Any, name: str) -> None:
        return None

    def end_span(self, span: Any, record: SpanRecord) -> None:
        return None

    def end_trace(self, trace: Any, record: TraceRecord) -> None:
        return None


class LangfuseTracer:
    """Sends traces to Langfuse through a client built by ``ledgercheck.connections``."""

    def __init__(self, client: Any) -> None:
        self._client = client

    def start_trace(self, name: str, metadata: Mapping[str, str]) -> Any:
        return self._client.start_span(name=name, metadata=dict(metadata))

    def start_span(self, trace: Any, name: str) -> Any:
        return trace.start_span(name=name)

    def end_span(self, span: Any, record: SpanRecord) -> None:
        span.update(metadata=_span_metadata(record), **_level(record.error))
        span.end()

    def end_trace(self, trace: Any, record: TraceRecord) -> None:
        metadata = {**record.metadata, "latency_ms": record.latency_ms}
        trace.update_trace(name=record.name, metadata=metadata)
        trace.update(metadata=metadata, **_level(record.error))
        trace.end()


def _span_metadata(record: SpanRecord) -> dict[str, Any]:
    usage = record.usage
    return {
        "latency_ms": record.latency_ms,
        "input_tokens": None if usage is None else usage.input_tokens,
        "output_tokens": None if usage is None else usage.output_tokens,
        "cost_usd": None if usage is None or usage.cost_usd is None else str(usage.cost_usd),
    }


def _level(error: str | None) -> dict[str, str]:
    return {} if error is None else {"level": "ERROR", "status_message": error}


@dataclass
class Step:
    """The live span of one step; set ``usage`` to report tokens/cost."""

    name: str
    usage: Usage | None = None


@dataclass
class RunTrace:
    """An open trace; ``span(name)`` times one step, ``annotate`` adds metadata."""

    tracer: Tracer
    handle: Any
    metadata: dict[str, str]
    spans: list[SpanRecord] = field(default_factory=list)

    def annotate(self, **metadata: str) -> None:
        self.metadata.update(metadata)

    @contextmanager
    def span(self, name: str) -> Iterator[Step]:
        step = Step(name)
        handle = self.tracer.start_span(self.handle, name)
        start, error = time.perf_counter(), None
        try:
            yield step
        except BaseException as exc:
            error = type(exc).__name__
            raise
        finally:
            record = SpanRecord(name, _ms_since(start), step.usage, error)
            self.spans.append(record)
            self.tracer.end_span(handle, record)


@contextmanager
def trace_run(
    tracer: Tracer, name: str, metadata: Mapping[str, str] | None = None
) -> Iterator[RunTrace]:
    """Open a trace on ``tracer``; it ends (with any error) when the block exits."""
    metadata = dict(metadata or {})
    run = RunTrace(tracer, tracer.start_trace(name, MappingProxyType(dict(metadata))), metadata)
    start, error = time.perf_counter(), None
    try:
        yield run
    except BaseException as exc:
        error = type(exc).__name__
        raise
    finally:
        tracer.end_trace(run.handle, TraceRecord(
            name, MappingProxyType(dict(run.metadata)), tuple(run.spans), _ms_since(start), error
        ))


def _ms_since(start: float) -> float:
    return (time.perf_counter() - start) * 1000
