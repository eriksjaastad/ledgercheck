"""Tracing: NullTracer default, env factory, pipeline spans, Langfuse adapter (faked)."""

import os
import re
import socket
import subprocess
import sys
import tomllib
import types
from pathlib import Path

import pytest

from ledgercheck import observability
from ledgercheck.agents import IntakeAgent, run_pipeline
from ledgercheck.fixtures_loader import load_cases
from ledgercheck.observability import (
    HOST_ENV,
    PIPELINE_TRACE,
    PUBLIC_KEY_ENV,
    SECRET_KEY_ENV,
    LangfuseTracer,
    LangfuseUnavailable,
    NullTracer,
    SpanRecord,
    TraceRecord,
    Tracer,
    Usage,
    trace_run,
    tracer_from_env,
)

ROOT = Path(__file__).resolve().parents[1]
KEYS = {PUBLIC_KEY_ENV: "pk-lf-test", SECRET_KEY_ENV: "sk-lf-test"}
STEPS = ("intake", "policy", "approval")


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)


class RecordingTracer:
    """Logs every lifecycle event; handles are labels."""

    def __init__(self):
        self.events = []
        self.spans = []
        self.traces = []

    def start_trace(self, name, metadata):
        self.events.append(("start_trace", name, dict(metadata)))
        return f"trace:{name}"

    def start_span(self, trace, name):
        self.events.append(("start_span", trace, name))
        return f"span:{name}"

    def end_span(self, span, record):
        self.events.append(("end_span", span))
        self.spans.append(record)

    def end_trace(self, trace, record):
        self.events.append(("end_trace", trace))
        self.traces.append(record)


@pytest.fixture
def sdk_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "langfuse", None)  # import langfuse -> ImportError


@pytest.fixture
def fake_sdk(monkeypatch):
    """A stand-in ``langfuse`` module that records calls and sends nothing."""
    calls = []

    class FakeSpan:
        def __init__(self, name, metadata=None):
            self.name = name
            calls.append(("start_span", name, metadata))

        def start_span(self, name):
            return FakeSpan(name)

        def update(self, **kwargs):
            calls.append(("update", self.name, kwargs))

        def update_trace(self, **kwargs):
            calls.append(("update_trace", self.name, kwargs))

        def end(self):
            calls.append(("end", self.name))

    class Langfuse:
        def __init__(self, **options):
            calls.append(("client", options))

        def start_span(self, name, metadata=None):
            return FakeSpan(name, metadata)

    monkeypatch.setitem(sys.modules, "langfuse", types.SimpleNamespace(Langfuse=Langfuse))
    return calls


def test_null_tracer_is_the_default_and_a_no_op():
    tracer = tracer_from_env()
    assert type(tracer) is NullTracer
    assert isinstance(tracer, Tracer)
    with trace_run(tracer, "t", {"k": "v"}) as run:
        with run.span("s"):
            pass
    assert run.handle is None
    assert vars(tracer) == {}
    record = SpanRecord("s", 0.0)
    assert tracer.end_span(None, record) is None
    assert tracer.end_trace(None, TraceRecord("t", {}, (record,), 0.0)) is None


@pytest.mark.parametrize("env", [
    {},
    {PUBLIC_KEY_ENV: "pk"},
    {SECRET_KEY_ENV: "sk"},
    {PUBLIC_KEY_ENV: "pk", SECRET_KEY_ENV: ""},
    {PUBLIC_KEY_ENV: " \t", SECRET_KEY_ENV: "sk"},
    {PUBLIC_KEY_ENV: "pk", SECRET_KEY_ENV: "  ", HOST_ENV: "https://langfuse.example"},
    {HOST_ENV: "https://langfuse.example"},
])
def test_factory_picks_null_tracer_when_a_key_is_missing_or_blank(env, sdk_missing):
    # sdk_missing: any attempt to build a LangfuseTracer would raise.
    assert type(tracer_from_env(env)) is NullTracer


def test_keys_set_but_sdk_missing_fails_clearly(sdk_missing, monkeypatch):
    with pytest.raises(LangfuseUnavailable, match=r"ledgercheck\[langfuse\]") as err:
        tracer_from_env(KEYS)
    assert KEYS[SECRET_KEY_ENV] not in str(err.value)
    for name, value in KEYS.items():
        monkeypatch.setenv(name, value)
    with pytest.raises(LangfuseUnavailable):
        run_pipeline("clean_baseline")  # the default tracer comes from the env


def test_pipeline_records_one_trace_with_spans_in_order():
    tracer = RecordingTracer()
    result = run_pipeline("missing_po", run_id="run-1", tracer=tracer)
    assert tracer.events == [
        ("start_trace", PIPELINE_TRACE, {"case_id": "missing_po"}),
        *[e for step in STEPS for e in (("start_span", f"trace:{PIPELINE_TRACE}", step),
                                        ("end_span", f"span:{step}"))],
        ("end_trace", f"trace:{PIPELINE_TRACE}"),
    ]
    [trace] = tracer.traces
    assert trace.spans == tuple(tracer.spans)
    assert [s.name for s in trace.spans] == list(STEPS)
    assert all(s.usage is None and s.error is None and s.latency_ms >= 0 for s in trace.spans)
    assert trace.error is None
    assert trace.latency_ms >= sum(s.latency_ms for s in trace.spans)
    assert dict(trace.metadata) == {
        "case_id": "missing_po", "run_id": "run-1", "outcome": result.decision.outcome.value,
    }


def test_pipeline_output_is_identical_with_null_and_recording_tracers():
    for case in load_cases():
        plain = run_pipeline(case.case_id, run_id="run-x", tracer=NullTracer())
        traced = run_pipeline(case.case_id, run_id="run-x", tracer=RecordingTracer())
        assert traced == plain, case.case_id


def test_failed_step_ends_its_span_and_the_trace_with_the_error(tmp_path):
    tracer = RecordingTracer()
    with pytest.raises(Exception) as err:
        run_pipeline("clean_baseline", intake=IntakeAgent(fixtures_root=tmp_path), tracer=tracer)
    [trace] = tracer.traces
    assert [(s.name, s.error) for s in trace.spans] == [("intake", type(err.value).__name__)]
    assert trace.error == type(err.value).__name__
    assert "run_id" not in trace.metadata


def test_step_usage_is_recorded_when_a_step_reports_it():
    tracer = RecordingTracer()
    usage = Usage(input_tokens=10, output_tokens=2)
    with trace_run(tracer, "t") as run:
        with run.span("llm") as step:
            step.usage = usage
    assert tracer.spans[0].usage == usage


def test_langfuse_tracer_maps_the_pipeline_onto_the_sdk(fake_sdk):
    env = {**KEYS, HOST_ENV: "https://langfuse.example"}
    tracer = tracer_from_env(env)
    assert type(tracer) is LangfuseTracer
    assert tracer_from_env(env) is tracer  # one client per key set
    run_pipeline("clean_baseline", run_id="run-1", tracer=tracer)
    assert fake_sdk[0] == ("client", {
        "public_key": "pk-lf-test", "secret_key": "sk-lf-test", "host": "https://langfuse.example",
    })
    assert fake_sdk[1] == ("start_span", PIPELINE_TRACE, {"case_id": "clean_baseline"})
    assert [c[1] for c in fake_sdk if c[0] == "end"] == [*STEPS, PIPELINE_TRACE]
    step_meta = [c[2]["metadata"] for c in fake_sdk if c[0] == "update" and c[1] in STEPS]
    assert [set(m) for m in step_meta] == [
        {"latency_ms", "input_tokens", "output_tokens", "cost_usd"}
    ] * 3
    [trace_update] = [c[2] for c in fake_sdk if c[0] == "update_trace"]
    assert trace_update["metadata"]["outcome"] == "approve"


def test_blank_host_is_left_to_the_sdk_default(fake_sdk):
    tracer_from_env({**KEYS, HOST_ENV: " "})
    assert fake_sdk == [("client", {"public_key": "pk-lf-test", "secret_key": "sk-lf-test"})]


def test_nested_spans_are_listed_in_end_order():
    tracer = RecordingTracer()
    with trace_run(tracer, "t") as run:
        with run.span("outer"):
            with run.span("inner"):
                pass
    [trace] = tracer.traces
    assert [s.name for s in trace.spans] == ["inner", "outer"]


def test_default_path_does_not_import_langfuse():
    # A meta_path finder logs every attempt to import langfuse, so a guarded
    # import (try/except ImportError) fails here even though the SDK is absent.
    env = {k: v for k, v in os.environ.items() if not k.startswith("LANGFUSE_")}
    code = (
        "import sys\n"
        "attempts = []\n"
        "class Spy:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] == 'langfuse':\n"
        "            attempts.append(name)\n"
        "        return None\n"
        "sys.meta_path.insert(0, Spy())\n"
        "import ledgercheck, ledgercheck.cli, ledgercheck.observability\n"
        "from ledgercheck.agents import run_pipeline\n"
        "run_pipeline('clean_baseline')\n"
        "assert attempts == [] and 'langfuse' not in sys.modules, attempts\n"
    )
    done = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env,
                          capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr


def test_docstring_env_vars_and_extra_match_the_code():
    doc = observability.__doc__ or ""
    assert set(re.findall(r"LANGFUSE_[A-Z_]+", doc)) == {PUBLIC_KEY_ENV, SECRET_KEY_ENV, HOST_ENV}
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert project["dependencies"] == []
    assert any(r.startswith("langfuse") for r in project["optional-dependencies"]["langfuse"])
    assert "pip install 'ledgercheck[langfuse]'" in doc
