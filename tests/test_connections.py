"""Connections: settings resolution, the spend gate and cap, the OpenRouter transport (a faked
opener, plus loopback servers for redirects), provider swapping by config, and the key never
leaking."""

import email.message
import http.client
import io
import json
import os
import re
import socketserver
import threading
import urllib.error
import urllib.request
from decimal import Decimal
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from ledgercheck import connections
from ledgercheck.agents.routing import MockRouter, OpenRouterRouter, TaskKind
from ledgercheck.cli import main as cli_main
from ledgercheck.connections import (
    API_KEY_ENV,
    CONNECTIONS_FILE_ENV,
    ENV_FLAG,
    ConnectionConfigError,
    LiveLLMDisabled,
    OpenRouterTransport,
    SpendCapReached,
    TransportError,
    load_settings,
)
from ledgercheck.eval import judge
from ledgercheck.golden import load_golden_cases

KEY = "test-key-not-real-0123456789"
MODEL = "vendor/large-y"
ENV = {name: var for name, (var, _) in connections.SETTINGS.items()}
PRICES = f'[prices."{MODEL}"]\nprompt = 1.0\ncompletion = 2.0\n'
LIVE_TOML = (f'provider = "openrouter"\nspend_cap_usd = 0.05\nmax_tokens = 100\n'
             f'[models]\nlarge = "{MODEL}"\n{PRICES}')
PERFECT = '{"accuracy": 5, "hallucination": 5, "formatting": 5, "notes": []}'
CASE = load_golden_cases()[0].case_id


@pytest.fixture
def local(tmp_path, monkeypatch):
    """Write ``.env`` / ``connections.local.toml`` in a temp dir that the default lookup reads."""
    monkeypatch.setattr(connections, "DOTENV_PATH", tmp_path / ".env")
    monkeypatch.setattr(connections, "DEFAULT_CONFIG", tmp_path / "connections.local.toml")

    def write(dotenv=None, toml=None):
        if dotenv is not None:
            (tmp_path / ".env").write_text(dotenv, encoding="utf-8")
        if toml is not None:
            (tmp_path / "connections.local.toml").write_text(toml, encoding="utf-8")
        return tmp_path

    return write


class FakeOpener:
    """Stands in for ``urllib.request.urlopen``: replays ``replies`` and records requests."""

    def __init__(self, *replies):
        self.replies, self.requests, self.timeouts = list(replies), [], []

    def __call__(self, request, timeout):
        self.requests.append(request)
        self.timeouts.append(timeout)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return io.BytesIO(json.dumps(reply).encode())  # a context manager with read()


def completion(text=PERFECT, **usage):
    return {"choices": [{"message": {"role": "assistant", "content": text}}], "usage": usage}


def http_error(code, body=b"", retry_after=None):
    headers = email.message.Message()
    if retry_after is not None:
        headers["Retry-After"] = retry_after
    return urllib.error.HTTPError("https://x.invalid", code, "error", headers, io.BytesIO(body))


def live_settings(**overrides):
    env = {ENV_FLAG: "1", API_KEY_ENV: KEY, ENV["provider"]: "openrouter",
           ENV["spend_cap_usd"]: "0.05", ENV["max_tokens"]: "100", ENV["model_large"]: MODEL}
    settings = load_settings({**env, **overrides})
    return connections.Settings(**{**settings.__dict__,
                                   "prices": {MODEL: (Decimal("1.0"), Decimal("2.0"))}})


# --- resolution ----------------------------------------------------------------


def test_no_config_is_the_offline_mock():
    s = load_settings()
    assert (s.provider, s.api_key, s.models, s.spend_cap_usd, s.live_flag) == (
        "mock", None, {}, None, False)
    assert set(s.sources.values()) == {"default"}
    assert type(connections.router()) is MockRouter


def test_env_wins_over_dotenv_wins_over_the_file(local, monkeypatch):
    root = local(
        dotenv=f"# local\nexport {ENV['model_small']}='dot/small'\n"
               f"{ENV['model_large']}=dot/large\n",
        toml=f'provider = "openrouter"\nbase_url = "https://proxy.example/v1/"\n'
             f'[models]\nsmall = "file/small"\nlarge = "file/large"\n{PRICES}',
    )
    monkeypatch.setenv(ENV["model_large"], " env/large ")
    s = load_settings()
    assert s.models == {"small": "dot/small", "large": "env/large"}
    assert (s.provider, s.base_url) == ("openrouter", "https://proxy.example/v1")
    file = str(root / "connections.local.toml")
    assert dict(s.sources) == {"provider": file, "api_key": "default", "model_small": ".env",
                               "model_large": "env", "base_url": file, "spend_cap_usd": "default",
                               "max_tokens": "default", "prices": file}
    assert s.prices == {MODEL: (Decimal("1.0"), Decimal("2.0"))}


def test_connections_file_env_names_another_file(local, tmp_path, monkeypatch):
    other = tmp_path / "elsewhere.toml"
    other.write_text('provider = "openrouter"\n')
    monkeypatch.setenv(CONNECTIONS_FILE_ENV, str(other))
    assert load_settings().provider == "openrouter"
    monkeypatch.setenv(CONNECTIONS_FILE_ENV, str(tmp_path / "missing.toml"))
    assert "does not exist" in load_settings().problems[0]
    monkeypatch.setenv(ENV_FLAG, "1")
    with pytest.raises(ConnectionConfigError, match="does not exist"):
        connections.require_live()


def test_the_live_flag_is_never_read_from_a_file(local):
    local(dotenv=f"{ENV_FLAG}=1\n{API_KEY_ENV}={KEY}\n", toml=LIVE_TOML)
    assert not load_settings().live_flag
    with pytest.raises(LiveLLMDisabled, match="live LLM calls are off"):
        connections.router()


@pytest.mark.parametrize("dotenv, toml, match", [
    (None, 'colour = "blue"\n', "unknown keys"),
    (f"{ENV['max_tokens']}=lots\n", None, "max_tokens"),
    (None, "provider = \n", "connections.local.toml"),
    (None, "spend_cap_usd = -1\n", "spend_cap_usd must be a finite number"),
    (None, "max_tokens = 0\n", "max_tokens"),
    (None, f'[prices."{MODEL}"]\nprompt = 1\n', "exactly prompt and completion"),
    (None, 'base_url = "ftp://x"\n', "base_url"),
])
def test_bad_config_stops_only_a_live_run(local, monkeypatch, capsys, dotenv, toml, match):
    local(dotenv=f"{API_KEY_ENV}={KEY}\n" + (dotenv or ""), toml=toml)
    assert any(re.search(match, p) for p in load_settings().problems)
    assert type(connections.router()) is MockRouter  # the offline default still works
    assert cli_main(["connections"]) == 2
    assert re.search(match, capsys.readouterr().err)
    monkeypatch.setenv(ENV_FLAG, "1")
    with pytest.raises(ConnectionConfigError, match=match) as exc:
        connections.require_live()
    assert KEY not in str(exc.value)


JUNK_DOTENV = (b"junk line\n=no name\n\xff\xfe not utf-8\nexport\n[section]\n"
               b"LEDGERCHECK_MAX_TOKENS=lots\nLANGFUSE_PUBLIC_KEY=pk-only\n")


def test_a_stray_dotenv_never_breaks_the_offline_default(local, capsys):
    from ledgercheck import web
    from ledgercheck.observability import NullTracer

    (local() / ".env").write_bytes(JUNK_DOTENV)
    assert cli_main(["judge"]) == judge.EXIT_PASS
    assert capsys.readouterr().out.rstrip().endswith("PASS")
    assert type(connections.tracer()) is NullTracer
    assert type(connections.router()) is MockRouter
    server = web.make_server(connections.run_store(local() / "runs"), "127.0.0.1", 0)
    server.server_close()


def test_dotenv_parsing_rules(local):
    root = local(dotenv=(
        "# comment\n"
        "  export A=plain value # trailing comment\n"
        "B=\"quoted # kept\" # dropped\n"
        "C='single'\n"
        "D=pass#word\n"
        "E=\n"
        "not a setting\n"
        "F = spaced\t# tab comment\n"))
    assert connections._read_dotenv(root / ".env") == {
        "A": "plain value", "B": "quoted # kept", "C": "single", "D": "pass#word", "E": "",
        "F": "spaced"}


# --- the gate and the key ----------------------------------------------------------


def test_key_from_any_source_opens_the_gate_and_never_shows(local, monkeypatch, capsys):
    monkeypatch.setenv(ENV_FLAG, "1")
    local(toml=f'api_key = "{KEY}"\n' + LIVE_TOML)
    settings = connections.require_live()
    assert settings.api_key == KEY and KEY not in repr(settings)
    router = connections.router()
    assert type(router) is OpenRouterRouter and KEY not in repr(router._transport)
    assert cli_main(["connections"]) == 0
    out = capsys.readouterr().out
    assert KEY not in out and "set (from " in out and "connections.local.toml" in out
    local(dotenv=f"{API_KEY_ENV}={KEY}\n")
    assert "api key      set (from .env)" in "\n".join(
        f"{n:<12} {v}" for n, v, _ in connections.describe())


def test_file_key_is_stripped_and_blank_means_missing(local, monkeypatch):
    monkeypatch.setenv(ENV_FLAG, "1")
    local(toml=f'api_key = "  {KEY}  "\n')
    assert connections.require_live().api_key == KEY
    local(toml='api_key = " \\t "\n')
    assert load_settings().api_key is None
    with pytest.raises(LiveLLMDisabled, match=f"{API_KEY_ENV} is not set"):
        connections.require_live()


@pytest.mark.parametrize("bad", [f"{KEY}\n", f"{KEY[:8]} {KEY[8:]}", f"{KEY}\x07x"])
def test_key_with_whitespace_or_control_chars_is_refused_without_echo(local, monkeypatch,
                                                                      capsys, bad):
    monkeypatch.setenv(ENV_FLAG, "1")
    monkeypatch.setenv(API_KEY_ENV, bad)
    if bad.strip() == KEY:  # surrounding whitespace is stripped, so this one is fine
        assert connections.require_live().api_key == KEY
        return
    with pytest.raises(ConnectionConfigError,
                       match="whitespace, control or non-ASCII characters") as exc:
        connections.require_live()
    assert KEY[8:] not in str(exc.value)
    assert cli_main(["connections"]) == 2
    out = capsys.readouterr()
    assert "invalid" in out.out and KEY[8:] not in out.out + out.err
    with pytest.raises(ConnectionConfigError):
        OpenRouterTransport(live_settings(**{API_KEY_ENV: "x y"}), opener=FakeOpener())


def test_invalid_header_errors_never_carry_the_key():
    leak = ValueError(f"Invalid header value b'Bearer {KEY}\\r\\n'")
    with pytest.raises(TransportError, match="details withheld") as exc:
        OpenRouterTransport(live_settings(), opener=FakeOpener(leak))(MODEL, "x")
    assert KEY not in str(exc.value) and exc.value.__cause__ is None


def test_missing_key_names_every_way_to_supply_it():
    with pytest.raises(LiveLLMDisabled) as exc:
        connections.require_live({ENV_FLAG: "1"})
    for where in (f"export {API_KEY_ENV}", ".env", "connections.local.toml",
                  "ledgercheck connections --help"):
        assert where in str(exc.value)


def test_connections_command_with_no_config(capsys):
    assert cli_main(["connections"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("provider     mock")
    assert "api key      missing" in out and "spend cap    missing" in out
    assert "tracing      off" in out


def test_connections_command_reports_a_bad_config(local, capsys):
    local(toml="max_tokens = -5\n")
    assert cli_main(["connections"]) == 2
    assert "max_tokens" in capsys.readouterr().err


def test_connections_help_explains_resolution_and_the_hook(capsys):
    with pytest.raises(SystemExit):
        cli_main(["connections", "--help"])
    text = capsys.readouterr().out
    for needle in ("process environment", ".env", "connections.local.toml", CONNECTIONS_FILE_ENV,
                   "doppler run --", "spend_cap_usd"):
        assert needle in text


# --- the transport -----------------------------------------------------------


def test_transport_posts_chat_completions_and_counts_reported_cost():
    opener = FakeOpener(completion("hi", prompt_tokens=12, completion_tokens=3, cost=0.0004))
    transport = OpenRouterTransport(live_settings(), opener=opener)
    assert transport(MODEL, "hello") == "hi"
    [request] = opener.requests
    assert request.full_url == "https://openrouter.ai/api/v1/chat/completions"
    assert request.get_method() == "POST"
    assert request.get_header("Authorization") == f"Bearer {KEY}"
    assert json.loads(request.data) == {
        "model": MODEL, "messages": [{"role": "user", "content": "hello"}], "max_tokens": 100}
    assert opener.timeouts == [connections.TIMEOUT_S]
    t = transport.totals
    assert (t.calls, t.prompt_tokens, t.completion_tokens, t.cost_usd) == (
        1, 12, 3, Decimal("0.0004"))
    assert "1 calls, 12 prompt + 3 completion tokens, $0.0004 of $0.05 cap" in t.line()


def test_cost_is_computed_from_prices_when_not_reported():
    opener = FakeOpener(completion(prompt_tokens=1000, completion_tokens=50))
    transport = OpenRouterTransport(live_settings(), opener=opener)
    transport(MODEL, "x" * 1000)
    assert transport.totals.cost_usd == Decimal("0.0011")  # 1000 * $1/M + 50 * $2/M


def test_cap_refuses_a_call_whose_worst_case_does_not_fit():
    opener = FakeOpener(completion(prompt_tokens=10, completion_tokens=10, cost=0.0499))
    transport = OpenRouterTransport(live_settings(), opener=opener)
    transport(MODEL, "x")
    with pytest.raises(SpendCapReached, match="spend cap"):
        transport(MODEL, "x")  # worst case ~$0.00022 > $0.0001 left
    assert len(opener.requests) == 1
    tight = OpenRouterTransport(live_settings(**{ENV["spend_cap_usd"]: "0.0001"}),
                                opener=FakeOpener())
    with pytest.raises(SpendCapReached):
        tight(MODEL, "x" * 3000)


@pytest.mark.parametrize("overrides, match", [
    ({ENV["spend_cap_usd"]: ""}, "spend cap"),
    ({ENV["spend_cap_usd"]: "0"}, "spend cap"),
    ({API_KEY_ENV: ""}, API_KEY_ENV),
])
def test_transport_refuses_to_build_without_cap_or_key(overrides, match):
    with pytest.raises(LiveLLMDisabled, match=match):
        OpenRouterTransport(live_settings(**overrides), opener=FakeOpener())


def test_unpriced_model_is_refused_before_sending():
    opener = FakeOpener()
    with pytest.raises(ConnectionConfigError, match=r'\[prices\."vendor/other"\]'):
        OpenRouterTransport(live_settings(), opener=opener)("vendor/other", "x")
    assert opener.requests == []


def test_429_retries_honour_retry_after_with_a_cap():
    sleeps = []
    opener = FakeOpener(http_error(429, retry_after="2"),
                        http_error(429, retry_after="900"),
                        http_error(429), completion("ok", cost=0))
    transport = OpenRouterTransport(live_settings(), opener=opener, sleep=sleeps.append)
    assert transport(MODEL, "x") == "ok"
    assert sleeps == [2.0, connections.MAX_BACKOFF_S, 4.0]
    assert transport.totals.retries == 3 and "3 rate-limit retries" in transport.totals.line()


def test_429_gives_up_after_three_retries():
    sleeps = []
    opener = FakeOpener(*[http_error(429, retry_after="soon") for _ in range(4)])
    transport = OpenRouterTransport(live_settings(), opener=opener, sleep=sleeps.append)
    with pytest.raises(TransportError, match="HTTP 429"):
        transport(MODEL, "x")
    assert sleeps == [1.0, 2.0, 4.0] and len(opener.requests) == 4


def test_other_http_errors_are_not_retried_and_never_show_the_key():
    body = json.dumps({"error": {"message": f"bad key {KEY}"}}).encode()
    opener = FakeOpener(http_error(401, body))
    with pytest.raises(TransportError) as exc:
        OpenRouterTransport(live_settings(), opener=opener,
                            sleep=lambda s: pytest.fail("retried"))(MODEL, "x")
    assert "HTTP 401" in str(exc.value) and "[redacted]" in str(exc.value)
    assert KEY not in str(exc.value) and exc.value.__cause__ is None


class BrokenResponse(io.BytesIO):
    """A response whose body fails mid-read, as a dropped or stalled connection does."""

    def __init__(self, error):
        super().__init__()
        self.error = error

    def read(self, *args):
        raise self.error


@pytest.mark.parametrize("error", [http.client.IncompleteRead(b"{\"cho", 200),
                                   TimeoutError("timed out")])
def test_a_failed_response_read_is_a_transport_error_without_the_key(error):
    transport = OpenRouterTransport(live_settings(),
                                    opener=lambda request, timeout: BrokenResponse(error))
    with pytest.raises(TransportError, match="openrouter.ai/api/v1/chat/completions") as exc:
        transport(MODEL, "x")
    assert KEY not in str(exc.value) and exc.value.__cause__ is None


@pytest.mark.parametrize("reply", [
    urllib.error.URLError("no route"), {"choices": []}, {"choices": [{"message": {}}]},
])
def test_unreachable_or_odd_responses_are_transport_errors(reply):
    with pytest.raises(TransportError):
        OpenRouterTransport(live_settings(), opener=FakeOpener(reply))(MODEL, "x")


def test_prompt_bound_holds_for_text_that_tokenizes_badly():
    prompt = "}{][)(;:!?.,'\"`~^|\\/" * 40 + "€✓🧾" * 20  # about one token per byte, or worse
    messages = [{"role": "user", "content": prompt}]
    bound = connections.max_prompt_tokens(messages)
    assert bound >= len(prompt.encode("utf-8")) + len("user")
    # A cap exactly at the worst case admits the call; the worst case really is worst.
    cap = (bound * 1 + 100 * 2) / Decimal(1_000_000)
    usage = {"prompt_tokens": len(prompt.encode("utf-8")), "completion_tokens": 100}
    opener = FakeOpener(completion(**usage))
    transport = OpenRouterTransport(live_settings(**{ENV["spend_cap_usd"]: str(cap)}),
                                    opener=opener)
    transport(MODEL, prompt)
    assert transport.totals.cost_usd <= transport.totals.cap_usd
    with pytest.raises(SpendCapReached):
        transport(MODEL, prompt)
    assert len(opener.requests) == 1


@pytest.mark.parametrize("usage, charged", [
    ({"prompt_tokens": 1_000_000, "completion_tokens": 3}, ("bound", 3)),
    ({"prompt_tokens": 4, "completion_tokens": 1_000_000}, (4, 100)),
    ({"prompt_tokens": 10 ** 30, "completion_tokens": 10 ** 30}, ("bound", 100)),
    ({}, ("bound", 100)),
])
def test_unreported_cost_never_exceeds_the_pre_call_worst_case(usage, charged):
    bound = connections.max_prompt_tokens([{"role": "user", "content": "x"}])
    worst = (bound * 1 + 100 * 2) / Decimal(1_000_000)
    transport = OpenRouterTransport(live_settings(), opener=FakeOpener(completion(**usage)))
    transport(MODEL, "x")
    t = transport.totals
    want_prompt = bound if charged[0] == "bound" else charged[0]
    assert (t.prompt_tokens, t.completion_tokens) == (want_prompt, charged[1])
    assert t.cost_usd == (want_prompt * 1 + charged[1] * 2) / Decimal(1_000_000) <= worst


def test_a_valid_reported_cost_is_recorded_as_billed():
    opener = FakeOpener(completion(prompt_tokens=10 ** 6, completion_tokens=10 ** 6, cost=0.04))
    transport = OpenRouterTransport(live_settings(), opener=opener)
    transport(MODEL, "x")
    assert transport.totals.cost_usd == Decimal("0.04")  # above the worst case, but billed


def test_a_billed_reply_with_bad_choices_still_counts_against_the_cap():
    reply = {"usage": {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.003}}
    transport = OpenRouterTransport(live_settings(), opener=FakeOpener(reply, {"choices": []}))
    with pytest.raises(TransportError, match="unexpected response shape"):
        transport(MODEL, "x")
    t = transport.totals
    assert (t.calls, t.prompt_tokens, t.completion_tokens, t.cost_usd) == (
        1, 10, 5, Decimal("0.003"))
    with pytest.raises(TransportError):
        transport(MODEL, "x")  # no usage at all: charged at the worst case
    bound = connections.max_prompt_tokens([{"role": "user", "content": "x"}])
    assert (t.calls, t.prompt_tokens, t.completion_tokens) == (2, 10 + bound, 105)


@pytest.mark.parametrize("cost", [-5, "NaN", 1e400, float("nan"), True, "-0.01", "1e400x", [1],
                                  "1e99999999999999999999999999", "1e1000000", 10 ** 12])
def test_invalid_reported_cost_falls_back_to_tokens_times_prices(cost):
    opener = FakeOpener(completion(prompt_tokens=1000, completion_tokens=50, cost=cost),
                        completion(prompt_tokens=1000, completion_tokens=50, cost=cost))
    transport = OpenRouterTransport(live_settings(), opener=opener)
    transport(MODEL, "x" * 1000)
    transport(MODEL, "x" * 1000)  # the budget is still a number, so the cap check still works
    assert transport.totals.cost_usd == Decimal("0.0022")  # 2 x (1000 * $1/M + 50 * $2/M)
    assert transport.totals.cost_usd.is_finite()


def test_invalid_token_counts_are_charged_at_the_worst_case():
    usage = {"prompt_tokens": -10, "completion_tokens": 2.5}
    transport = OpenRouterTransport(live_settings(), opener=FakeOpener(completion(**usage)))
    transport(MODEL, "x")
    t = transport.totals
    bound = connections.max_prompt_tokens([{"role": "user", "content": "x"}])
    assert (t.prompt_tokens, t.completion_tokens) == (bound, 100)
    assert t.cost_usd == (t.prompt_tokens * 1 + 100 * 2) / Decimal(1_000_000)


@pytest.mark.parametrize("usage", [[], "lots", 7])
def test_usage_that_is_not_an_object_is_charged_at_the_worst_case(usage):
    reply = {"choices": [{"message": {"content": "ok"}}], "usage": usage}
    transport = OpenRouterTransport(live_settings(), opener=FakeOpener(reply))
    assert transport(MODEL, "x") == "ok"
    assert transport.totals.completion_tokens == 100


@pytest.mark.parametrize("spent", [Decimal("NaN"), Decimal("sNaN"), Decimal("-Infinity")])
def test_a_broken_budget_refuses_instead_of_crashing(spent):
    opener = FakeOpener()
    transport = OpenRouterTransport(live_settings(), opener=opener)
    transport.totals.cost_usd = spent
    with pytest.raises(SpendCapReached, match="spend cap"):
        transport(MODEL, "x")
    assert opener.requests == []


def test_cli_live_judge_survives_bad_reported_costs(local, monkeypatch, capsys):
    cases = load_golden_cases()[:2]
    local(toml=LIVE_TOML)
    monkeypatch.setenv(ENV_FLAG, "1")
    monkeypatch.setenv(API_KEY_ENV, KEY)
    opener = FakeOpener(completion(PERFECT, prompt_tokens=900, completion_tokens=40, cost=-5),
                        completion(PERFECT, prompt_tokens=900, completion_tokens=40, cost="NaN"))
    monkeypatch.setattr(connections, "_default_opener", lambda: opener)
    args = ["judge", "--live", "--case", cases[0].case_id, "--case", cases[1].case_id]
    assert cli_main(args) == judge.EXIT_PASS
    out = capsys.readouterr().out
    assert "2 calls, 1800 prompt + 80 completion tokens, $0.00196 of $0.05 cap" in out


# --- swapping providers by config only ---------------------------------------------


@pytest.mark.parametrize("provider", ["mock", "openrouter"])
def test_same_code_path_runs_on_either_provider(provider, local, monkeypatch):
    local(toml=LIVE_TOML.replace('"openrouter"', f'"{provider}"'))
    monkeypatch.setenv(ENV_FLAG, "1")
    monkeypatch.setenv(API_KEY_ENV, KEY)
    opener = FakeOpener(completion(PERFECT, prompt_tokens=900, completion_tokens=40, cost=0.001))

    router = connections.router(opener=opener)  # the code under test never changes
    reply = router.complete(TaskKind.JUDGE, "grade this")

    assert router.name == provider
    if provider == "mock":
        assert reply == "" and opener.requests == []
    else:
        assert reply == PERFECT and json.loads(opener.requests[0].data)["model"] == MODEL


def test_cli_live_judge_runs_one_case_and_prints_totals(local, monkeypatch, capsys):
    local(toml=LIVE_TOML)
    monkeypatch.setenv(ENV_FLAG, "1")
    monkeypatch.setenv(API_KEY_ENV, KEY)
    opener = FakeOpener(completion(PERFECT, prompt_tokens=900, completion_tokens=40, cost=0.001))
    monkeypatch.setattr(connections, "_default_opener", lambda: opener)
    assert cli_main(["judge", "--live", "--case", CASE]) == judge.EXIT_PASS
    out = capsys.readouterr().out
    assert "suite (openrouter judge): 1/1 cases passed" in out
    assert (f"live run ({MODEL}): 1 calls, 900 prompt + 40 completion tokens, "
            "$0.001 of $0.05 cap") in out
    assert KEY not in out and len(opener.requests) == 1


def test_cli_live_judge_stops_at_the_cap(local, monkeypatch, capsys):
    local(toml=LIVE_TOML.replace("spend_cap_usd = 0.05", "spend_cap_usd = 0.0001"))
    monkeypatch.setenv(ENV_FLAG, "1")
    monkeypatch.setenv(API_KEY_ENV, KEY)
    opener = FakeOpener()
    monkeypatch.setattr(connections, "_default_opener", lambda: opener)
    assert cli_main(["judge", "--live", "--case", CASE]) == judge.EXIT_ERROR
    err = capsys.readouterr().err
    assert "spend cap" in err and "0 calls" in err and opener.requests == []


def test_run_store_comes_from_connections(tmp_path):
    store = connections.run_store(tmp_path / "runs")
    assert store.root == tmp_path / "runs"


def test_the_suite_guard_fails_any_non_loopback_connection():
    import socket

    with pytest.raises(pytest.fail.Exception, match="network connection attempted"):
        socket.create_connection(("203.0.113.7", 443), timeout=1)
    with pytest.raises(pytest.fail.Exception):
        urllib.request.urlopen("https://openrouter.ai/api/v1/models", timeout=1)


def test_example_config_parses_and_holds_only_placeholders(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.setenv(CONNECTIONS_FILE_ENV, str(root / "connections.example.toml"))
    s = load_settings()
    assert s.provider == "openrouter" and s.api_key is None
    assert set(s.models.values()) == set(s.prices) == {"provider/model-id"}
    assert s.spend_cap_usd == Decimal("0.5")
    assert "your-openrouter-key-here" in (root / ".env.example").read_text(encoding="utf-8")


def test_a_broken_llm_config_does_not_break_the_offline_pipeline(local):
    from ledgercheck.agents import run_pipeline
    from ledgercheck.observability import NullTracer

    local(toml="max_tokens = -5\n")
    assert type(connections.tracer()) is NullTracer
    assert run_pipeline("clean_baseline").decision.outcome.value == "approve"


# --- redirects are refused (real local servers, loopback only) ------------------------


class _LocalServer(ThreadingHTTPServer):
    daemon_threads = True

    def server_bind(self):
        socketserver.TCPServer.server_bind(self)  # skip the reverse DNS lookup


@pytest.fixture
def local_server(monkeypatch):
    """Start loopback HTTP servers whose handler calls ``respond(handler)``; returns base URLs."""
    for name in ("http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY", "all_proxy",
                 "ALL_PROXY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("no_proxy", "*")  # a proxy must not see or reroute these requests
    started = []

    def start(respond):
        class Handler(BaseHTTPRequestHandler):
            def handle_any(self):
                self.rfile.read(int(self.headers.get("Content-Length") or 0))
                respond(self)

            do_GET = do_POST = handle_any

            def log_message(self, *args):
                pass

        server = _LocalServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, args=(0.05,), daemon=True)
        thread.start()
        started.append((server, thread))
        return f"http://127.0.0.1:{server.server_address[1]}"

    yield start
    for server, thread in started:
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)


def _at(base_url):
    return connections.Settings(**{**live_settings().__dict__, "base_url": base_url})


@pytest.mark.parametrize("code", [301, 302, 303, 307, 308])
def test_redirects_are_refused_and_the_key_is_never_forwarded(local_server, code):
    caught = []

    def accept(handler):  # would take the key and answer
        caught.append(handler.headers.get("Authorization") is not None)
        body = json.dumps(completion("ok", cost=0)).encode()
        handler.send_response(200)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)

    target = local_server(accept)

    def redirect(handler):
        handler.send_response(code)
        handler.send_header("Location", f"{target}/api/v1/chat/completions?q=query-part#frag-part")
        handler.send_header("Content-Length", "0")
        handler.end_headers()

    origin = local_server(redirect)
    transport = OpenRouterTransport(_at(f"{origin}/api/v1"),
                                    sleep=lambda s: pytest.fail("retried"))
    with pytest.raises(TransportError, match=f"HTTP {code} \\(a redirect\\)") as exc:
        transport(MODEL, "x")
    message = str(exc.value)
    assert KEY not in message and "query-part" not in message and "frag-part" not in message
    assert caught == []  # the second server never saw a request, so the key was never forwarded
    # Control: the same default opener talking to the accepting server directly does send it.
    assert OpenRouterTransport(_at(f"{target}/api/v1"))(MODEL, "x") == "ok"
    assert caught == [True]


def test_local_config_files_are_ignored_by_git_and_docker():
    root = Path(__file__).resolve().parents[1]
    for ignore in (".gitignore", ".dockerignore"):
        lines = set((root / ignore).read_text(encoding="utf-8").splitlines())
        assert {".env", ".env.*", "connections.local*"} <= lines, ignore
    assert "!.env.example" in (root / ".gitignore").read_text(encoding="utf-8").splitlines()


# --- error text: redact first, then cut; one scrub path ------------------------------

URL = "https://openrouter.ai/api/v1/chat/completions"


def _longest_key_run(text):
    """Length of the longest piece of KEY in ``text`` (so assertions never print the key)."""
    best = 0
    for i in range(len(text)):
        n = 0
        while i + n < len(text) and text[i:i + n + 1] in KEY:
            n += 1
        best = max(best, n)
    return best


def _provider_error(detail, code=401):
    body = json.dumps({"error": {"message": detail}}).encode()
    transport = OpenRouterTransport(live_settings(), opener=FakeOpener(http_error(code, body)))
    with pytest.raises(TransportError) as exc:
        transport(MODEL, "x")
    assert transport.totals.calls == 0  # a non-2xx answer is not charged
    return str(exc.value)


PREFIX = len(f"{URL} returned HTTP 401: ")


@pytest.mark.parametrize("padding", [290, 290 - PREFIX, connections.MAX_ERROR_CHARS - PREFIX - 3])
def test_a_key_straddling_the_error_length_limit_never_leaks(padding):
    message = _provider_error("A" * padding + KEY + " rejected")
    assert _longest_key_run(message) < 8, "a piece of the key reached the error message"


def test_an_echoed_key_fragment_is_masked():
    message = _provider_error(f"key ending in {KEY[-12:]} was rejected")
    assert _longest_key_run(message) < 8 and "was rejected" in message


def test_short_provider_errors_read_clearly_and_lose_control_characters():
    assert _provider_error("Invalid model id", code=400) == (
        f"{URL} returned HTTP 400: Invalid model id")
    message = _provider_error("bad\x1b[31m red\nline\x00", code=400)
    assert "\x1b" not in message and "\n" not in message and "\x00" not in message
    assert message.endswith("bad [31m red line ")


def test_scrub_masks_before_cutting():
    text = "x" * 295 + KEY
    assert _longest_key_run(connections._scrub(text, KEY)) < 8
    assert connections._scrub(text, None) == "x" * 295 + KEY[:5] + "..."  # no key known: cut only


def test_an_unreadable_error_body_is_not_a_traceback():
    exc = urllib.error.HTTPError(URL, 502, "Bad Gateway", email.message.Message(),
                                 BrokenResponse(http.client.IncompleteRead(b"{", 50)))
    transport = OpenRouterTransport(live_settings(), opener=FakeOpener(exc))
    with pytest.raises(TransportError, match="HTTP 502: Bad Gateway"):
        transport(MODEL, "x")


# --- a call that may have been billed is charged its worst case ---------------------


class RawOpener:
    """Replays raw response bodies (bytes), response objects or exceptions."""

    def __init__(self, *replies):
        self.replies, self.calls = list(replies), 0

    def __call__(self, request, timeout):
        self.calls += 1
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return io.BytesIO(reply) if isinstance(reply, bytes) else reply


def _worst(prompt="x", max_tokens=100):
    bound = connections.max_prompt_tokens([{"role": "user", "content": prompt}])
    return bound, (bound * 1 + max_tokens * 2) / Decimal(1_000_000)


POSSIBLY_BILLED = {
    "truncated JSON": b'{"choices": [{"message": {"content": "o',
    "not UTF-8": b'\xff\xfe{"choices": []}',
    "IncompleteRead": BrokenResponse(http.client.IncompleteRead(b'{"cho', 200)),
    "read timeout": BrokenResponse(TimeoutError("timed out")),
    "JSON array": b"[1, 2]",
    "JSON string": b'"just text"',
    "JSON null": b"null",
    "absurd nesting": b"[" * 100_000,
    "timeout waiting for the answer": TimeoutError("timed out"),
    "dropped after sending": http.client.RemoteDisconnected("closed"),
}


@pytest.mark.parametrize("case", list(POSSIBLY_BILLED))
def test_a_possibly_billed_bad_reply_is_charged_the_worst_case(case):
    bound, worst = _worst()
    opener = RawOpener(POSSIBLY_BILLED[case])
    settings = live_settings(**{ENV["spend_cap_usd"]: str(worst * Decimal("1.5"))})
    transport = OpenRouterTransport(settings, opener=opener)
    with pytest.raises(TransportError) as exc:
        transport(MODEL, "x")
    t = transport.totals
    assert (t.calls, t.prompt_tokens, t.completion_tokens, t.cost_usd) == (1, bound, 100, worst)
    assert KEY not in str(exc.value)
    with pytest.raises(SpendCapReached):  # what is left no longer covers another worst case
        transport(MODEL, "x")
    assert opener.calls == 1


def test_an_oversized_reply_is_charged_and_not_read_whole(monkeypatch):
    monkeypatch.setattr(connections, "MAX_RESPONSE_BYTES", 16)
    _, worst = _worst()
    transport = OpenRouterTransport(live_settings(),
                                    opener=RawOpener(json.dumps(completion("ok")).encode()))
    with pytest.raises(TransportError, match="larger than 16 bytes"):
        transport(MODEL, "x")
    assert (transport.totals.calls, transport.totals.cost_usd) == (1, worst)


@pytest.mark.parametrize("error", [
    urllib.error.URLError(ConnectionRefusedError(61, "Connection refused")),
    urllib.error.URLError("nodename nor servname provided"),
    http_error(500, b'{"error": {"message": "upstream"}}'),
    ValueError("Invalid header value"),
])
def test_a_request_that_never_reached_the_provider_is_not_charged(error):
    transport = OpenRouterTransport(live_settings(), opener=RawOpener(error))
    with pytest.raises(TransportError):
        transport(MODEL, "x")
    assert (transport.totals.calls, transport.totals.cost_usd) == (0, 0)


def test_connection_refused_by_a_real_closed_port_is_not_charged(local_server):
    import socket

    with socket.socket() as probe:  # a loopback port with nothing listening
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    transport = OpenRouterTransport(_at(f"http://127.0.0.1:{port}/api/v1"))
    with pytest.raises(TransportError, match="cannot reach"):
        transport(MODEL, "x")
    assert (transport.totals.calls, transport.totals.cost_usd) == (0, 0)


def test_cli_live_judge_reports_a_possibly_billed_call(local, monkeypatch, capsys):
    local(toml=LIVE_TOML)
    monkeypatch.setenv(ENV_FLAG, "1")
    monkeypatch.setenv(API_KEY_ENV, KEY)
    monkeypatch.setattr(connections, "_default_opener", lambda: RawOpener(b'{"choices": ['))
    assert cli_main(["judge", "--live", "--case", CASE]) == judge.EXIT_ERROR
    err = capsys.readouterr().err
    assert "is not JSON" in err and f"live run ({MODEL}): 1 calls" in err and KEY not in err


# --- config read errors are config problems (exit 2), never tracebacks ---------------


def _connections_exit(capsys):
    code = cli_main(["connections"])
    out = capsys.readouterr()
    assert "Traceback" not in out.err
    return code, out.err


def test_a_non_utf8_connections_file_exits_2(local, monkeypatch, capsys):
    (local() / "connections.local.toml").write_bytes(b'provider = "\xff"\n')
    code, err = _connections_exit(capsys)
    assert code == 2 and "not UTF-8" in err
    monkeypatch.setenv(ENV_FLAG, "1")
    assert cli_main(["judge", "--live"]) == judge.EXIT_ERROR  # the live path says the same
    assert "not UTF-8" in capsys.readouterr().err


@pytest.mark.skipif(os.name != "posix" or os.geteuid() == 0, reason="needs a non-root POSIX user")
def test_an_unreadable_connections_file_or_directory_exits_2(local, monkeypatch, capsys):
    root = local(toml=LIVE_TOML)
    path = root / "connections.local.toml"
    path.chmod(0)
    try:
        code, err = _connections_exit(capsys)
        assert code == 2 and "permission denied" in err
    finally:
        path.chmod(0o644)
    locked = root / "locked"
    locked.mkdir()
    (locked / ".env").write_text(f"{API_KEY_ENV}={KEY}\n")
    monkeypatch.setattr(connections, "DEFAULT_CONFIG", locked / "connections.local.toml")
    monkeypatch.setattr(connections, "DOTENV_PATH", locked / ".env")
    locked.chmod(0)
    try:
        code, err = _connections_exit(capsys)  # stat() itself is refused
        assert code == 2 and "permission denied" in err
        assert load_settings().api_key is None  # an unreadable .env is skipped, not a crash
        assert cli_main(["judge", "--case", CASE]) == judge.EXIT_PASS
    finally:
        locked.chmod(0o755)


@pytest.mark.parametrize("make, problem", [
    (lambda p: p.mkdir(), "is a directory"),
    (lambda p: os.mkfifo(p), "not a regular file"),
])
def test_odd_files_at_the_config_path_exit_2_without_hanging(local, capsys, make, problem):
    make(local() / "connections.local.toml")
    code, err = _connections_exit(capsys)
    assert code == 2 and problem in err


def test_odd_dotenv_files_are_skipped(local, capsys):
    os.mkfifo(local() / ".env")  # reading a FIFO would block forever
    assert load_settings().problems == ()
    (local() / "connections.local.toml").write_text(LIVE_TOML)
    assert cli_main(["connections"]) == 0


def test_a_nul_byte_in_the_connections_file_path_exits_2(local, capsys):
    local(dotenv=f"{connections.CONNECTIONS_FILE_ENV}=conn\x00ections.toml\n")
    code, err = _connections_exit(capsys)
    assert code == 2 and "not a usable file path" in err
    assert "\x00" not in capsys.readouterr().out


# Built at runtime: a URL with a user and password is what is being tested.
USERINFO_URL = "https:/" + "/user:pw-secret" + "@host.example/v1"
ODD_TOML = [
    ("provider = false\n", "provider must be a string"),
    ("api_key = 12345\n", "api_key in"),
    ("[models]\nsmall = 5\n", "model small must be a string"),
    ('models = "vendor/x"\n', "[models] must be a table"),
    ('[models]\nmedium = "vendor/x"\n', "[models] has unknown keys"),
    ("base_url = 5\n", "base_url must be an https"),
    (f'base_url = "{USERINFO_URL}"\n', "base_url must be an https"),
    ('base_url = "https://host.example/v1?x=1"\n', "base_url must be an https"),
    ('base_url = "https://[::1/v1"\n', "base_url must be an https"),
    ("spend_cap_usd = nan\n", "spend_cap_usd must be a finite number"),
    ("spend_cap_usd = inf\n", "spend_cap_usd must be a finite number"),
    ("spend_cap_usd = -1\n", "spend_cap_usd must be a finite number"),
    ("spend_cap_usd = 1e300\n", "spend_cap_usd must be a finite number"),
    ('spend_cap_usd = "abc"\n', "spend_cap_usd must be a number"),
    ("spend_cap_usd = true\n", "spend_cap_usd must be a number"),
    ("spend_cap_usd = {a = 1}\n", "spend_cap_usd must be a number"),
    ("spend_cap_usd = [1]\n", "spend_cap_usd must be a number"),
    ("max_tokens = 1.5\n", "max_tokens must be a whole number"),
    ("max_tokens = true\n", "max_tokens must be a whole number"),
    ('max_tokens = "²"\n', "max_tokens must be a whole number"),
    ("max_tokens = 100000000\n", "max_tokens must be a whole number"),
    (f'max_tokens = "{"9" * 5000}"\n', "max_tokens must be a whole number"),
    (f'[prices."{MODEL}"]\nprompt = {{a = 1}}\ncompletion = 1\n', "prompt price must be a number"),
    (f'[prices."{MODEL}"]\nprompt = nan\ncompletion = 1\n', "prompt price must be a finite"),
    (f'[prices."{MODEL}"]\nprompt = 1\ncompletion = 1e300\n', "completion price must be a"),
]


@pytest.mark.parametrize("toml, problem", ODD_TOML)
def test_odd_value_types_are_named_problems_not_crashes(local, capsys, toml, problem):
    local(toml=toml)
    assert any(problem in p for p in load_settings().problems), load_settings().problems
    code, err = _connections_exit(capsys)
    assert code == 2 and problem in err and "pw-secret" not in err
    assert type(connections.router()) is MockRouter  # the offline default still works


@pytest.mark.parametrize("name, value", [
    ("spend_cap_usd", "1e999999999"), ("spend_cap_usd", "sNaN"), ("spend_cap_usd", "Infinity"),
    ("max_tokens", "²"), ("max_tokens", "1e3"), ("max_tokens", "-5"),
])
def test_odd_numeric_env_values_are_named_problems(monkeypatch, name, value):
    monkeypatch.setenv(ENV[name], value)
    assert any(name in p for p in load_settings().problems)


def test_a_key_pasted_into_another_setting_is_masked(monkeypatch, capsys):
    monkeypatch.setenv(API_KEY_ENV, KEY)
    monkeypatch.setenv(ENV["max_tokens"], KEY)
    problems = load_settings().problems
    assert problems and all(_longest_key_run(p) < 8 for p in problems)
    assert cli_main(["connections"]) == 2
    out = capsys.readouterr()
    assert _longest_key_run(out.out + out.err) < 8


def test_a_non_ascii_key_is_refused(monkeypatch):
    monkeypatch.setenv(ENV_FLAG, "1")
    monkeypatch.setenv(API_KEY_ENV, KEY.replace("real", "réal"))
    with pytest.raises(ConnectionConfigError, match="non-ASCII"):
        connections.require_live()
