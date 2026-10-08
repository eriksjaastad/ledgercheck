"""Connections: settings resolution, the spend gate and cap, the OpenRouter transport (faked
opener, no sockets), provider swapping by config, and the key never leaking."""

import email.message
import io
import json
import re
import urllib.error
import urllib.request
from decimal import Decimal
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
    with pytest.raises(ConnectionConfigError, match="whitespace or control characters") as exc:
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
                   "doppler run --", "git config core.hooksPath .githooks", "spend_cap_usd"):
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
    opener = FakeOpener(completion(prompt_tokens=1000, completion_tokens=500))
    transport = OpenRouterTransport(live_settings(), opener=opener)
    transport(MODEL, "x")
    assert transport.totals.cost_usd == Decimal("0.002")  # 1000 * $1/M + 500 * $2/M


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


@pytest.mark.parametrize("reply", [
    urllib.error.URLError("no route"), {"choices": []}, {"choices": [{"message": {}}]},
    completion(prompt_tokens="many"),
])
def test_unreachable_or_odd_responses_are_transport_errors(reply):
    with pytest.raises(TransportError):
        OpenRouterTransport(live_settings(), opener=FakeOpener(reply))(MODEL, "x")


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
    monkeypatch.setattr(urllib.request, "urlopen", opener)
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
    monkeypatch.setattr(urllib.request, "urlopen", opener)
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
