"""Model routing: tier mapping, configured models, the live gate, no network, docstring drift."""

import re
import socket
from pathlib import Path

import pytest

from ledgercheck import connections
from ledgercheck.agents import routing
from ledgercheck.agents.llm_client import API_KEY_ENV, ENV_FLAG, LiveLLMDisabled
from ledgercheck.agents.routing import (
    MOCK_MODELS,
    MODEL_ENV,
    TASK_TIERS,
    MockRouter,
    OpenRouterRouter,
    Route,
    TaskKind,
    Tier,
    resolve_models,
)
from ledgercheck.connections import ConnectionConfigError
from ledgercheck.eval.judge import LiveJudge

SMALL, LARGE = MOCK_MODELS[Tier.SMALL], MOCK_MODELS[Tier.LARGE]
LIVE_SMALL, LIVE_LARGE = "vendor/small-x", "vendor/large-y"
PROVIDER_ENV = connections.SETTINGS["provider"][0]
CAP_ENV = connections.SETTINGS["spend_cap_usd"][0]
OPEN = {
    ENV_FLAG: "1", API_KEY_ENV: "fake-not-a-real-key", PROVIDER_ENV: "openrouter",
    MODEL_ENV[Tier.SMALL]: LIVE_SMALL, MODEL_ENV[Tier.LARGE]: LIVE_LARGE, CAP_ENV: "0.50",
}


@pytest.fixture
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        pytest.fail("network call attempted")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)


# --- mapping and overrides --------------------------------------------------


@pytest.mark.parametrize(
    "task, tier, model, fallback",
    [
        (TaskKind.EXTRACTION, Tier.SMALL, SMALL, LARGE),
        (TaskKind.RERANK, Tier.LARGE, LARGE, None),
        (TaskKind.APPROVAL_REASONING, Tier.LARGE, LARGE, None),
        (TaskKind.JUDGE, Tier.LARGE, LARGE, None),
    ],
)
def test_mock_router_routes_each_task_kind(task, tier, model, fallback):
    assert MockRouter({}).route(task) == Route(task, tier, model, fallback)


def test_every_task_kind_has_a_tier_and_every_tier_a_model_and_env():
    assert set(TASK_TIERS) == set(TaskKind)
    assert set(MOCK_MODELS) == set(MODEL_ENV) == set(Tier)


def test_route_accepts_the_task_value_string():
    assert MockRouter({}).route("judge").task is TaskKind.JUDGE


def test_env_overrides_are_stripped():
    env = {MODEL_ENV[Tier.SMALL]: " vendor/small-x ", MODEL_ENV[Tier.LARGE]: "vendor/large-y"}
    assert resolve_models(env) == {Tier.SMALL: "vendor/small-x", Tier.LARGE: "vendor/large-y"}
    route = MockRouter(env).route(TaskKind.EXTRACTION)
    assert (route.model, route.fallback) == ("vendor/small-x", "vendor/large-y")


@pytest.mark.parametrize("value", ["", "  ", "\t\n"])
def test_blank_env_override_falls_back_to_default(value):
    env = {name: value for name in MODEL_ENV.values()}
    assert resolve_models(env) == dict(MOCK_MODELS)


def test_os_environ_is_the_default_env(monkeypatch):
    monkeypatch.setenv(MODEL_ENV[Tier.LARGE], "vendor/from-shell")
    assert MockRouter().route(TaskKind.RERANK).model == "vendor/from-shell"


def test_same_model_for_both_tiers_has_no_fallback():
    env = {MODEL_ENV[Tier.SMALL]: "vendor/one", MODEL_ENV[Tier.LARGE]: "vendor/one"}
    assert MockRouter(env).route(TaskKind.EXTRACTION).fallback is None


def test_mock_router_records_calls_and_returns_canned_replies(no_network):
    router = MockRouter({}, replies={TaskKind.JUDGE: "5/5/5"})
    assert router.complete(TaskKind.JUDGE, "score this") == "5/5/5"
    assert router.complete(TaskKind.EXTRACTION, "Invoice INV-1") == ""
    assert router.calls == [
        (TaskKind.JUDGE, LARGE, "score this"),
        (TaskKind.EXTRACTION, SMALL, "Invoice INV-1"),
    ]


# --- live router: the existing spend gate ----------------------------------


@pytest.mark.parametrize(
    "env",
    [
        {API_KEY_ENV: "sk-test"},  # a key without the flag is not consent
        {ENV_FLAG: "1"},  # the flag without a key
        {ENV_FLAG: "1", API_KEY_ENV: "  "},  # a blank key
    ],
)
def test_live_router_refuses_without_the_gate(env):
    def transport(model, prompt):
        pytest.fail("transport reached with the gate closed")

    with pytest.raises(LiveLLMDisabled) as exc:
        OpenRouterRouter(env, transport=transport)
    assert "sk-test" not in str(exc.value)


def test_live_router_off_by_default(monkeypatch):
    with pytest.raises(LiveLLMDisabled, match=ENV_FLAG):
        OpenRouterRouter()


def test_live_router_needs_a_spend_cap_for_its_own_transport(no_network):
    env = {k: v for k, v in OPEN.items() if k != CAP_ENV}
    with pytest.raises(ConnectionConfigError, match="spend cap"):
        OpenRouterRouter(env)
    assert OpenRouterRouter(OPEN).totals.calls == 0


def test_live_router_sends_the_routed_model_to_the_transport(no_network):
    sent = []
    router = OpenRouterRouter(OPEN, transport=lambda model, prompt: sent.append(model) or "ok")
    assert router.complete(TaskKind.EXTRACTION, "x") == "ok"
    assert router.complete(TaskKind.RERANK, "y") == "ok"
    assert sent == [LIVE_SMALL, LIVE_LARGE]
    assert router.route(TaskKind.EXTRACTION).fallback == LIVE_LARGE
    assert router.totals is None  # a bare callable reports no totals


def test_live_router_has_no_default_model_ids(no_network):
    env = {k: v for k, v in OPEN.items() if k != MODEL_ENV[Tier.LARGE]}
    router = OpenRouterRouter(env, transport=lambda model, prompt: pytest.fail("sent"))
    assert router.route(TaskKind.EXTRACTION) == Route(TaskKind.EXTRACTION, Tier.SMALL,
                                                       LIVE_SMALL, None)
    with pytest.raises(ConnectionConfigError, match=MODEL_ENV[Tier.LARGE]) as exc:
        router.complete(TaskKind.JUDGE, "x")
    assert "connections.local.toml" in str(exc.value)


closers = pytest.mark.parametrize(
    "close",
    [
        lambda env: env.update({ENV_FLAG: "0"}),
        lambda env: env.pop(API_KEY_ENV),
        lambda env: env.update({API_KEY_ENV: "  "}),
    ],
    ids=["flag-off", "key-removed", "key-blank"],
)


@closers
def test_live_router_rechecks_the_gate_on_complete(close, no_network):
    def transport(model, prompt):
        pytest.fail("transport reached with the gate closed")

    env = dict(OPEN)
    router = OpenRouterRouter(env, transport=transport)
    close(env)
    with pytest.raises(LiveLLMDisabled) as exc:
        router.complete(TaskKind.EXTRACTION, "x")
    assert OPEN[API_KEY_ENV] not in str(exc.value)


@closers
def test_live_router_rechecks_the_gate_on_route(close, no_network):
    env = dict(OPEN)
    router = OpenRouterRouter(env)
    assert router.route(TaskKind.EXTRACTION).model == LIVE_SMALL
    close(env)
    with pytest.raises(LiveLLMDisabled) as exc:
        router.route(TaskKind.EXTRACTION)
    assert OPEN[API_KEY_ENV] not in str(exc.value)


def test_live_router_rechecks_os_environ_on_route(monkeypatch, no_network):
    for name, value in OPEN.items():
        monkeypatch.setenv(name, value)
    router = OpenRouterRouter()
    assert router.route(TaskKind.JUDGE).model == LIVE_LARGE
    monkeypatch.delenv(API_KEY_ENV)
    with pytest.raises(LiveLLMDisabled, match=API_KEY_ENV):
        router.route(TaskKind.JUDGE)


def test_live_router_rechecks_os_environ_on_complete(monkeypatch, no_network):
    def transport(model, prompt):
        pytest.fail("transport reached with the gate closed")

    for name, value in OPEN.items():
        monkeypatch.setenv(name, value)
    router = OpenRouterRouter(transport=transport)
    monkeypatch.delenv(API_KEY_ENV)
    with pytest.raises(LiveLLMDisabled, match=API_KEY_ENV):
        router.complete(TaskKind.JUDGE, "x")


# --- connections.router(): the provider is config ---------------------------


@pytest.mark.parametrize(
    "env",
    [{}, {API_KEY_ENV: "sk-test"}, {ENV_FLAG: "1", API_KEY_ENV: "sk-test"}, {PROVIDER_ENV: "mock"}],
)
def test_router_is_mock_unless_the_provider_is_openrouter(env, no_network):
    router = connections.router(env)
    assert type(router) is MockRouter
    assert router.complete(TaskKind.JUDGE, "x") == ""


def test_router_reads_os_environ(monkeypatch):
    assert type(connections.router()) is MockRouter
    for name, value in OPEN.items():
        monkeypatch.setenv(name, value)
    assert type(connections.router()) is OpenRouterRouter


def test_openrouter_provider_is_live_and_still_gated(no_network):
    assert type(connections.router(OPEN)) is OpenRouterRouter
    with pytest.raises(LiveLLMDisabled):
        connections.router({**OPEN, ENV_FLAG: "0"})
    with pytest.raises(ConnectionConfigError, match="unknown provider"):
        connections.router({PROVIDER_ENV: "elsewhere"})


# --- caller wiring: LiveJudge asks the router ------------------------------


def test_live_judge_takes_the_judge_model_from_the_router(no_network):
    assert LiveJudge(OPEN).model == LIVE_LARGE
    assert LiveJudge({**OPEN, MODEL_ENV[Tier.LARGE]: "vendor/judge"}).model == "vendor/judge"
    assert LiveJudge({}).model == LARGE  # provider mock


# --- drift: the module docstring matches the constants ---------------------

DOC = routing.__doc__ or ""


def test_docstring_task_table_matches_task_tiers():
    rows = dict(re.findall(r"^ +``(\w+)`` +(small|large) ", DOC, re.MULTILINE))
    assert rows == {task.value: tier.value for task, tier in TASK_TIERS.items()}


def test_docstring_mock_models_match_constants():
    rows = dict(re.findall(r"^ +(small|large) +``([^`]+)``$", DOC, re.MULTILINE))
    assert rows == {tier.value: model for tier, model in MOCK_MODELS.items()}


def test_connections_is_the_only_production_router_construction():
    package = Path(routing.__file__).parents[1]
    call = re.compile(r"\b(\w+Router|\w*Transport)\(")
    builds = {
        str(path.relative_to(package)): call.findall(path.read_text())
        for path in package.rglob("*.py")
        if path != Path(routing.__file__)
    }
    assert {name: found for name, found in builds.items() if found} == {
        "connections.py": ["OpenRouterTransport", "MockRouter", "OpenRouterTransport",
                           "OpenRouterRouter"],
    }
    assert "connections.router()``" in DOC and "the only production caller is ``LiveJudge``" in DOC


def test_docstring_names_env_vars_and_runbook_sections():
    for name in (*MODEL_ENV.values(), ENV_FLAG, API_KEY_ENV):
        assert f"``{name}``" in DOC
    for heading in ("Runbook: rate limits", "Runbook: context-window failures"):
        assert heading in DOC
    assert DOC.count("Today:") == 2 and DOC.count("Planned") == 2
