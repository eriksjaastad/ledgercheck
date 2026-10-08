"""ledgercheck connections — every outside connection is configured and built here.

This module is the one place that reads connection settings (environment
variables, a local ``.env`` file, a local ``connections.local.toml``) and
builds the clients that talk to the outside: the LLM transport (OpenRouter,
over ``urllib``), the Langfuse client behind the tracer, and the run store.
Everything else receives those objects or asks this module for them, so
swapping a provider is a config change, not a code change.

Out of the box nothing is configured: the provider is ``mock``, tracing is
off, and nothing leaves your machine.

Settings
--------
::

    setting       environment variable         connections.local.toml
    provider      LEDGERCHECK_LLM_PROVIDER     provider = "mock" | "openrouter"
    API key       OPENROUTER_API_KEY           api_key = "..."
    small model   LEDGERCHECK_MODEL_SMALL      [models] small = "provider/model-id"
    large model   LEDGERCHECK_MODEL_LARGE      [models] large = "provider/model-id"
    base URL      LEDGERCHECK_LLM_BASE_URL     base_url (default https://openrouter.ai/api/v1)
    spend cap     LEDGERCHECK_SPEND_CAP_USD    spend_cap_usd (USD per run; required to go live)
    max tokens    LEDGERCHECK_MAX_TOKENS       max_tokens (per reply, default 1024)
    prices        (file only)                  [prices."provider/model-id"] prompt, completion

Prices are USD per million tokens and are required for every model a live run
uses. There are no built-in model ids or prices: you choose the models and
copy their prices from your provider. Tracing reads ``LANGFUSE_PUBLIC_KEY``,
``LANGFUSE_SECRET_KEY`` and ``LANGFUSE_HOST`` from the environment or ``.env``
(see ``ledgercheck.observability``).

Where settings come from (first match wins)
-------------------------------------------
1. The process environment: ``export OPENROUTER_API_KEY=...``, CI secrets,
   ``docker run -e``, or a secrets manager that injects variables, e.g.
   ``doppler run -- ledgercheck judge --live --case <case_id>``.
2. A ``.env`` file in the working directory: ``KEY=VALUE`` lines, ``#``
   comments, optional ``export`` and quotes. It never overrides a variable
   that is already set, and lines it cannot parse are skipped.
3. ``connections.local.toml`` in the working directory, or the file named by
   ``LEDGERCHECK_CONNECTIONS_FILE``. It may hold the key as well.

``.env`` and ``connections.local*`` are gitignored; start from the committed
``.env.example`` and ``connections.example.toml``. The key is never printed,
logged, put in an error message, a repr, a run record or a trace.

The spend gate
--------------
Live calls need provider ``openrouter``, a key, and ``LEDGERCHECK_LLM=1`` in the
process environment. The flag is read from the environment only, never from
a file, so nothing spends by accident. Each live run also has a hard cap:
before every call the worst case (an upper bound on prompt tokens, one per
UTF-8 byte of the messages plus a fixed overhead, plus ``max_tokens``, at
your prices) must fit in what is left of ``spend_cap_usd``, or the call is
refused. After the call its actual cost is added: the provider's reported
cost, else tokens times prices with each count capped at its pre-call bound,
so an unreported cost never exceeds that worst case. HTTP 429 is retried up
to 3 times, honouring ``Retry-After`` (capped at 30 s).

Checking your setup
-------------------
``ledgercheck connections`` prints the resolved settings and where each one
came from, and exits 2 if a setting is invalid. The key only shows as
``set (from env|.env|<file>)`` or ``missing``. An invalid setting never
stops the offline mock path; it stops a live run before any request.

Not here on purpose: ``ledgercheck serve`` (``ledgercheck.web``) listens on a
loopback socket and parses requests with ``urllib.parse``; it connects to
nothing. The run store's file I/O stays in ``ledgercheck.run_store``; this
module only picks its root.

Before you push, enable the leak guard hook once per clone:
``git config core.hooksPath .githooks``. It runs ``scripts/leak_guard.py``
on the commits being pushed and stops the push if any of them adds something
that looks like a key, or a local ``.env`` or ``connections.local*`` file.
"""

from __future__ import annotations

import argparse
import http.client
import json
import math
import os
import re
import sys
import time
import tomllib
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Mapping

from ledgercheck.observability import LangfuseTracer, LangfuseUnavailable, NullTracer, Tracer
from ledgercheck.run_store import DEFAULT_ROOT, RunStore

ENV_FLAG = "LEDGERCHECK_LLM"
API_KEY_ENV = "OPENROUTER_API_KEY"
CONNECTIONS_FILE_ENV = "LEDGERCHECK_CONNECTIONS_FILE"
PUBLIC_KEY_ENV = "LANGFUSE_PUBLIC_KEY"
SECRET_KEY_ENV = "LANGFUSE_SECRET_KEY"
HOST_ENV = "LANGFUSE_HOST"
PROVIDERS = ("mock", "openrouter")
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MAX_TOKENS = 1024
MAX_RETRIES = 3
MAX_BACKOFF_S = 30.0
TIMEOUT_S = 60.0
# Read from the working directory at call time; tests point these elsewhere.
DOTENV_PATH = Path(".env")
DEFAULT_CONFIG = Path("connections.local.toml")

# setting -> (environment variable, path in connections.local.toml)
SETTINGS: Mapping[str, tuple[str, tuple[str, ...]]] = {
    "provider": ("LEDGERCHECK_LLM_PROVIDER", ("provider",)),
    "api_key": (API_KEY_ENV, ("api_key",)),
    "model_small": ("LEDGERCHECK_MODEL_SMALL", ("models", "small")),
    "model_large": ("LEDGERCHECK_MODEL_LARGE", ("models", "large")),
    "base_url": ("LEDGERCHECK_LLM_BASE_URL", ("base_url",)),
    "spend_cap_usd": ("LEDGERCHECK_SPEND_CAP_USD", ("spend_cap_usd",)),
    "max_tokens": ("LEDGERCHECK_MAX_TOKENS", ("max_tokens",)),
}
_FILE_KEYS = {"provider", "api_key", "models", "base_url", "spend_cap_usd", "max_tokens", "prices"}
_MTOK = Decimal(1_000_000)
# NAME=value, NAME="value", NAME='value'; an unquoted value ends at " #".
_DOTENV_LINE = re.compile(
    r"""\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*"""
    r"""(?:"([^"]*)"|'([^']*)'|(.*?))\s*(?:\s#.*)?""")
_WHERE_KEY = (f"export {API_KEY_ENV}, put it in .env, or set api_key in connections.local.toml "
              "(see ledgercheck connections --help)")


class LiveLLMDisabled(RuntimeError):
    """A live LLM call was requested but is not explicitly enabled."""


class ConnectionConfigError(LiveLLMDisabled):
    """A connection setting is missing or invalid."""


class SpendCapReached(LiveLLMDisabled):
    """The next call could exceed what is left of the run's spend cap."""


class TransportError(RuntimeError):
    """The provider answered with an error or could not be reached."""


def _read_dotenv(path: Path) -> dict[str, str]:
    """``KEY=VALUE`` lines with optional ``export``, quotes and `` # comment``.

    Tolerant on purpose: ``.env`` is often shared with other tools, so a line
    this parser does not understand is skipped, never an error.
    """
    if not (path.is_file() and os.access(path, os.R_OK)):
        return {}
    values = {}
    for line in path.read_bytes().decode("utf-8", errors="replace").splitlines():
        m = _DOTENV_LINE.fullmatch(line)
        if m:
            name, double, single, bare = m.groups()
            values[name] = next(v for v in (double, single, bare) if v is not None)
    return values


def _read_config(path: Path, required: bool) -> dict[str, Any]:
    if not path.is_file():
        if required:
            raise ConnectionConfigError(f"{CONNECTIONS_FILE_ENV} names {path}, which does not "
                                        "exist")
        return {}
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ConnectionConfigError(f"{path}: {exc}") from None
    unknown = set(data) - _FILE_KEYS
    if unknown:
        raise ConnectionConfigError(f"{path}: unknown keys {sorted(unknown)}; "
                                    f"allowed: {sorted(_FILE_KEYS)}")
    return data


class _Sources:
    """The environment, then ``.env``, then the connections file. No repr: it holds keys."""

    def __init__(self, env: Mapping[str, str] | None, *, with_file: bool = True) -> None:
        self.config, self.config_path, self.problems = {}, None, []
        if env is not None:  # an explicit mapping is the whole configuration
            self.env, self.dotenv = env, {}
            return
        self.env, self.dotenv = os.environ, _read_dotenv(DOTENV_PATH)
        if not with_file:
            return
        named = self.value(CONNECTIONS_FILE_ENV)
        self.config_path = Path(named) if named else DEFAULT_CONFIG
        try:
            self.config = _read_config(self.config_path, required=bool(named))
        except ConnectionConfigError as exc:  # reported by require_live and `connections`
            self.problems.append(str(exc))

    def value(self, name: str) -> str | None:
        return (self.find(name) or (None, None))[0]

    def find(self, name: str, path: tuple[str, ...] = ()) -> tuple[Any, str] | None:
        """``(value, source)`` of the first non-blank setting, or ``None``."""
        for values, source in ((self.env, "env"), (self.dotenv, ".env")):
            value = values.get(name)
            if value is not None and value.strip():
                return value.strip(), source
        node: Any = self.config
        for part in path:
            node = node.get(part) if isinstance(node, Mapping) else None
        if isinstance(node, str):
            node = node.strip()
        if path and node is not None and node != "":
            return node, str(self.config_path)
        return None

    def blank(self, name: str) -> bool:
        return any(v is not None and not v.strip() for v in (self.env.get(name),
                                                             self.dotenv.get(name)))


def _decimal(value: Any, what: str) -> Decimal:
    try:
        number = Decimal(str(value))
    except InvalidOperation:
        raise ConnectionConfigError(f"{what} must be a number, got {value!r}") from None
    if isinstance(value, bool) or not number.is_finite() or number < 0:
        raise ConnectionConfigError(f"{what} must be a finite number >= 0, got {value!r}")
    return number


def _max_tokens(raw: Any) -> int:
    text = str(DEFAULT_MAX_TOKENS if raw is None else raw)
    if not text.isdigit() or int(text) <= 0:
        raise ConnectionConfigError(f"max_tokens must be a whole number > 0, got {raw!r}")
    return int(text)


def _base_url(raw: Any) -> str:
    url = str(raw or DEFAULT_BASE_URL).rstrip("/")
    if not url.startswith("https://"):  # the key travels in a header
        raise ConnectionConfigError(f"base_url must start with https://, got {url!r}")
    return url


def _cap(raw: Any) -> Decimal | None:
    return None if raw is None else _decimal(raw, "spend_cap_usd")


def _parsed(problems: list[str], parse: Callable[[Any], Any], raw: Any, fallback: Any) -> Any:
    """``parse(raw)``, or ``fallback`` with the error added to ``problems``."""
    try:
        return parse(raw)
    except ConnectionConfigError as exc:
        problems.append(str(exc))
    return fallback


def _key_ok(key: str) -> bool:
    return not any(c.isspace() or not c.isprintable() for c in key)


def _prices(raw: Any) -> dict[str, tuple[Decimal, Decimal]]:
    if not isinstance(raw, Mapping):
        raise ConnectionConfigError("[prices] must map model ids to {prompt, completion}")
    prices = {}
    for model, entry in raw.items():
        if not isinstance(entry, Mapping) or set(entry) != {"prompt", "completion"}:
            raise ConnectionConfigError(f'[prices."{model}"] needs exactly prompt and completion '
                                        "(USD per million tokens)")
        prices[model] = (_decimal(entry["prompt"], f"{model} prompt price"),
                         _decimal(entry["completion"], f"{model} completion price"))
    return prices


@dataclass(frozen=True)
class Settings:
    """Resolved connection settings; ``sources`` says where each came from."""

    provider: str
    models: Mapping[str, str]  # tier ("small"/"large") -> model id, only those configured
    base_url: str
    spend_cap_usd: Decimal | None
    max_tokens: int
    prices: Mapping[str, tuple[Decimal, Decimal]]
    live_flag: bool
    sources: Mapping[str, str]
    config_file: Path | None
    api_key: str | None = field(default=None, repr=False)
    api_key_blank: bool = False
    problems: tuple[str, ...] = ()  # invalid settings; only a live run refuses on them


def load_settings(env: Mapping[str, str] | None = None) -> Settings:
    """Resolve the settings. ``env`` defaults to the process environment plus the
    local files; an explicit mapping is used alone (no files are read).

    Never raises on bad values: an invalid setting falls back to its default
    and is listed in ``problems``, so the offline mock path cannot be broken
    by local config. ``require_live`` refuses while there are problems.
    """
    src = _Sources(env)
    problems = src.problems
    raw, sources = {}, {}
    for name, (var, path) in SETTINGS.items():
        found = src.find(var, path)
        raw[name], sources[name] = found if found else (None, "default")
    if "prices" in src.config:
        sources["prices"] = str(src.config_path)
    key = None if raw["api_key"] is None else str(raw["api_key"])
    if key is not None and not _key_ok(key):
        problems.append(f"{API_KEY_ENV} (from {sources['api_key']}) contains whitespace or "
                        "control characters; check for a stray space, quote or line break")
    return Settings(
        provider=str(raw["provider"] or "mock").strip().lower(),
        models={tier: str(raw[f"model_{tier}"]).strip()
                for tier in ("small", "large") if raw[f"model_{tier}"]},
        base_url=_parsed(problems, _base_url, raw["base_url"], DEFAULT_BASE_URL),
        spend_cap_usd=_parsed(problems, _cap, raw["spend_cap_usd"], None),
        max_tokens=_parsed(problems, _max_tokens, raw["max_tokens"], DEFAULT_MAX_TOKENS),
        prices=_parsed(problems, _prices, src.config.get("prices", {}), {}),
        live_flag=src.env.get(ENV_FLAG) == "1",
        sources=sources,
        config_file=src.config_path,
        api_key=key,
        api_key_blank=src.blank(API_KEY_ENV),
        problems=tuple(problems),
    )


def require_live(env: Mapping[str, str] | None = None) -> Settings:
    """The spend gate: the settings if live calls are allowed, else ``LiveLLMDisabled``."""
    settings = load_settings(env)
    if not settings.live_flag:
        raise LiveLLMDisabled(f"live LLM calls are off: set {ENV_FLAG}=1 and {API_KEY_ENV} to "
                              "enable it")
    if settings.problems:
        raise ConnectionConfigError("; ".join(settings.problems))
    if settings.api_key is None:
        if settings.api_key_blank:
            raise LiveLLMDisabled(f"{ENV_FLAG}=1 but {API_KEY_ENV} is blank")
        raise LiveLLMDisabled(f"{ENV_FLAG}=1 but {API_KEY_ENV} is not set: {_WHERE_KEY}")
    return settings


def env_flag(name: str) -> bool:
    """A feature flag from the process environment: on only when exactly ``"1"``."""
    return os.environ.get(name) == "1"


# --- LLM transport -----------------------------------------------------------


MESSAGE_OVERHEAD_TOKENS = 16  # role and chat-template markers, per message


def max_prompt_tokens(messages: list[Mapping[str, str]]) -> int:
    """An upper bound on the prompt tokens ``messages`` can cost.

    Byte-level tokenizers emit at most one token per UTF-8 byte, so the bound
    is the byte length of every message's role and content plus a fixed
    per-message overhead. It is loose on purpose: the cap must hold for any
    text, including punctuation-heavy text that tokenizes badly.
    """
    return sum(len(m["role"].encode("utf-8")) + len(m["content"].encode("utf-8"))
               + MESSAGE_OVERHEAD_TOKENS for m in messages)


@dataclass
class Totals:
    """What a live run has used so far."""

    cap_usd: Decimal
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: Decimal = Decimal(0)
    retries: int = 0

    def line(self) -> str:
        return (f"{self.calls} calls, {self.prompt_tokens} prompt + {self.completion_tokens} "
                f"completion tokens, ${self.cost_usd} of ${self.cap_usd} cap"
                + (f", {self.retries} rate-limit retries" if self.retries else ""))


Opener = Callable[..., Any]  # urllib.request.urlopen(request, timeout=...)


class OpenRouterTransport:
    """``(model, prompt) -> reply`` via ``POST {base_url}/chat/completions``, capped per run.

    OpenRouter returns ``usage`` (``prompt_tokens``, ``completion_tokens`` and
    ``cost`` in USD) on every non-streaming response; that cost is what is
    added to ``totals``, even when the reply itself turns out to be unusable.
    A valid reported cost is recorded as billed. Without one, the charge is
    tokens times your prices, where a token count that is missing, invalid
    or above its pre-call bound (``max_prompt_tokens`` for the prompt,
    ``max_tokens`` for the reply) counts as that bound. So a call without a
    reported cost is never charged more than the worst case it was admitted
    under. An invalid reported cost (negative, not a number, infinite,
    absurdly large) is treated as missing.

    Build one per run (``totals`` is that run's spend). ``opener`` and
    ``sleep`` default to ``urllib.request.urlopen`` and ``time.sleep``; tests
    inject fakes.
    """

    name = "openrouter"

    def __init__(self, settings: Settings, *, opener: Opener | None = None,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        if not settings.api_key:
            raise LiveLLMDisabled(f"{API_KEY_ENV} is not set: {_WHERE_KEY}")
        if not _key_ok(settings.api_key):
            raise ConnectionConfigError(f"{API_KEY_ENV} contains whitespace or control characters")
        if settings.spend_cap_usd is None or settings.spend_cap_usd <= 0:
            raise ConnectionConfigError(
                "a live run needs a spend cap: set LEDGERCHECK_SPEND_CAP_USD or spend_cap_usd "
                "in connections.local.toml (USD, > 0)")
        self._key = settings.api_key
        self.url = f"{settings.base_url}/chat/completions"
        self._max_tokens, self._prices = settings.max_tokens, settings.prices
        self._open = urllib.request.urlopen if opener is None else opener
        self._sleep = sleep
        self.totals = Totals(settings.spend_cap_usd)

    def __repr__(self) -> str:
        return f"OpenRouterTransport(url={self.url!r}, totals={self.totals!r})"

    def __call__(self, model: str, prompt: str) -> str:
        if model not in self._prices:
            raise ConnectionConfigError(
                f'no price for {model!r}: add [prices."{model}"] prompt = ..., completion = ... '
                "(USD per million tokens) to connections.local.toml")
        prompt_price, completion_price = self._prices[model]
        messages = [{"role": "user", "content": prompt}]
        bound = max_prompt_tokens(messages)
        try:
            worst = (bound * prompt_price + self._max_tokens * completion_price) / _MTOK
            left = self.totals.cap_usd - self.totals.cost_usd
            refuse = not (left.is_finite() and worst <= left)  # fail closed on NaN/infinity
        except ArithmeticError:  # decimal errors included
            raise SpendCapReached("spend cap: the remaining budget cannot be computed") from None
        if refuse:
            raise SpendCapReached(f"spend cap: the next call could cost up to ${worst:.6f} but "
                                  f"${left:.6f} of ${self.totals.cap_usd} is left")
        data = self._post({"model": model, "messages": messages, "max_tokens": self._max_tokens})
        # Count the call before looking at the reply: a 200 response is billed
        # even when its choices are missing or malformed. Token counts are
        # clamped to their bounds, so a computed charge never exceeds `worst`.
        usage = data.get("usage") if isinstance(data, Mapping) else None
        if not isinstance(usage, Mapping):
            usage = {}
        p_tok = _count(usage.get("prompt_tokens"), bound)
        c_tok = _count(usage.get("completion_tokens"), self._max_tokens)
        cost = _reported_cost(usage.get("cost"))
        if cost is None:
            cost = min((p_tok * prompt_price + c_tok * completion_price) / _MTOK, worst)
        totals = self.totals
        totals.calls += 1
        totals.prompt_tokens += p_tok
        totals.completion_tokens += c_tok
        totals.cost_usd += cost
        try:
            reply = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise TransportError(f"unexpected response shape from {self.url}") from None
        if not isinstance(reply, str):
            raise TransportError(f"no text reply from {self.url}")
        return reply

    def _post(self, payload: Mapping[str, Any]) -> Any:
        body = json.dumps(payload).encode("utf-8")
        headers = {"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"}
        for attempt in range(MAX_RETRIES + 1):
            request = urllib.request.Request(self.url, data=body, headers=headers, method="POST")
            try:
                with self._open(request, timeout=TIMEOUT_S) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                if exc.code != 429 or attempt == MAX_RETRIES:
                    raise TransportError(self._redact(_http_message(exc, self.url))) from None
                self.totals.retries += 1
                self._sleep(_backoff(exc.headers.get("Retry-After") if exc.headers else None,
                                     attempt))
            except (urllib.error.URLError, OSError) as exc:
                reason = getattr(exc, "reason", exc)
                raise TransportError(self._redact(f"cannot reach {self.url}: {reason}")) from None
            except http.client.HTTPException as exc:  # e.g. IncompleteRead mid-response
                raise TransportError(self._redact(
                    f"bad or truncated response from {self.url}: {type(exc).__name__}")) from None
            except (UnicodeDecodeError, json.JSONDecodeError):
                raise TransportError(f"response from {self.url} is not JSON") from None
            except ValueError:  # e.g. http.client's "Invalid header value", which quotes the key
                raise TransportError(
                    f"could not send the request to {self.url}: invalid header or URL "
                    "(details withheld, they may contain the key)") from None
        raise AssertionError("unreachable")

    def _redact(self, text: str) -> str:
        return text.replace(self._key, "[redacted]")


def _count(value: Any, bound: int) -> int:
    """A reported token count if it is a whole number in ``0..bound``, else ``bound``.

    ``bound`` is the pre-call upper bound, so a count above it is implausible.
    """
    valid = isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= bound
    return value if valid else bound


_COST = re.compile(r"\d+(?:\.\d*)?(?:[eE][+-]?\d+)?")
MAX_REPORTED_COST = Decimal(10) ** 9  # USD per call; anything larger is not a real cost


def _reported_cost(value: Any) -> Decimal | None:
    """A reported USD cost if it is a plausible finite number >= 0, else ``None``.

    Rejects bools, negatives, NaN, infinity (JSON's ``1e400`` arrives as
    infinity) and exponents too large to add up, so a bad value can never
    widen the budget or break the cap arithmetic.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    text = str(value).strip()
    if not _COST.fullmatch(text):
        return None
    try:
        cost = Decimal(text)
    except InvalidOperation:  # an exponent beyond what Decimal can hold: too large
        cost = MAX_REPORTED_COST + 1
    return cost if cost <= MAX_REPORTED_COST else None


def _http_message(exc: urllib.error.HTTPError, url: str) -> str:
    detail = ""
    try:
        detail = json.loads(exc.read().decode("utf-8"))["error"]["message"]
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        detail = exc.reason or ""
    rate = " (rate limited after retries)" if exc.code == 429 else ""
    return f"{url} returned HTTP {exc.code}{rate}: {str(detail)[:300]}"


def _backoff(retry_after: str | None, attempt: int) -> float:
    """Seconds to wait before retry ``attempt``: ``Retry-After`` if numeric, else 1, 2, 4."""
    try:
        wait = float(retry_after) if retry_after is not None else 2.0 ** attempt
    except ValueError:
        wait = 2.0 ** attempt
    return min(max(wait, 0.0), MAX_BACKOFF_S) if math.isfinite(wait) else MAX_BACKOFF_S


# --- clients ------------------------------------------------------------------


def router(env: Mapping[str, str] | None = None, *, opener: Opener | None = None,
           sleep: Callable[[float], None] = time.sleep) -> Any:
    """The model router for the configured provider: ``MockRouter`` or ``OpenRouterRouter``.

    The openrouter router passes the spend gate first and gets a fresh
    ``OpenRouterTransport`` (one per run).
    """
    from ledgercheck.agents import routing  # routing imports this module

    provider = load_settings(env).provider
    if provider == "mock":
        return routing.MockRouter(env)
    if provider == "openrouter":
        transport = OpenRouterTransport(require_live(env), opener=opener, sleep=sleep)
        return routing.OpenRouterRouter(env, transport=transport)
    raise ConnectionConfigError(f"unknown provider {provider!r}: use one of {list(PROVIDERS)}")


def tracer(env: Mapping[str, str] | None = None) -> Tracer:
    """``LangfuseTracer`` when both Langfuse keys are non-blank, else ``NullTracer``.

    One client is built per distinct (public key, secret key, host).
    """
    src = _Sources(env, with_file=False)  # Langfuse keys: environment and .env only
    public, secret = src.value(PUBLIC_KEY_ENV), src.value(SECRET_KEY_ENV)
    if public is None or secret is None:
        return NullTracer()
    return _langfuse_tracer(public, secret, src.value(HOST_ENV))


@lru_cache(maxsize=4)
def _langfuse_tracer(public: str, secret: str, host: str | None) -> LangfuseTracer:
    try:
        from langfuse import Langfuse  # pyright: ignore[reportMissingImports]
    except ImportError as exc:
        raise LangfuseUnavailable(
            f"{PUBLIC_KEY_ENV} and {SECRET_KEY_ENV} are set but the langfuse SDK is "
            "not installed: pip install 'ledgercheck[langfuse]', or unset the keys"
        ) from exc
    options = {"public_key": public, "secret_key": secret}
    if host:
        options["host"] = host
    return LangfuseTracer(Langfuse(**options))


def run_store(root: str | os.PathLike[str] | None = None) -> RunStore:
    """The run store: JSON files under ``root`` (default ``.scratch/runs``)."""
    return RunStore(DEFAULT_ROOT if root is None else root)


# --- ledgercheck connections ------------------------------------------------------


def describe(env: Mapping[str, str] | None = None) -> list[tuple[str, str, str]]:
    """``(setting, value, source)`` rows for ``ledgercheck connections``; never the key."""
    s, src = load_settings(env), _Sources(env, with_file=False)
    key = ("missing" if s.api_key is None
           else f"set (from {s.sources['api_key']})")
    if s.api_key is None and s.api_key_blank:
        key = "missing (blank)"
    elif s.api_key is not None and not _key_ok(s.api_key):
        key += ", invalid (whitespace or control characters)"
    langfuse = all(src.value(n) for n in (PUBLIC_KEY_ENV, SECRET_KEY_ENV))
    config = ("" if s.config_file is None
              else f"{s.config_file} ({'found' if s.config_file.is_file() else 'not found'})")
    return [
        ("provider", s.provider, s.sources["provider"]),
        ("live calls", f"on ({ENV_FLAG}=1)" if s.live_flag else f"off ({ENV_FLAG} is not 1)",
         "env"),
        ("api key", key, ""),
        *((f"model {t}", s.models.get(t, "missing"), s.sources[f"model_{t}"])
          for t in ("small", "large")),
        ("base url", s.base_url, s.sources["base_url"]),
        ("spend cap", "missing" if s.spend_cap_usd is None else f"${s.spend_cap_usd}",
         s.sources["spend_cap_usd"]),
        ("max tokens", str(s.max_tokens), s.sources["max_tokens"]),
        ("prices", ", ".join(s.prices) or "none", s.sources.get("prices", "")),
        ("tracing", "langfuse" if langfuse else "off", ""),
        ("config file", config, ""),
    ]


def build_parser(parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    """Add the connections options to ``parser``; --help shows this module's docstring."""
    if parser is None:
        parser = argparse.ArgumentParser(prog="python -m ledgercheck.connections")
    parser.description = __doc__
    parser.formatter_class = argparse.RawDescriptionHelpFormatter
    return parser


def run(args: argparse.Namespace) -> int:
    """Print the resolved settings; exit 2 if any setting is invalid, else 0."""
    for name, value, source in describe():
        print(f"{name:<12} {value:<44} {source}".rstrip())
    problems = load_settings().problems
    for problem in problems:
        print(f"connections: {problem}", file=sys.stderr)
    return 2 if problems else 0


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
