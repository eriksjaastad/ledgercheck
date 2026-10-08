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
``.env.example`` and ``connections.example.toml``. No configured secret (the
OpenRouter key, the Langfuse keys, a password in a URL setting), nor any 8+
character piece of one, is printed, logged, put in an error message, a repr, a
report, a run record or a trace, even if it was pasted into another setting.

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
"""

from __future__ import annotations

import argparse
import errno
import http.client
import json
import math
import os
import re
import stat
import sys
import time
import tomllib
import traceback
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field, fields
from decimal import Decimal, InvalidOperation
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

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
MAX_LOCAL_FILE_BYTES = 1 << 20  # .env and connections.local.toml
MAX_RESPONSE_BYTES = 10 << 20
MAX_ERROR_CHARS = 300
MAX_SPEND_CAP_USD = Decimal(1_000_000)
MAX_PRICE_PER_MTOK = Decimal(1_000_000)
MAX_MAX_TOKENS = 1_000_000
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


_MIN_KEY_RUN = 8  # a run of this many characters of a secret is masked
_MIN_SECRET = 4  # shorter secrets cannot be masked without garbling ordinary text


def _sanitize(text: str) -> str:
    return "".join(c if c.isprintable() else " " for c in text)


def _forms(secrets: Iterable[Any]) -> list[str]:
    """Each secret as it can appear in output: raw, repr()- and JSON-escaped, all with
    control characters made spaces (as the output will be), longest first."""
    forms = set()
    for secret in secrets:
        if isinstance(secret, str) and len(secret.strip()) >= _MIN_SECRET:
            secret = secret.strip()
            forms |= {_sanitize(f) for f in (secret, repr(secret)[1:-1], json.dumps(secret)[1:-1])}
    return sorted((f for f in forms if len(f) >= _MIN_SECRET), key=len, reverse=True)


def _run_at(text: str, i: int, forms: list[str]) -> int:
    """Length of the secret piece starting at ``text[i]``: a whole short secret, or the
    longest 8+ character run of a long one; 0 if none."""
    best = 0
    for form in forms:
        if len(form) < _MIN_KEY_RUN:
            if text.startswith(form, i):
                best = max(best, len(form))
            continue
        n = 0
        while n < len(form) and i + n < len(text) and text[i:i + n + 1] in form:
            n += 1
        if n >= _MIN_KEY_RUN:
            best = max(best, n)
    return best


def _mask(text: Any, secrets: Iterable[Any], limit: int = MAX_ERROR_CHARS) -> str:
    """THE masking choke point: every string shown to a person or raised in an error
    passes through here. Control characters become spaces; every secret, and every run of
    8+ consecutive characters of one, becomes ``[redacted]``; only then is the result cut
    to ``limit`` characters, so a cut can never leave a piece of a secret behind."""
    forms = _forms(secrets)
    text = str(text)
    longest = max(map(len, forms), default=0)
    cut = limit < sys.maxsize and len(text) > limit + longest
    if cut:  # nothing past here can reach the first `limit` characters of the output
        text = text[:limit + longest]
    text = _sanitize(text)
    out, i = [], 0
    while i < len(text):
        n = _run_at(text, i, forms) if forms else 0
        out.append("[redacted]" if n else text[i])
        i += n or 1
    text = "".join(out)
    return text[:limit] + "..." if cut or len(text) > limit else text


def _scrub(text: Any, key: str | None, limit: int = MAX_ERROR_CHARS) -> str:
    """``_mask`` with a single secret."""
    return _mask(text, (key,), limit)


def _holds_secret(text: str, secret: str | None) -> bool:
    """``text`` contains ``secret`` or an 8+ character piece of it."""
    return bool(secret) and _mask(text, (secret,), sys.maxsize) != _sanitize(text)


def _url_userinfo(url: Any) -> list[str]:
    """The user and password in a URL (raw and percent-decoded), if it has any."""
    if not isinstance(url, str):
        return []
    try:
        parts = urllib.parse.urlsplit(url.strip())
        found = [parts.username, parts.password]
    except ValueError:  # e.g. a broken IPv6 host: no userinfo to find
        found = []
    return [v for x in found if x for v in (x, urllib.parse.unquote(x))]


def _shown(value: Any) -> str:
    """A setting's value for an error message, kept short."""
    try:
        text = repr(value)
    except (ValueError, RecursionError):  # an int over the digit limit, absurd nesting
        text = f"a {type(value).__name__}"
    return text if len(text) <= 40 else text[:40] + "..."


def _read_limited(path: Path) -> bytes:
    """A regular file's bytes, at most ``MAX_LOCAL_FILE_BYTES``; ``OSError`` or
    ``ValueError`` (a NUL byte in the path) otherwise, never a hang on a FIFO."""
    mode = path.stat().st_mode
    if stat.S_ISDIR(mode):
        raise IsADirectoryError(errno.EISDIR, "is a directory")
    if not stat.S_ISREG(mode):
        raise OSError(errno.EINVAL, "not a regular file")
    with path.open("rb") as fh:
        data = fh.read(MAX_LOCAL_FILE_BYTES + 1)
    if len(data) > MAX_LOCAL_FILE_BYTES:
        raise OSError(errno.EFBIG, f"larger than {MAX_LOCAL_FILE_BYTES} bytes")
    return data


def _read_dotenv(path: Path) -> dict[str, str]:
    """``KEY=VALUE`` lines with optional ``export``, quotes and `` # comment``.

    Tolerant on purpose: ``.env`` is often shared with other tools, so a line
    this parser does not understand is skipped, and a missing, unreadable or
    odd ``.env`` (a directory, a FIFO, too large) is ignored, never an error.
    """
    try:
        data = _read_limited(path)
    except (OSError, ValueError):
        data = b""
    values = {}
    for line in data.decode("utf-8", errors="replace").splitlines():
        m = _DOTENV_LINE.fullmatch(line)
        if m:
            name, double, single, bare = m.groups()
            values[name] = next(v for v in (double, single, bare) if v is not None)
    return values


def _read_config(path: Path, required: bool) -> dict[str, Any]:
    """The connections file as a dict (``{}`` when absent and not ``required``).

    Every read or parse failure is a ``ConnectionConfigError`` naming the path
    and the problem, never the file's contents.
    """
    raw, problem = None, None
    try:
        raw = _read_limited(path)
    except (FileNotFoundError, NotADirectoryError):
        if required:
            problem = f"{CONNECTIONS_FILE_ENV} names {path}, which does not exist"
    except PermissionError:
        problem = f"cannot read {path}: permission denied"
    except ValueError:  # a NUL byte in the path
        problem = f"{CONNECTIONS_FILE_ENV} is not a usable file path"
    except OSError as exc:
        problem = f"cannot read {path}: {exc.strerror or type(exc).__name__}"
    if problem is not None:
        raise ConnectionConfigError(problem)
    if raw is None:
        return {}
    try:
        data = tomllib.loads(raw.decode("utf-8"))
    except UnicodeDecodeError:
        raise ConnectionConfigError(f"cannot read {path}: not UTF-8") from None
    except tomllib.TOMLDecodeError as exc:
        raise ConnectionConfigError(f"{path}: {exc}") from None
    except RecursionError:
        raise ConnectionConfigError(f"{path}: nested too deeply") from None
    except (ValueError, OverflowError):  # e.g. an integer over Python's 4300-digit limit
        raise ConnectionConfigError(f"{path}: not valid TOML (a value is too large)") from None
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


SECRET_VARS = (API_KEY_ENV, SECRET_KEY_ENV, PUBLIC_KEY_ENV)
URL_VARS = (SETTINGS["base_url"][0], HOST_ENV)  # a user:password@ in these is a secret too


def _secrets(src: _Sources) -> tuple[str, ...]:
    """Every secret the configuration knows, from every source (not only the one that
    wins): the OpenRouter key, both Langfuse keys, and any user or password in a URL
    setting. Masked everywhere, even when a value failed validation."""
    found: list[Any] = []
    for values in (src.env, src.dotenv):
        found += [values.get(name) for name in SECRET_VARS]
        for name in URL_VARS:
            found += _url_userinfo(values.get(name))
    found.append(src.config.get("api_key"))
    found += _url_userinfo(src.config.get("base_url"))
    return tuple(dict.fromkeys(v.strip() for v in found if isinstance(v, str) and v.strip()))


def _decimal(value: Any, what: str, maximum: Decimal) -> Decimal:
    number = None
    if isinstance(value, (int, float, str)) and not isinstance(value, bool):
        try:
            number = Decimal(str(value).strip())
        except (InvalidOperation, ValueError):  # str() of an int past the digit limit
            number = None
    if number is None:
        raise ConnectionConfigError(f"{what} must be a number, got {_shown(value)}")
    if not number.is_finite() or number < 0 or number > maximum:
        raise ConnectionConfigError(
            f"{what} must be a finite number from 0 to {maximum}, got {_shown(value)}")
    return number


_WHOLE = re.compile(r"[0-9]{1,7}")  # ASCII digits only: "²".isdigit() is true


def _max_tokens(raw: Any) -> int:
    if raw is None:
        return DEFAULT_MAX_TOKENS
    text = ""
    if isinstance(raw, (int, str)) and not isinstance(raw, bool):
        try:
            text = str(raw).strip()
        except ValueError:  # an int past the digit limit
            text = ""
    if not _WHOLE.fullmatch(text) or not 0 < int(text) <= MAX_MAX_TOKENS:
        raise ConnectionConfigError(
            f"max_tokens must be a whole number from 1 to {MAX_MAX_TOKENS}, got {_shown(raw)}")
    return int(text)


def _base_url(raw: Any) -> str:
    """An https URL with a host and no ``user:password@``, query or fragment.

    The key travels in a header, and the URL appears in error messages, so it
    must not carry credentials of its own. Bad values are not echoed.
    """
    if raw is None:
        return DEFAULT_BASE_URL
    url = raw.strip().rstrip("/") if isinstance(raw, str) else ""
    try:
        parts = urllib.parse.urlsplit(url)
        valid = (parts.scheme == "https" and bool(parts.hostname) and parts.username is None
                 and parts.password is None and not parts.query and not parts.fragment)
    except ValueError:
        valid = False
    if not valid:
        raise ConnectionConfigError("base_url must be an https:// URL with a host and no "
                                    "user:password@, query or fragment")
    return url


def _cap(raw: Any) -> Decimal | None:
    return None if raw is None else _decimal(raw, "spend_cap_usd", MAX_SPEND_CAP_USD)


def _provider(raw: Any) -> str:
    if raw is None:
        return "mock"
    if not isinstance(raw, str):
        raise ConnectionConfigError(f"provider must be a string, got a {type(raw).__name__}")
    return raw.strip().lower() or "mock"


def _live_gaps(provider: str, key: str | None, cap: Decimal | None, cap_raw: Any,
               models: Mapping[str, str], prices: Mapping[str, Any]) -> list[str]:
    """What a live run with ``provider`` still needs; checked up front, not only at use."""
    if provider != "openrouter":
        return []
    gaps = []
    if key is None:
        gaps.append(f"an API key: {_WHERE_KEY}")
    if cap is None and cap_raw is None:
        gaps.append("a spend cap: set LEDGERCHECK_SPEND_CAP_USD or spend_cap_usd")
    if "large" not in models:
        gaps.append("a large model id (the judge uses it): set LEDGERCHECK_MODEL_LARGE or "
                    "[models] large")
    gaps += [f'a price for model {tier} ({model}): add [prices."{model}"] prompt and completion '
             "(USD per million tokens)" for tier, model in models.items() if model not in prices]
    return gaps


def _model_id(tier: str) -> Callable[[Any], str | None]:
    def parse(raw: Any) -> str | None:
        if raw is not None and not isinstance(raw, str):
            raise ConnectionConfigError(f"model {tier} must be a string like "
                                        f"'provider/model-id', got a {type(raw).__name__}")
        if raw is None:
            return None
        return raw.strip() or None
    return parse


def _parsed(problems: list[str], parse: Callable[[Any], Any], raw: Any, fallback: Any) -> Any:
    """``parse(raw)``, or ``fallback`` with the error added to ``problems``."""
    try:
        return parse(raw)
    except ConnectionConfigError as exc:
        problems.append(str(exc))
    return fallback


def _key_ok(key: str) -> bool:
    """Printable ASCII with no whitespace: anything else breaks or corrupts the header."""
    return key.isascii() and not any(c.isspace() or not c.isprintable() for c in key)


def _prices(raw: Any) -> dict[str, tuple[Decimal, Decimal]]:
    if not isinstance(raw, Mapping):
        raise ConnectionConfigError("[prices] must map model ids to {prompt, completion}")
    prices = {}
    for model, entry in raw.items():
        if not isinstance(entry, Mapping) or set(entry) != {"prompt", "completion"}:
            raise ConnectionConfigError(f'[prices."{model}"] needs exactly prompt and completion '
                                        "(USD per million tokens)")
        prices[model] = (_decimal(entry["prompt"], f"{model} prompt price", MAX_PRICE_PER_MTOK),
                         _decimal(entry["completion"], f"{model} completion price",
                                  MAX_PRICE_PER_MTOK))
    return prices


@dataclass(frozen=True, repr=False)
class Settings:
    """Resolved connection settings; ``sources`` says where each came from.

    Its repr never shows the key, nor any value that holds a piece of it.
    """

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
    live_gaps: tuple[str, ...] = ()  # what the configured live provider still needs
    # Every secret in the configuration (see ``_secrets``): what all output is masked with.
    secrets: tuple[str, ...] = field(default=(), repr=False, compare=False)

    def mask(self, text: Any, limit: int = sys.maxsize) -> str:
        """``text`` with every secret of this configuration masked (see ``_mask``)."""
        return _mask(text, (*self.secrets, self.api_key), limit)

    def __repr__(self) -> str:
        shown = ", ".join(f"{f.name}={getattr(self, f.name)!r}" for f in fields(self)
                          if f.name not in ("api_key", "secrets"))
        return self.mask(f"Settings({shown})")


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
    models_table = src.config.get("models", {})
    if not isinstance(models_table, Mapping):
        problems.append("[models] must be a table with small and/or large")
    elif unknown_tiers := sorted(set(models_table) - {"small", "large"}):
        problems.append(f"[models] has unknown keys {unknown_tiers}; allowed: small, large")
    key = raw["api_key"]
    if key is not None and not isinstance(key, str):
        problems.append(f"api_key in {sources['api_key']} must be a string")
        key = None
    if key is not None and not _key_ok(key):
        problems.append(f"{API_KEY_ENV} (from {sources['api_key']}) contains whitespace, control "
                        "or non-ASCII characters; check for a stray space, quote or line break")
    models = {tier: _parsed(problems, _model_id(tier), raw[f"model_{tier}"], None)
              for tier in ("small", "large")}
    provider = _parsed(problems, _provider, raw["provider"], "mock")
    base_url = _parsed(problems, _base_url, raw["base_url"], DEFAULT_BASE_URL)
    cap = _parsed(problems, _cap, raw["spend_cap_usd"], None)
    max_tokens = _parsed(problems, _max_tokens, raw["max_tokens"], DEFAULT_MAX_TOKENS)
    prices = _parsed(problems, _prices, src.config.get("prices", {}), {})
    if provider not in PROVIDERS:
        problems.append(f"unknown provider {_shown(provider)}: use one of {list(PROVIDERS)}")
    # A secret pasted into another setting would be sent as a model id or URL: refuse it.
    secrets = _secrets(src)
    strong = [secret for secret in secrets if len(secret) >= _MIN_KEY_RUN]
    pasted = [("provider", raw["provider"]), ("model small", raw["model_small"]),
              ("model large", raw["model_large"]), ("base_url", raw["base_url"]),
              (CONNECTIONS_FILE_ENV, str(src.config_path or ""))]
    pasted += [("a [prices] model id", model) for model in prices]
    problems += [f"{name} contains a secret (your API key, a Langfuse key or a URL password, or "
                 "part of one); it was probably pasted into the wrong setting"
                 for name, value in pasted
                 if isinstance(value, str) and any(_holds_secret(value, x) for x in strong)]
    gaps = _live_gaps(provider, key, cap, raw["spend_cap_usd"],
                      {tier: model for tier, model in models.items() if model}, prices)
    return Settings(
        provider=provider,
        models={tier: model for tier, model in models.items() if model},
        base_url=base_url,
        spend_cap_usd=cap,
        max_tokens=max_tokens,
        prices=prices,
        live_flag=src.env.get(ENV_FLAG) == "1",
        sources=sources,
        config_file=src.config_path,
        api_key=key,
        api_key_blank=src.blank(API_KEY_ENV),
        # Values are echoed in problems, so a key pasted into the wrong setting is masked.
        problems=tuple(_mask(problem, (*secrets, key), 500) for problem in problems),
        live_gaps=tuple(_mask(gap, (*secrets, key), 500) for gap in gaps),
        secrets=secrets,
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


class _Redirected(Exception):
    """A 3xx answer, refused; the transport turns it into a ``TransportError``."""

    def __init__(self, code: int) -> None:
        super().__init__(code)
        self.code = code


class _PossiblyBilled(Exception):
    """The request reached the provider but no usable reply came back."""

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


class _RefuseRedirects(urllib.request.HTTPRedirectHandler):
    """Refuse every redirect: following one would re-send the Authorization header to
    whatever the Location names (another host, or plain http)."""

    def redirect_request(self, req: urllib.request.Request, fp: Any, code: int, msg: str,
                         headers: Any, newurl: str) -> None:
        fp.close()
        raise _Redirected(code)


def _default_opener() -> Opener:
    """``urlopen`` without redirects (see ``_RefuseRedirects``)."""
    return urllib.request.build_opener(_RefuseRedirects).open


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

    A call that may have been billed but gave no usable reply (a 2xx body that
    is cut off, not UTF-8, not JSON or too large, or a timeout or dropped
    connection after the request was sent) is charged its full pre-call worst
    case before the ``TransportError``. A request that never reached the
    provider (connection refused, DNS, TLS) and non-2xx answers are not charged.

    Redirects are never followed: a 3xx answer is a ``TransportError`` naming
    the status, so the key is only ever sent to ``base_url``. Every error
    message built from outside data goes through one scrubber that masks the
    key (and any 8+ character piece of it) before anything is truncated.

    Build one per run (``totals`` is that run's spend). ``opener`` and
    ``sleep`` default to a urllib opener that refuses redirects and
    ``time.sleep``; tests inject fakes.
    """

    name = "openrouter"

    def __init__(self, settings: Settings, *, opener: Opener | None = None,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        if not settings.api_key:
            raise LiveLLMDisabled(f"{API_KEY_ENV} is not set: {_WHERE_KEY}")
        if not _key_ok(settings.api_key):
            raise ConnectionConfigError(
                f"{API_KEY_ENV} contains whitespace, control or non-ASCII characters")
        if settings.spend_cap_usd is None or settings.spend_cap_usd <= 0:
            raise ConnectionConfigError(
                "a live run needs a spend cap: set LEDGERCHECK_SPEND_CAP_USD or spend_cap_usd "
                "in connections.local.toml (USD, > 0)")
        self._key = settings.api_key
        self._secrets = (*settings.secrets, settings.api_key)  # what every message is masked with
        self.url = f"{settings.base_url}/chat/completions"
        self._max_tokens, self._prices = settings.max_tokens, settings.prices
        self._open = _default_opener() if opener is None else opener
        self._sleep = sleep
        self.totals = Totals(settings.spend_cap_usd)

    def __repr__(self) -> str:
        return _mask(f"OpenRouterTransport(url={self.url!r}, totals={self.totals!r})",
                     self._secrets, limit=sys.maxsize)

    def __call__(self, model: str, prompt: str) -> str:
        if model not in self._prices:
            raise ConnectionConfigError(_mask(
                f'no price for {model!r}: add [prices."{model}"] prompt = ..., completion = ... '
                "(USD per million tokens) to connections.local.toml", self._secrets, 500))
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
        try:
            data = self._post({"model": model, "messages": messages,
                               "max_tokens": self._max_tokens})
        except _PossiblyBilled as exc:
            self._charge(bound, self._max_tokens, worst)  # unknown usage: the worst case
            raise self._error(exc.detail) from None
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
        self._charge(p_tok, c_tok, cost)
        try:
            reply = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise self._error(f"unexpected response shape from {self.url}") from None
        if not isinstance(reply, str):
            raise self._error(f"no text reply from {self.url}")
        return reply

    def _charge(self, prompt_tokens: int, completion_tokens: int, cost: Decimal) -> None:
        totals = self.totals
        totals.calls += 1
        totals.prompt_tokens += prompt_tokens
        totals.completion_tokens += completion_tokens
        totals.cost_usd += cost

    def _error(self, text: str) -> TransportError:
        """The only way the transport builds an error: the text is always scrubbed."""
        return TransportError(_mask(text, self._secrets))

    def _post(self, payload: Mapping[str, Any]) -> Any:
        """The parsed JSON reply. ``_PossiblyBilled`` once the request may have been
        processed; ``TransportError`` when it surely was not."""
        body = json.dumps(payload).encode("utf-8")
        headers = {"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"}
        for attempt in range(MAX_RETRIES + 1):
            request = urllib.request.Request(self.url, data=body, headers=headers, method="POST")
            try:
                response = self._open(request, timeout=TIMEOUT_S)
            except _Redirected as exc:
                raise self._error(f"{self.url} answered HTTP {exc.code} (a redirect); redirects "
                                  "are refused so the key is never sent anywhere else") from None
            except urllib.error.HTTPError as exc:  # a non-2xx answer: not billed
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
                if exc.code != 429 or attempt == MAX_RETRIES:
                    raise self._error(_http_message(exc, self.url)) from None
                exc.close()
                self.totals.retries += 1
                self._sleep(_backoff(retry_after, attempt))
                continue
            except urllib.error.URLError as exc:  # connecting or sending failed: not billed
                raise self._error(f"cannot reach {self.url}: {exc.reason}") from None
            except (OSError, http.client.HTTPException) as exc:
                # urllib wraps connect/send errors in URLError, so these came after the
                # request was sent (a read timeout, a dropped connection): possibly billed.
                raise _PossiblyBilled(
                    f"no usable response from {self.url}: {type(exc).__name__}") from None
            except ValueError:  # e.g. http.client's "Invalid header value", which quotes the key
                raise self._error(f"could not send the request to {self.url}: invalid header "
                                  "or URL (details withheld, they may contain the key)") from None
            return self._read_reply(response)
        raise AssertionError("unreachable")

    def _read_reply(self, response: Any) -> Any:
        """Parse a 2xx reply; from here on any failure may have been billed."""
        try:
            with response:
                raw = response.read(MAX_RESPONSE_BYTES + 1)
            if len(raw) > MAX_RESPONSE_BYTES:
                raise _PossiblyBilled(f"response from {self.url} is larger than "
                                      f"{MAX_RESPONSE_BYTES} bytes")
            return json.loads(raw.decode("utf-8"))
        except (OSError, http.client.HTTPException) as exc:  # e.g. IncompleteRead, timeout
            raise _PossiblyBilled(
                f"bad or truncated response from {self.url}: {type(exc).__name__}") from None
        except (ValueError, RecursionError):  # not UTF-8, not JSON, absurd numbers or nesting
            raise _PossiblyBilled(f"response from {self.url} is not JSON") from None


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
    """The provider's error, unredacted and uncut: callers pass it to ``_scrub``."""
    detail: Any = ""
    try:
        detail = json.loads(exc.read(64 << 10).decode("utf-8"))["error"]["message"]
    except (OSError, ValueError, KeyError, TypeError, AttributeError, RecursionError,
            http.client.HTTPException):
        detail = exc.reason or ""
    rate = " (rate limited after retries)" if exc.code == 429 else ""
    return f"{url} returned HTTP {exc.code}{rate}: {detail}"


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

    settings = load_settings(env)
    if settings.provider == "mock":
        return routing.MockRouter(env)
    if settings.provider == "openrouter":
        transport = OpenRouterTransport(require_live(env), opener=opener, sleep=sleep)
        return routing.OpenRouterRouter(env, transport=transport)
    raise ConnectionConfigError(settings.mask(f"unknown provider {_shown(settings.provider)}: "
                                              f"use one of {list(PROVIDERS)}", 500))


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
    try:
        client = Langfuse(**options)
    except Exception as exc:  # an SDK error can quote its options: re-raise it masked
        raise LangfuseUnavailable(_mask(
            f"cannot start the Langfuse client: {type(exc).__name__}: {exc}",
            (public, secret, *_url_userinfo(host)))) from None
    return LangfuseTracer(client)


def run_store(root: str | os.PathLike[str] | None = None) -> RunStore:
    """The run store: JSON files under ``root`` (default ``.scratch/runs``)."""
    return RunStore(DEFAULT_ROOT if root is None else root)


# --- ledgercheck connections ------------------------------------------------------


def _file_state(path: Path) -> str:
    try:
        state = "found" if path.is_file() else "not found"
    except (OSError, ValueError):  # an unreadable directory, a NUL byte in the path
        state = "unreadable"
    return state


def masker(env: Mapping[str, str] | None = None,
           limit: int = sys.maxsize) -> Callable[[Any], str]:
    """A function that makes text safe to print: control characters removed, and every
    secret of the configuration (see ``_secrets``) masked. Resolve it once per command
    and pass everything that command prints through it."""
    secrets = _secrets(_Sources(env))
    return lambda text: _mask(text, secrets, limit)


def mask(text: Any, env: Mapping[str, str] | None = None, limit: int = 2000) -> str:
    """``masker(env, limit)(text)``, for a one-off message."""
    return masker(env, limit)(text)


def mask_lines(text: Any, env: Mapping[str, str] | None = None) -> str:
    """Multi-line text (a report, a traceback) masked line by line, newlines kept."""
    show = masker(env)
    return "\n".join(show(line) for line in str(text).splitlines())


def install_masked_excepthook(env: Mapping[str, str] | None = None) -> None:
    """Make an unexpected crash print a masked traceback (still exit 1)."""

    def hook(kind: type[BaseException], exc: BaseException, tb: Any) -> None:
        text = "".join(traceback.format_exception(kind, exc, tb))
        try:
            shown = mask_lines(text, env)
        except Exception:  # never fall back to the unmasked text
            shown = f"ledgercheck: unexpected {kind.__name__} (details withheld)"
        print(shown, file=sys.stderr)

    sys.excepthook = hook


def describe(env: Mapping[str, str] | None = None) -> list[tuple[str, str, str]]:
    """``(setting, value, source)`` rows for ``ledgercheck connections``.

    Every cell is masked: a key pasted into any setting never shows.
    """
    s, src = load_settings(env), _Sources(env, with_file=False)
    key = ("missing" if s.api_key is None
           else f"set (from {s.sources['api_key']})")
    if s.api_key is None and s.api_key_blank:
        key = "missing (blank)"
    elif s.api_key is not None and not _key_ok(s.api_key):
        key += ", invalid (whitespace, control or non-ASCII characters)"
    langfuse = all(src.value(n) for n in (PUBLIC_KEY_ENV, SECRET_KEY_ENV))
    config = "" if s.config_file is None else f"{s.config_file} ({_file_state(s.config_file)})"
    rows = [
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
    return [(name, s.mask(value, 200), s.mask(source, 200)) for name, value, source in rows]


def build_parser(parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    """Add the connections options to ``parser``; --help shows this module's docstring."""
    if parser is None:
        parser = argparse.ArgumentParser(prog="python -m ledgercheck.connections")
    parser.description = __doc__
    parser.formatter_class = argparse.RawDescriptionHelpFormatter
    return parser


def run(args: argparse.Namespace) -> int:
    """Print the resolved settings; exit 2 if a setting is invalid or the configured live
    provider is missing something it needs, else 0."""
    settings = load_settings()
    for name, value, source in describe():  # already masked
        print(settings.mask(f"{name:<12} {value:<44} {source}".rstrip()))
    for problem in settings.problems:
        print(settings.mask(f"connections: {problem}"), file=sys.stderr)
    for gap in settings.live_gaps:
        print(settings.mask(f"connections: provider {settings.provider} needs {gap}"),
              file=sys.stderr)
    return 2 if settings.problems or settings.live_gaps else 0


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    install_masked_excepthook()
    sys.exit(main())
