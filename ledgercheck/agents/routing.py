"""Model routing: one interface picks a small or large model per task.

Each LLM task kind maps to a model tier. Small, fast extraction goes to the small
model, and reasoning goes to the large one:

    ``extraction``          small   intake: invoice fields from raw text
    ``rerank``              large   policy: reorder unknown-vendor chunks
    ``approval_reasoning``  large   approval: explain or argue an outcome
    ``judge``               large   eval: score output against the rubric

Model ids are configuration, not code: set ``LEDGERCHECK_MODEL_SMALL`` and
``LEDGERCHECK_MODEL_LARGE`` (or ``[models]`` in ``connections.local.toml``,
see ``ledgercheck.connections``) to ``vendor/model`` ids your provider
serves. Surrounding whitespace is stripped, and an empty or whitespace-only
value counts as unset. The mock router fills an unset tier with a fake id:

    small  ``mock/small``
    large  ``mock/large``

Routers
-------
``connections.router()`` builds the router for the configured provider:
``MockRouter`` for ``mock`` (the default) and ``OpenRouterRouter`` for
``openrouter``. It is the only place that constructs one. No pipeline stage
calls a router yet; the only production caller is ``LiveJudge``
(``ledgercheck judge --live``).

``MockRouter`` is the offline router. It records each call and returns a canned
reply (``""`` unless one is given).

``OpenRouterRouter`` is the live router, behind the spend gate. Constructing
it, ``route`` and ``complete`` each run ``connections.require_live`` against the
same env (re-read every time when none was given), so it raises
``LiveLLMDisabled`` unless ``LEDGERCHECK_LLM`` is exactly ``"1"`` and
``OPENROUTER_API_KEY`` is non-blank, and clearing the flag or key after
construction closes it too. A routed tier with no configured model id raises
``ConnectionConfigError``. A transport is a ``(model, prompt) -> reply``
callable; without one the router builds ``connections.OpenRouterTransport``,
which needs a spend cap and per-model prices, and reports ``totals``.

Runbook: rate limits (HTTP 429)
-------------------------------
Today: ``OpenRouterTransport`` retries a 429 up to 3 times, waiting for
``Retry-After`` when the response sends a number of seconds and 1, 2, 4
seconds otherwise, each wait capped at 30 s. Then it raises
``TransportError``, and ``ledgercheck judge --live`` stops with exit 2 and
prints the run totals, retries included. Other HTTP errors are not retried.

Planned:

- When the retries run out, try ``Route.fallback`` once. A small-tier task
  escalates to the large model. Large-tier tasks have no fallback today.
- If that also fails, fail closed. The run stops at ``needs_human``
  (``Outcome.NEEDS_HUMAN``) with the error recorded. Never auto-approve.
- Record 429s, retries and the fallback in the trace and in the
  ``rate_limit_errors`` field of ``ledgercheck beta-notes``.

Runbook: context-window failures
--------------------------------
Today: the transport estimates prompt tokens only to bound the worst-case cost
of a call (see the spend cap in ``ledgercheck.connections``). The judge's
prompts are small, and nothing else is sent.

Planned:

- Estimate tokens before sending (characters / 4 is enough to start). Keep
  each model's context limit next to its id here.
- For an oversized invoice or PDF text, chunk it by page or line-item block.
  Extract each chunk and merge, rather than silently truncating. A truncated
  invoice can drop line items and still look valid.
- Context-length errors escalate through ``Route.fallback`` to the
  larger-context model.
- If it still does not fit, or the merged chunks disagree on header totals,
  fail closed to ``needs_human`` with the reason.
- Record it in the ``context_errors`` field of ``ledgercheck beta-notes``.

Not used: LiteLLM, the openai SDK and httpx. ``dependencies`` stays empty; the
transport uses ``urllib`` from the standard library.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Callable, Mapping, Protocol

from ledgercheck import connections
from ledgercheck.connections import ConnectionConfigError, SETTINGS, Totals


class TaskKind(StrEnum):
    EXTRACTION = "extraction"
    RERANK = "rerank"
    APPROVAL_REASONING = "approval_reasoning"
    JUDGE = "judge"


class Tier(StrEnum):
    SMALL = "small"
    LARGE = "large"


TASK_TIERS: Mapping[TaskKind, Tier] = {
    TaskKind.EXTRACTION: Tier.SMALL,
    TaskKind.RERANK: Tier.LARGE,
    TaskKind.APPROVAL_REASONING: Tier.LARGE,
    TaskKind.JUDGE: Tier.LARGE,
}
MOCK_MODELS: Mapping[Tier, str] = {Tier.SMALL: "mock/small", Tier.LARGE: "mock/large"}
MODEL_ENV: Mapping[Tier, str] = {tier: SETTINGS[f"model_{tier}"][0] for tier in Tier}

Transport = Callable[[str, str], str]


def resolve_models(env: Mapping[str, str] | None = None) -> dict[Tier, str]:
    """Mock model id per tier: the configured id, or the fake ``mock/<tier>`` when unset."""
    configured = connections.load_settings(env).models
    return {tier: configured.get(tier, default) for tier, default in MOCK_MODELS.items()}


@dataclass(frozen=True, slots=True)
class Route:
    """Where a task goes: its tier, model, and the model to escalate to (or ``None``)."""

    task: TaskKind
    tier: Tier
    model: str
    fallback: str | None


class ModelRouter(Protocol):
    name: str

    def route(self, task: TaskKind | str) -> Route: ...

    def complete(self, task: TaskKind, prompt: str) -> str: ...


def _route(task: TaskKind | str, models: Mapping[Any, str]) -> Route:
    tier = TASK_TIERS[TaskKind(task)]
    model = models.get(tier)
    if not model:
        raise ConnectionConfigError(
            f"no model id for the {tier} tier: set {MODEL_ENV[tier]} or [models] {tier} in "
            "connections.local.toml (see ledgercheck connections --help)")
    large = models.get(Tier.LARGE)
    fallback = large if tier is Tier.SMALL and large and large != model else None
    return Route(TaskKind(task), tier, model, fallback)


class MockRouter:
    """Offline router: records ``(task, model, prompt)`` in ``calls`` and returns a canned reply."""

    name = "mock"

    def __init__(
        self,
        env: Mapping[str, str] | None = None,
        replies: Mapping[TaskKind, str] | None = None,
    ) -> None:
        self.models = resolve_models(env)
        self.replies = dict(replies or {})
        self.calls: list[tuple[TaskKind, str, str]] = []

    def route(self, task: TaskKind | str) -> Route:
        return _route(task, self.models)

    def complete(self, task: TaskKind, prompt: str) -> str:
        route = self.route(task)
        self.calls.append((route.task, route.model, prompt))
        return self.replies.get(route.task, "")


class OpenRouterRouter:
    """Live router behind the spend gate, checked on construction and every call."""

    name = "openrouter"

    def __init__(
        self,
        env: Mapping[str, str] | None = None,
        transport: Transport | None = None,
    ) -> None:
        settings = connections.require_live(env)  # the gate: raises LiveLLMDisabled first
        self._env = env  # None keeps re-reading the environment and local files
        self._transport = (connections.OpenRouterTransport(settings)
                           if transport is None else transport)

    @property
    def totals(self) -> Totals | None:
        """The transport's spend so far, when it reports one."""
        return getattr(self._transport, "totals", None)

    def route(self, task: TaskKind | str) -> Route:
        # The gate again: the flag or key may be gone since __init__.
        return _route(task, connections.require_live(self._env).models)

    def complete(self, task: TaskKind, prompt: str) -> str:
        route = self.route(task)
        return self._transport(route.model, prompt)
