"""Model routing: one interface picks a small or large model per task. Spends nothing.

Each LLM task kind maps to a model tier. Small, fast extraction goes to the small
model, and reasoning goes to the large one:

    ``extraction``          small   intake: invoice fields from raw text
    ``rerank``              large   policy: reorder unknown-vendor chunks
    ``approval_reasoning``  large   approval: explain or argue an outcome
    ``judge``               large   eval: score output against the rubric

Default model ids (OpenRouter ``vendor/model`` form). They are placeholders:
nobody has checked that OpenRouter serves them or what they cost.

    small  ``anthropic/claude-haiku-4.5``
    large  ``anthropic/claude-sonnet-4.5``

``LEDGERCHECK_MODEL_SMALL`` and ``LEDGERCHECK_MODEL_LARGE`` override a tier's
model id. Surrounding whitespace is stripped, and an unset, empty or
whitespace-only value falls back to the default.

Routers
-------
``default_router()`` picks the router for a caller that does not choose one:
``MockRouter`` unless ``LEDGERCHECK_LLM`` is exactly ``"1"``, and
``OpenRouterRouter`` (which still needs the key) when it is. No pipeline stage
calls a router yet. The only production caller is ``LiveJudge``
(``ledgercheck judge --live``), and it always builds ``OpenRouterRouter``.

``MockRouter`` is the offline router. It records each call and returns a canned
reply (``""`` unless one is given). Tests use it, and also build
``OpenRouterRouter`` with the gate opened by a fake key. Those live routers get
an injected transport or raise ``NotImplementedError``, so no test sends a
request.

``OpenRouterRouter`` is the live router, behind the existing spend gate.
Constructing it constructs ``llm_client.LLMClient``, so it raises
``LiveLLMDisabled`` unless ``LEDGERCHECK_LLM`` is exactly ``"1"`` and
``OPENROUTER_API_KEY`` is non-blank. There is no second flag. ``route`` and
``complete`` run the same gate again on every call, against the same env
(``os.environ`` read afresh when none was given), so clearing the flag or key
after construction closes it too. Even past the gate, ``complete`` makes no request. With no injected ``transport`` it raises
``NotImplementedError``. A transport is a ``(model, prompt) -> reply`` callable;
tests use it to see which model was picked. The real HTTP transport is planned.

Live mode (your own key; live calls cost money): set ``LEDGERCHECK_LLM=1``
and ``OPENROUTER_API_KEY`` (from your secret store), and optionally the two
model env vars.
Today that only lets the live objects construct. Every live call still raises
``NotImplementedError``, so nothing spends by default or with the flag on.

Runbook: rate limits (HTTP 429)
-------------------------------
Today: no live request is sent, so no 429 can happen. A transport's exceptions
propagate unchanged. There is no retry, backoff or fallback.

Planned:

- On 429 or 503, wait for ``Retry-After`` if the response sends one. Otherwise
  use exponential backoff with jitter (1s, 2s, 4s, capped).
- A retry budget per document run, a few attempts in total, so a stuck
  provider cannot stall a batch.
- When the budget runs out, try ``Route.fallback`` once. A small-tier task
  escalates to the large model. Large-tier tasks have no fallback today.
- If that also fails, fail closed. The run stops at ``needs_human``
  (``Outcome.NEEDS_HUMAN``) with the error recorded. Never auto-approve.
- Record 429s, retries and the fallback in the trace and in the
  ``rate_limit_errors`` field of ``ledgercheck beta-notes``.

Runbook: context-window failures
--------------------------------
Today: nothing measures prompt size. Inputs are the fixture texts, all small,
and no live call sends them anywhere.

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

Not used: LiteLLM, the openai SDK and httpx. ``dependencies`` stays empty, and
the planned transport is meant to use ``urllib`` from the standard library.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import StrEnum
from typing import Callable, Mapping, Protocol

from ledgercheck.agents.llm_client import ENV_FLAG, LLMClient


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
DEFAULT_MODELS: Mapping[Tier, str] = {
    Tier.SMALL: "anthropic/claude-haiku-4.5",
    Tier.LARGE: "anthropic/claude-sonnet-4.5",
}
MODEL_ENV: Mapping[Tier, str] = {
    Tier.SMALL: "LEDGERCHECK_MODEL_SMALL",
    Tier.LARGE: "LEDGERCHECK_MODEL_LARGE",
}

Transport = Callable[[str, str], str]


def resolve_models(env: Mapping[str, str] | None = None) -> dict[Tier, str]:
    """Model id per tier: the stripped env override, or the default when unset or blank."""
    env = os.environ if env is None else env
    models = {}
    for tier, default in DEFAULT_MODELS.items():
        override = (env.get(MODEL_ENV[tier]) or "").strip()
        models[tier] = override or default
    return models


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


class _BaseRouter:
    def __init__(self, env: Mapping[str, str] | None = None) -> None:
        self.models = resolve_models(env)

    def route(self, task: TaskKind | str) -> Route:
        tier = TASK_TIERS[TaskKind(task)]
        model = self.models[tier]
        large = self.models[Tier.LARGE]
        fallback = large if tier is Tier.SMALL and large != model else None
        return Route(TaskKind(task), tier, model, fallback)


class MockRouter(_BaseRouter):
    """Offline router: records ``(task, model, prompt)`` in ``calls`` and returns a canned reply."""

    name = "mock"

    def __init__(
        self,
        env: Mapping[str, str] | None = None,
        replies: Mapping[TaskKind, str] | None = None,
    ) -> None:
        super().__init__(env)
        self.replies = dict(replies or {})
        self.calls: list[tuple[TaskKind, str, str]] = []

    def complete(self, task: TaskKind, prompt: str) -> str:
        route = self.route(task)
        self.calls.append((route.task, route.model, prompt))
        return self.replies.get(route.task, "")


class OpenRouterRouter(_BaseRouter):
    """Live router behind the ``LLMClient`` spend gate, checked on construction and every call."""

    name = "openrouter"

    def __init__(
        self,
        env: Mapping[str, str] | None = None,
        transport: Transport | None = None,
    ) -> None:
        self._client = LLMClient(env)  # the gate: raises LiveLLMDisabled first
        super().__init__(env)
        self._env = env  # None keeps reading os.environ live
        self._transport = transport

    def route(self, task: TaskKind | str) -> Route:
        LLMClient(self._env)  # the gate again: the flag or key may be gone since __init__
        return super().route(task)

    def complete(self, task: TaskKind, prompt: str) -> str:
        LLMClient(self._env)  # the gate again: the flag or key may be gone since __init__
        route = self.route(task)
        if self._transport is None:
            raise NotImplementedError(
                f"live OpenRouter completion ({route.model}) is not implemented yet"
            )
        return self._transport(route.model, prompt)


def default_router(env: Mapping[str, str] | None = None) -> ModelRouter:
    """``MockRouter`` unless the live flag is ``"1"``; then ``OpenRouterRouter``, which checks the key."""
    env = os.environ if env is None else env
    if env.get(ENV_FLAG) != "1":
        return MockRouter(env)
    return OpenRouterRouter(env)
