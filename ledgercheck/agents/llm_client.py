"""Gate for live LLM calls (OpenRouter). Off by default; spends nothing.

Live LLM calls (intake extraction, policy rerank) are opt-in twice over.
``LLMClient`` refuses to construct unless ``LEDGERCHECK_LLM`` is exactly
``"1"`` **and** ``OPENROUTER_API_KEY`` is non-blank (not empty or whitespace
only), raising ``LiveLLMDisabled`` that names what is missing. Any other flag value (``"true"``, ``"0"``, unset) counts
as off.

Even with both set, this module makes no network call: ``extract_invoice``
and ``rerank`` (the policy stage's optional reranker) raise
``NotImplementedError``. Model choice per task (small for extraction, large
for rerank, approval reasoning and judging) lives in
``ledgercheck.agents.routing``, whose live router is built through this same
gate; its module docstring also holds the rate-limit and context-window
runbook. The real OpenRouter request is not implemented yet. Tests inject
fakes (an ``IntakeAgent`` client, a ``PolicyAgent`` reranker) instead of
enabling the flag.
"""

from __future__ import annotations

import os
from typing import Any, Mapping, Sequence

ENV_FLAG = "LEDGERCHECK_LLM"
API_KEY_ENV = "OPENROUTER_API_KEY"


class LiveLLMDisabled(RuntimeError):
    """A live LLM call was requested but is not explicitly enabled."""


class LLMClient:
    """Live LLM client (extraction and rerank); constructing one is the spend gate.

    ``env`` defaults to ``os.environ``; pass a mapping to check a specific
    environment. The key is kept on the instance and never logged or echoed
    in errors.
    """

    name = "openrouter"

    def __init__(self, env: Mapping[str, str] | None = None) -> None:
        env = os.environ if env is None else env
        if env.get(ENV_FLAG) != "1":
            raise LiveLLMDisabled(
                f"live LLM calls are off: set {ENV_FLAG}=1 and {API_KEY_ENV} to enable it"
            )
        key = env.get(API_KEY_ENV)
        if key is None:
            raise LiveLLMDisabled(f"{ENV_FLAG}=1 but {API_KEY_ENV} is not set")
        if not key.strip():
            raise LiveLLMDisabled(f"{ENV_FLAG}=1 but {API_KEY_ENV} is blank")
        self._api_key = key

    def extract_invoice(self, text: str) -> Mapping[str, Any]:
        """Return invoice fields in ``Invoice.from_dict`` shape for ``text``.

        Not implemented yet: raises ``NotImplementedError`` without any request.
        """
        raise NotImplementedError("live OpenRouter extraction is not implemented yet")

    def rerank(self, query: str, candidates: Sequence[str]) -> Sequence[str]:
        """Return ``candidates`` (policy chunk ids) reordered by relevance to ``query``.

        Not implemented yet: raises ``NotImplementedError`` without any request.
        """
        raise NotImplementedError("live OpenRouter rerank is not implemented yet")
