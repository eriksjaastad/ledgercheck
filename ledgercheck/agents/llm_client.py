"""Live extraction and rerank client. Behind the spend gate; not implemented yet.

``LLMClient`` is what intake (``IntakeAgent.extract_text``) and the policy
reranker use when no client is injected. Constructing one runs the spend gate
in ``ledgercheck.connections``: it raises ``LiveLLMDisabled`` unless
``LEDGERCHECK_LLM`` is exactly ``"1"`` **and** ``OPENROUTER_API_KEY`` is
non-blank (from the environment, ``.env`` or ``connections.local.toml``),
naming what is missing. Any other flag value (``"true"``, ``"0"``, unset)
counts as off.

Past the gate, ``extract_invoice`` and ``rerank`` raise
``NotImplementedError`` without any request: parsing invoice text and
reranking with a model are not built yet. The live path that is built is the
LLM judge (``ledgercheck judge --live``, see ``ledgercheck.eval.judge``).
Tests inject fakes (an ``IntakeAgent`` client, a ``PolicyAgent`` reranker)
instead of enabling the flag.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ledgercheck.connections import API_KEY_ENV, ENV_FLAG, LiveLLMDisabled, require_live

__all__ = ["API_KEY_ENV", "ENV_FLAG", "LLMClient", "LiveLLMDisabled"]


class LLMClient:
    """Live LLM client (extraction and rerank); constructing one is the spend gate.

    ``env`` is passed to ``connections.require_live``: the process environment
    and local files by default, or exactly the given mapping. The key stays in
    ``connections`` and is never logged or echoed in errors.
    """

    name = "openrouter"

    def __init__(self, env: Mapping[str, str] | None = None) -> None:
        require_live(env)

    def extract_invoice(self, text: str) -> Mapping[str, Any]:
        """Return invoice fields in ``Invoice.from_dict`` shape for ``text``.

        Not implemented yet: raises ``NotImplementedError`` without any request.
        """
        raise NotImplementedError("live LLM extraction is not implemented yet")

    def rerank(self, query: str, candidates: Sequence[str]) -> Sequence[str]:
        """Return ``candidates`` (policy chunk ids) reordered by relevance to ``query``.

        Not implemented yet: raises ``NotImplementedError`` without any request.
        """
        raise NotImplementedError("live LLM rerank is not implemented yet")
