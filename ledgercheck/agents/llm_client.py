"""Live extraction and rerank client, behind the spend gate. Rerank is not built yet.

``LLMClient`` is what intake (``IntakeAgent.extract_text``) and the policy
reranker use when no client is injected. Constructing one runs the spend gate
in ``ledgercheck.connections``: it raises ``LiveLLMDisabled`` unless
``LEDGERCHECK_LLM`` is exactly ``"1"`` **and** ``OPENROUTER_API_KEY`` is
non-blank (from the environment, ``.env`` or ``connections.local.toml``),
naming what is missing. Any other flag value (``"true"``, ``"0"``, unset)
counts as off.

Past the gate, ``extract_invoice`` sends one request to the router's small-tier
model through ``connections.router()``, so it needs provider ``openrouter``, a
small-tier model id with prices and a spend cap (see
``ledgercheck connections --help``). It is tested only against a fake server.
Text over ``MAX_INPUT_TOKENS`` (estimated at 4 characters per token) is refused
with ``InputTooLarge`` before any request; it is never truncated or split.
``rerank`` raises ``NotImplementedError`` without any request. Tests inject
fakes (an ``IntakeAgent`` client, a ``PolicyAgent`` reranker, or a router over
a fake opener) instead of enabling the flag.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from ledgercheck import connections
from ledgercheck.agents.routing import ModelRouter, TaskKind
from ledgercheck.connections import (
    API_KEY_ENV,
    ENV_FLAG,
    ConnectionConfigError,
    LiveLLMDisabled,
    require_live,
)

__all__ = ["API_KEY_ENV", "ENV_FLAG", "LLMClient", "LiveLLMDisabled"]

EXTRACT_PROMPT = (
    "Extract the invoice below. Reply with only one JSON object with keys "
    "invoice_number, vendor_name, invoice_date (YYYY-MM-DD), currency (3-letter ISO code), "
    "line_items (a list of objects with description, quantity, unit_price, amount, sku), "
    "subtotal, tax_amount, total, and, only when the document states them, vendor_id, "
    "po_number, tax_rate, due_date (YYYY-MM-DD). Write numbers as plain decimal strings "
    'with "." as the decimal separator, no thousands separators or currency symbols; '
    "credit notes are negative. Copy values as written: never compute or guess a missing "
    "value, omit it (sku: null).\n\n"
)


CHARS_PER_TOKEN = 4  # the usual rough estimate; enough to size a prompt, not to bill it
MAX_INPUT_TOKENS = 6000  # estimated tokens of invoice text per request, well inside any window


class InputTooLarge(ValueError):
    """The invoice text is over ``MAX_INPUT_TOKENS``; nothing was sent. Its message is only counts."""


class LLMClient:
    """Live LLM client (extraction and rerank); constructing one is the spend gate.

    ``env`` is passed to ``connections.require_live``: the process environment
    and local files by default, or exactly the given mapping. ``router``
    defaults to ``connections.router(env)``. The key stays in ``connections``
    and is never logged or echoed in errors.
    """

    name = "openrouter"

    def __init__(self, env: Mapping[str, str] | None = None, *,
                 router: ModelRouter | None = None) -> None:
        require_live(env)
        self._router = connections.router(env) if router is None else router

    def extract_invoice(self, text: str) -> Mapping[str, Any]:
        """Return invoice fields in ``Invoice.from_dict`` shape for ``text``.

        One request to the extraction model. Text whose estimated size
        (``CHARS_PER_TOKEN`` characters per token) is over ``MAX_INPUT_TOKENS``
        raises ``InputTooLarge`` before any request: a truncated invoice can drop
        line items and still look valid. The reply's JSON object is the
        text from its first ``{`` to its last ``}``, so a fenced reply still
        parses. Raises ``ValueError`` when there is no JSON object or it does
        not parse; nothing is retried. Raises ``ConnectionConfigError``, before
        any request, while the provider is still ``mock``.
        """
        if self._router.name == "mock":  # the gate passes on flag and key alone
            raise ConnectionConfigError(
                "live extraction needs provider openrouter (now 'mock'): set "
                "LEDGERCHECK_LLM_PROVIDER=openrouter or provider in connections.local.toml")
        tokens = -(-len(text) // CHARS_PER_TOKEN)
        if tokens > MAX_INPUT_TOKENS:
            raise InputTooLarge(
                f"the invoice text is about {tokens} tokens, over the {MAX_INPUT_TOKENS} limit: "
                "split the input or use a model with a larger window")
        reply = self._router.complete(TaskKind.EXTRACTION, EXTRACT_PROMPT + text)
        start, end = reply.find("{"), reply.rfind("}")
        if not 0 <= start < end:
            raise ValueError("no JSON object in the model reply")
        return json.loads(reply[start:end + 1])

    def rerank(self, query: str, candidates: Sequence[str]) -> Sequence[str]:
        """Return ``candidates`` (policy chunk ids) reordered by relevance to ``query``.

        Not implemented yet: raises ``NotImplementedError`` without any request.
        """
        raise NotImplementedError("live LLM rerank is not implemented yet")
