"""Suite-wide guards: tests never see real Langfuse keys (tracing stays off) or model overrides."""

import pytest

from ledgercheck.agents.routing import MODEL_ENV
from ledgercheck.observability import HOST_ENV, PUBLIC_KEY_ENV, SECRET_KEY_ENV, _langfuse_tracer


@pytest.fixture(autouse=True)
def no_langfuse_keys(monkeypatch):
    for name in (PUBLIC_KEY_ENV, SECRET_KEY_ENV, HOST_ENV, *MODEL_ENV.values()):
        monkeypatch.delenv(name, raising=False)
    _langfuse_tracer.cache_clear()
    yield
    _langfuse_tracer.cache_clear()
