"""Suite-wide guards: no real connection settings, no local config files, no network.

Every test starts with the connection env vars removed (so keys or a live flag
in your shell can't reach a test), with ``.env`` and ``connections.local.toml``
pointed at an empty temp dir, and with any socket connection outside loopback
failing the test. ``ledgercheck serve`` tests talk to 127.0.0.1 only.
"""

import socket

import pytest

from ledgercheck import connections

CONNECTION_ENV = (
    connections.ENV_FLAG, connections.CONNECTIONS_FILE_ENV, connections.PUBLIC_KEY_ENV,
    connections.SECRET_KEY_ENV, connections.HOST_ENV, "LEDGERCHECK_POLICY_RERANK",
    *(var for var, _ in connections.SETTINGS.values()),
)


@pytest.fixture(autouse=True)
def no_connection_settings(monkeypatch, tmp_path_factory):
    for name in CONNECTION_ENV:
        monkeypatch.delenv(name, raising=False)
    empty = tmp_path_factory.mktemp("no-connections")
    monkeypatch.setattr(connections, "DOTENV_PATH", empty / ".env")
    monkeypatch.setattr(connections, "DEFAULT_CONFIG", empty / "connections.local.toml")
    connections._langfuse_tracer.cache_clear()
    yield
    connections._langfuse_tracer.cache_clear()


def _loopback(host) -> bool:
    host = None if host is None else str(host).split("%")[0]
    return host in (None, "localhost", "::1") or (host or "").startswith("127.")


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Fail any connection to a non-loopback address (DNS lookups included)."""
    real_connect, real_connect_ex = socket.socket.connect, socket.socket.connect_ex
    real_getaddrinfo, real_create = socket.getaddrinfo, socket.create_connection

    def guard(address):
        host = address[0] if isinstance(address, tuple) else None
        if not isinstance(address, (str, bytes)) and not _loopback(host):
            pytest.fail(f"network connection attempted: {address!r}")

    def connect(self, address):
        guard(address)
        return real_connect(self, address)

    def connect_ex(self, address):
        guard(address)
        return real_connect_ex(self, address)

    def getaddrinfo(host, *args, **kwargs):
        guard((host,))
        return real_getaddrinfo(host, *args, **kwargs)

    def create_connection(address, *args, **kwargs):
        guard(address)
        return real_create(address, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", connect_ex)
    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
    monkeypatch.setattr(socket, "create_connection", create_connection)
