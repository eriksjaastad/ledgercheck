"""Smoke: package imports and CLI help exits 0."""

from ledgercheck import __version__
from ledgercheck.cli import main


def test_version_is_semver_stub() -> None:
    assert __version__ == "0.1.0"


def test_cli_help_exits_zero() -> None:
    assert main([]) == 0
