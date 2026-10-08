"""Offline checks for the GitHub Actions workflows in .github/workflows.

Plain text and regex checks (no YAML parser): CI stays offline on Python 3.11,
and the image workflow only logs in and pushes on a push to main.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS = ROOT / ".github" / "workflows"
MAIN_PUSH_ONLY = "if: github.event_name == 'push' && github.ref == 'refs/heads/main'"


def _text(name: str) -> str:
    return (WORKFLOWS / name).read_text(encoding="utf-8")


def _step(text: str, marker: str) -> str:
    """The workflow step (from its ``- `` line to the next one) containing ``marker``."""
    for step in re.split(r"\n(?=      - )", text):
        if marker in step:
            return step
    raise AssertionError(f"no step containing {marker!r}")


@pytest.mark.parametrize("name", ["ci.yml", "image.yml"])
def test_only_the_github_token_secret_is_referenced(name: str) -> None:
    secrets = set(re.findall(r"secrets\.\w+", _text(name)))
    assert secrets <= {"secrets.GITHUB_TOKEN"}


@pytest.mark.parametrize("name", ["ci.yml", "image.yml"])
def test_default_token_is_read_only(name: str) -> None:
    assert "\npermissions:\n  contents: read\n" in _text(name)


def test_ci_runs_offline_tests_and_the_judge_on_python_311() -> None:
    text = _text("ci.yml")
    assert '\nenv:\n  LEDGERCHECK_LLM: "0"\n' in text
    for unset in ("OPENROUTER_API_KEY", "LANGFUSE_"):
        assert unset not in text
    assert 'python-version: "3.11"' in text
    assert re.search(r"run: pytest -q\b", text)
    assert re.search(r"run: ledgercheck judge(?! --live)", text)
    assert "--live" not in text


def test_image_is_built_smoke_tested_and_labelled_for_the_repo() -> None:
    text = _text("image.yml")
    assert "images: ${{ env.IMAGE }}" in text and "IMAGE: ghcr.io/eriksjaastad/ledgercheck\n" in text
    assert "org.opencontainers.image.source=https://github.com/eriksjaastad/ledgercheck\n" in text
    build = _step(text, "docker/build-push-action@")
    assert "load: true" in build and "push: false" in build
    assert 'docker run --rm --network none "$LOCAL_TAG"\n' in text
    assert 'docker run --rm --network none "$LOCAL_TAG" --help\n' in text


def test_image_logs_in_and_pushes_only_on_push_to_main() -> None:
    text = _text("image.yml")
    login = _step(text, "docker/login-action@")
    assert MAIN_PUSH_ONLY in login and "password: ${{ secrets.GITHUB_TOKEN }}" in login
    assert MAIN_PUSH_ONLY in _step(text, "docker push")
    assert text.count("docker push") == 1 and "push: true" not in text
