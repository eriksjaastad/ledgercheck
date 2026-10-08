"""scripts/leak_guard.py: clean on this repo, loud on planted fakes (offline, temp git repos)."""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GUARD = ROOT / "scripts" / "leak_guard.py"
# Built at runtime so this file never holds a key-shaped string itself.
FAKE_OPENROUTER = "sk-or-v1-" + "a1B2" * 10
FAKE_GENERIC = "sk-" + "Zy9x" * 10
FAKE_VALUE = "q8" * 12


def guard(root: Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(GUARD), "--root", str(root)],
                          capture_output=True, text=True, timeout=60)


def repo(tmp_path: Path, files: dict[str, str]) -> Path:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True, timeout=60)
    for name, text in files.items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(text, encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "-A"], check=True, timeout=60)
    return tmp_path


def test_this_repo_is_clean():
    done = guard(ROOT)
    assert done.returncode == 0, done.stderr
    assert "leak guard: clean" in done.stdout


def test_the_examples_and_test_fakes_pass(tmp_path):
    root = repo(tmp_path, {
        ".env.example": "OPENROUTER_API_KEY=your-openrouter-key-here\n"
                        "LEDGERCHECK_LLM_BASE_URL=https://openrouter.ai/api/v1\n",
        "connections.example.toml": 'api_key = "your-key-here"\n'
                                    'base_url = "https://openrouter.ai/api/v1"\n',
        "tests/test_x.py": 'api_key = "test-key-not-real"\n',
    })
    done = guard(root)
    assert done.returncode == 0, done.stderr


@pytest.mark.parametrize("name, text, expect", [
    (".env", "X=1\n", ".env: local .env file is tracked"),
    ("config/.env.production", "X=1\n", "local .env file is tracked"),
    ("connections.local.toml", 'provider = "mock"\n', "local connections file is tracked"),
    ("app/settings.py", f'KEY = "{FAKE_OPENROUTER}"\n', "app/settings.py:1: OpenRouter key"),
    ("notes.md", f"token: {FAKE_GENERIC}\n", "notes.md:1: API key"),
    ("conf.toml", f'\n api_key = "{FAKE_VALUE}"\n', "conf.toml:2: api_key assignment"),
    (".env.example", f"OPENROUTER_API_KEY={FAKE_VALUE}\n", "example OPENROUTER_API_KEY"),
    ("connections.example.toml", 'base_url = "https://llm.internal.corp/v1"\n',
     "example endpoint"),
])
def test_planted_fakes_fail_with_redacted_output(tmp_path, name, text, expect):
    done = guard(repo(tmp_path, {name: text}))
    assert done.returncode == 1
    assert expect in done.stderr and "nothing pushed" in done.stderr
    for secret in (FAKE_OPENROUTER, FAKE_GENERIC, FAKE_VALUE, "llm.internal.corp"):
        assert secret not in done.stderr + done.stdout


def test_not_a_git_checkout_is_an_error(tmp_path):
    assert guard(tmp_path).returncode == 2


def test_pre_push_hook_runs_the_guard_and_is_executable_in_git():
    hook = ROOT / ".githooks" / "pre-push"
    assert "scripts/leak_guard.py" in hook.read_text(encoding="utf-8")
    done = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-s", ".githooks/pre-push"],
                          capture_output=True, text=True, timeout=60)
    assert done.stdout.startswith("100755 "), done.stdout


def test_local_config_files_are_ignored_by_git_and_docker():
    for ignore in (".gitignore", ".dockerignore"):
        lines = (ROOT / ignore).read_text(encoding="utf-8").splitlines()
        assert {".env", ".env.*", "connections.local*"} <= set(lines)
