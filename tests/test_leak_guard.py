"""scripts/leak_guard.py scans committed content only (HEAD, --history, --pre-push), offline."""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GUARD = ROOT / "scripts" / "leak_guard.py"
ZERO = "0" * 40
# Built at runtime so this file never holds a key-shaped string itself.
FAKE_OPENROUTER = "sk-or-v1-" + "a1B2" * 10
FAKE_GENERIC = "sk-" + "Zy9x" * 10
FAKE_VALUE = "q8" * 12
FAKE_LANGFUSE = "sk-lf-" + "c3D4" * 6
# Temp repos must not run your own git hooks or need signing keys.
GIT = ["git", "-c", "core.hooksPath=/dev/null", "-c", "commit.gpgsign=false",
       "-c", "user.name=Leak Test", "-c", "user.email=leak@test.invalid"]


def guard(root: Path, *args: str, stdin: str = "") -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(GUARD), "--root", str(root), *args],
                          input=stdin, capture_output=True, text=True, timeout=60)


def git(root: Path, *args: str) -> str:
    done = subprocess.run([*GIT, "-C", str(root), *args], check=True, capture_output=True,
                          text=True, timeout=60)
    return done.stdout.strip()


def commit(root: Path, files: dict[str, str | None], message: str = "change") -> str:
    """Write (or, for ``None``, delete via git) ``files``, commit, return the sha."""
    if not (root / ".git").exists():
        git(root, "init", "-q")
    for name, text in files.items():
        if text is None:
            git(root, "rm", "-q", name)
            continue
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text, encoding="utf-8")
        git(root, "add", name)
    git(root, "commit", "-q", "--allow-empty", "-m", message)
    return git(root, "rev-parse", "HEAD")


def pre_push(local: str, remote: str = ZERO) -> str:
    return f"refs/heads/main {local} refs/heads/main {remote}\n"


def test_this_repo_is_clean_at_head_and_in_history():
    for args in ((), ("--history",)):
        done = guard(ROOT, *args)
        assert done.returncode == 0, done.stderr
        assert "leak guard: clean" in done.stdout


def test_the_examples_and_test_fakes_pass(tmp_path):
    commit(tmp_path, {
        ".env.example": "OPENROUTER_API_KEY=your-openrouter-key-here\n"
                        "LEDGERCHECK_LLM_BASE_URL=https://openrouter.ai/api/v1\n",
        "connections.example.toml": 'api_key = "your-key-here"\n'
                                    'base_url = "https://openrouter.ai/api/v1"\n',
        "tests/test_x.py": 'api_key = "test-key-not-real"\n',
    })
    for args in ((), ("--history",)):
        assert guard(tmp_path, *args).returncode == 0


SLUGS = ("risk-assessment-for-vendor-invoices-and-approvals",
         "task-reconcile-every-open-purchase-order-line-items",
         "disk-usage-report-for-the-quarterly-ledger-archive",
         "desk-lf-layout-notes-for-the-finance-office-team")


def commit_bytes(root: Path, name: str, data: bytes, message: str = "change") -> str:
    if not (root / ".git").exists():
        git(root, "init", "-q")
    (root / name).parent.mkdir(parents=True, exist_ok=True)
    (root / name).write_bytes(data)
    git(root, "add", name)
    return commit(root, {}, message)


def test_hyphenated_words_are_not_keys(tmp_path):
    text = "".join(f"See {s}, docs/{s}.md, \"line\\n{s}\" and q=%3D{s}\n" for s in SLUGS)
    commit(tmp_path, {"docs/notes.md": text})
    commit_bytes(tmp_path, "docs/notes-utf16.txt", text.encode("utf-16"))
    done = guard(tmp_path, "--history")
    assert done.returncode == 0, done.stderr


def blob_cases():
    key = FAKE_OPENROUTER.encode()
    return {
        "latin-1 stray byte": b"caf\xe9 au lait\nOPENROUTER_API_KEY=" + key + b"\n",
        "utf-16 with BOM": f"note\nkey: {FAKE_GENERIC}\n".encode("utf-16"),
        "utf-16-be, no BOM": f"note\nkey: {FAKE_LANGFUSE}\n".encode("utf-16-be"),
        "binary blob": b"SQLite format 3\x00\x10\x00\x01\x01" + bytes(range(256)) + key
                       + b"\x00\xff\xfe" + bytes(range(255, -1, -1)),
        "utf-16 at an odd offset in binary": b"\x89PNG\r\n\x1a\x00\x07"
                                             + FAKE_OPENROUTER.encode("utf-16-le") + b"\xff",
    }


@pytest.mark.parametrize("case", list(blob_cases()))
def test_keys_are_found_in_any_encoding_and_in_binary_blobs(tmp_path, case):
    commit_bytes(tmp_path, "data/blob.bin", blob_cases()[case])
    done = guard(tmp_path)
    assert done.returncode == 1 and "data/blob.bin:" in done.stderr, done.stderr
    for key in (FAKE_OPENROUTER, FAKE_GENERIC, FAKE_LANGFUSE):
        assert key not in done.stderr


ESCAPED = ('{{"a": "line\\n{key}"}}\n', '{{"a": "tab\\t{key}"}}\n', '{{"a": "\\u0022{key}"}}\n',
           "https://x.invalid/cb?token%3D{key}\n", "x=%22{key}%22\n")


@pytest.mark.parametrize("text, key", [
    *((t, k) for t in ESCAPED for k in (FAKE_OPENROUTER, FAKE_GENERIC, FAKE_LANGFUSE)),
    # The generic sk- shape must start a token (snake_case words); the prefixed ones need not.
    ("export foo_{key}\n", FAKE_OPENROUTER), ("export foo_{key}\n", FAKE_LANGFUSE),
])
def test_keys_right_after_escapes_or_letters_are_found(tmp_path, text, key):
    commit(tmp_path, {"app/data.json": text.format(key=key)})
    done = guard(tmp_path)
    assert done.returncode == 1 and "app/data.json:1:" in done.stderr, done.stderr
    assert key not in done.stderr


def test_commit_and_tag_messages_are_scanned(tmp_path):
    base = commit(tmp_path, {"README.md": "hello\n"}, "base")
    leaky = commit(tmp_path, {"README.md": "hello again\n"}, f"rotate {FAKE_OPENROUTER}")
    done = guard(tmp_path, "--history")
    assert done.returncode == 1 and f"{leaky[:12]} (commit message):" in done.stderr
    assert guard(tmp_path, "--pre-push", stdin=pre_push(leaky, base)).returncode == 1
    assert guard(tmp_path).returncode == 1  # HEAD's own message
    clean = commit(tmp_path, {"README.md": "bye\n"}, "clean")
    git(tmp_path, "tag", "-a", "v1", "-m", f"release notes {FAKE_LANGFUSE}", clean)
    tag = git(tmp_path, "rev-parse", "v1")
    done = guard(tmp_path, "--pre-push", stdin=f"refs/tags/v1 {tag} refs/tags/v1 {ZERO}\n")
    assert done.returncode == 1 and "(tag message)" in done.stderr
    assert FAKE_LANGFUSE not in done.stderr and FAKE_OPENROUTER not in done.stderr


@pytest.mark.parametrize("line", [
    "{key}\n", "OPENROUTER_API_KEY={key}\n", 'value: "{key}"\n', "value: '{key}'\n",
    "export KEY {key}\n", "\t{key} trailing\n",
])
@pytest.mark.parametrize("key", [FAKE_OPENROUTER, FAKE_GENERIC, FAKE_LANGFUSE])
def test_keys_at_a_token_start_still_fail(tmp_path, line, key):
    commit(tmp_path, {"app/config.txt": line.format(key=key)})
    done = guard(tmp_path)
    assert done.returncode == 1 and "app/config.txt:1:" in done.stderr
    assert key not in done.stderr


@pytest.mark.parametrize("name, text, expect", [
    (".env", "X=1\n", ".env: local .env file is committed"),
    ("deploy/.ENV", "X=1\n", "deploy/.ENV: local .env file is committed"),
    ("config/.env.production", "X=1\n", "local .env file is committed"),
    ("connections.local.toml", 'provider = "mock"\n', "local connections file is committed"),
    ("app/settings.py", f'KEY = "{FAKE_OPENROUTER}"\n', "app/settings.py:1: OpenRouter key"),
    ("notes.md", f"token: {FAKE_GENERIC}\n", "notes.md:1: API key"),
    ("conf.toml", f'\n api_key = "{FAKE_VALUE}"\n', "conf.toml:2: api_key assignment"),
    (".env.example", f"OPENROUTER_API_KEY={FAKE_VALUE}\n", "example OPENROUTER_API_KEY"),
    ("connections.example.toml", 'base_url = "https://llm.internal.corp/v1"\n',
     "example endpoint"),
])
def test_planted_fakes_fail_with_redacted_output(tmp_path, name, text, expect):
    commit(tmp_path, {name: text})
    done = guard(tmp_path)
    assert done.returncode == 1
    assert expect in done.stderr and "before pushing" in done.stderr
    for secret in (FAKE_OPENROUTER, FAKE_GENERIC, FAKE_VALUE, "llm.internal.corp"):
        assert secret not in done.stderr + done.stdout


@pytest.fixture
def key_then_removed(tmp_path):
    """base (clean) -> leak (adds a key) -> fix (deletes it): HEAD itself is clean."""
    base = commit(tmp_path, {"README.md": "hello\n"}, "base")
    leak = commit(tmp_path, {"app/settings.py": f'KEY = "{FAKE_OPENROUTER}"\n'}, "leak")
    fix = commit(tmp_path, {"app/settings.py": None}, "fix")
    return tmp_path, base, leak, fix


def test_a_key_removed_later_is_still_found_in_history(key_then_removed):
    root, _, leak, _ = key_then_removed
    assert guard(root).returncode == 0  # the HEAD tree is clean
    done = guard(root, "--history")
    assert done.returncode == 1
    assert f"{leak[:12]} app/settings.py:1: OpenRouter key" in done.stderr
    assert FAKE_OPENROUTER not in done.stderr


def test_pre_push_scans_every_commit_in_the_pushed_range(key_then_removed):
    root, base, leak, fix = key_then_removed
    assert guard(root, "--pre-push", stdin=pre_push(fix)).returncode == 1  # new branch
    assert guard(root, "--pre-push", stdin=pre_push(fix, base)).returncode == 1  # base..fix
    assert guard(root, "--pre-push", stdin=pre_push(fix, leak)).returncode == 0  # only the fix
    assert guard(root, "--pre-push", stdin=pre_push(ZERO, fix)).returncode == 0  # ref deletion
    assert guard(root, "--pre-push").returncode == 0  # nothing to push


def test_a_new_branch_is_scanned_whole_even_if_a_remote_seems_to_have_it(key_then_removed):
    root, _, _, fix = key_then_removed
    git(root, "remote", "add", "public", "https://public.invalid/x.git")  # never fetched
    git(root, "update-ref", "refs/remotes/public/main", fix)  # possibly stale tracking ref
    unknown_tip = "1234567" * 5 + "89abc"  # a remote sha we never fetched
    for remote_sha in (ZERO, unknown_tip):
        done = guard(root, "--pre-push", stdin=pre_push(fix, remote_sha))
        assert done.returncode == 1 and "OpenRouter key" in done.stderr


def test_the_hook_scans_what_git_says_is_being_pushed(key_then_removed):
    root, base, _, fix = key_then_removed
    hook = ROOT / ".githooks" / "pre-push"
    git(root, "update-ref", "refs/remotes/origin/main", base)  # the remote already has base
    (root / "scripts").mkdir()
    (root / "scripts" / "leak_guard.py").write_text(GUARD.read_text())  # left uncommitted
    for args, stdin in ((["origin", "x"], pre_push(fix, base)),
                        (["https://public.invalid/x.git"] * 2, pre_push(fix))):
        run = subprocess.run(["sh", str(hook), *args], input=stdin, cwd=root,
                             capture_output=True, text=True, timeout=60)
        assert run.returncode == 1 and "OpenRouter key" in run.stderr


def test_uncommitted_files_are_not_scanned(tmp_path):
    commit(tmp_path, {"app/settings.py": "KEY = None\n"})
    (tmp_path / "app" / "settings.py").write_text(f'KEY = "{FAKE_OPENROUTER}"\n')
    (tmp_path / ".env").write_text(f"OPENROUTER_API_KEY={FAKE_OPENROUTER}\n")
    git(tmp_path, "add", "app/settings.py")  # staged but not committed either
    for args in ((), ("--history",)):
        assert guard(tmp_path, *args).returncode == 0


def test_not_a_git_checkout_is_an_error(tmp_path):
    assert guard(tmp_path).returncode == 2


def test_pre_push_hook_passes_stdin_to_the_guard_and_is_executable_in_git():
    hook = (ROOT / ".githooks" / "pre-push").read_text(encoding="utf-8")
    assert 'exec python3 "$root/scripts/leak_guard.py" --root "$root" --pre-push\n' in hook
    done = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-s", ".githooks/pre-push"],
                          capture_output=True, text=True, timeout=60)
    assert done.stdout.startswith("100755 "), done.stdout
    run = subprocess.run(["sh", str(ROOT / ".githooks" / "pre-push"), "origin", "x"], input="",
                         cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert run.returncode == 0 and "leak guard: clean" in run.stdout


def test_local_config_files_are_ignored_by_git_and_docker():
    for ignore in (".gitignore", ".dockerignore"):
        lines = (ROOT / ignore).read_text(encoding="utf-8").splitlines()
        assert {".env", ".env.*", "connections.local*"} <= set(lines)
