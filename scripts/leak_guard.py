#!/usr/bin/env python3
"""Fail if a tracked file looks like a secret or a local connections file.

Usage: python3 scripts/leak_guard.py [--root DIR]

Checks every file ``git ls-files`` lists (names and contents):

- a tracked ``.env`` or ``.env.*`` other than ``.env.example``, or a tracked
  ``connections.local*``: those files are for your machine only;
- an API-key-shaped string (OpenRouter ``sk-or-v1-...``, other ``sk-...``
  keys, Langfuse ``pk-lf-``/``sk-lf-`` keys) anywhere;
- ``api_key = "<value>"`` style assignments (also ``secret``, ``token``,
  ``password``) with a long value that is not an obvious placeholder;
- in ``.env.example`` and ``connections.example.toml``: any non-empty key
  value that is not a placeholder, and any URL outside the allowed hosts.

Placeholders are values containing one of ``ALLOWED_MARKERS``. Matches are
printed as ``file:line: what (redacted)`` and never in full. Exit 0 when
clean, 1 on findings, 2 when DIR is not a git checkout. Standard library only;
run by ``.githooks/pre-push`` (enable with ``git config core.hooksPath
.githooks``) and by CI.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath

KEY_PATTERNS = (
    ("OpenRouter key", re.compile(r"sk-or-v1-[A-Za-z0-9]{20,}")),
    ("API key", re.compile(r"sk-(?!or-v1-)(?:ant-|proj-)?[A-Za-z0-9_-]{32,}")),
    ("Langfuse key", re.compile(r"[ps]k-lf-[A-Za-z0-9-]{20,}")),
)
ASSIGNMENT = re.compile(
    r"""(?i)\b[\w-]*(api[_-]?key|secret|token|password)\s*[:=]\s*["']([^"'\s]{16,})["']""")
EXAMPLE_FILES = (".env.example", "connections.example.toml")
EXAMPLE_VALUE = re.compile(r"""(?i)^\s*(?:export\s+)?([\w.-]+)\s*=\s*["']?([^"'#\s]*)""")
URL = re.compile(r"https?://([^/\s\"']+)")
ALLOWED_HOSTS = ("openrouter.ai", "cloud.langfuse.com", "github.com", "ghcr.io")
# Keep this list short: anything containing one of these is a placeholder.
ALLOWED_MARKERS = ("your-", "-here", "not-real", "placeholder", "example", "redacted", "fake")


def _placeholder(value: str) -> bool:
    return any(marker in value.lower() for marker in ALLOWED_MARKERS)


def _redact(text: str) -> str:
    return f"{text[:6]}...[redacted]"


def tracked_files(root: Path) -> list[str]:
    done = subprocess.run(["git", "-C", str(root), "ls-files", "-z"], capture_output=True,
                          timeout=60)
    if done.returncode != 0:
        raise NotADirectoryError(f"{root} is not a git checkout")
    return [name for name in done.stdout.decode("utf-8").split("\0") if name]


def check_name(name: str) -> str | None:
    base = PurePosixPath(name).name
    if (base == ".env" or base.startswith(".env.")) and base != ".env.example":
        return "local .env file is tracked"
    if base.startswith("connections.local"):
        return "local connections file is tracked"
    return None


def check_text(name: str, text: str) -> list[tuple[int, str]]:
    """``(line, finding)`` pairs for one file's content; matches are redacted."""
    findings = []
    example = PurePosixPath(name).name in EXAMPLE_FILES
    for n, line in enumerate(text.splitlines(), 1):
        for what, pattern in KEY_PATTERNS:
            findings += [(n, f"{what} {_redact(m.group())}") for m in pattern.finditer(line)
                         if not _placeholder(m.group())]
        findings += [(n, f"{m.group(1)} assignment {_redact(m.group(2))}")
                     for m in ASSIGNMENT.finditer(line) if not _placeholder(m.group(2))]
        if not example or line.lstrip().startswith("#"):
            continue
        m = EXAMPLE_VALUE.match(line)
        if m and re.search(r"(?i)(key|secret|token)$", m.group(1)) and m.group(2) \
                and not _placeholder(m.group(2)):
            findings.append((n, f"example {m.group(1)} is not a placeholder"))
        for url in URL.finditer(line):
            host = url.group(1).lower()
            if not (host in ALLOWED_HOSTS or host.endswith(".example") or _placeholder(host)):
                findings.append((n, f"example endpoint {_redact(host)} is not an allowed host"))
    return findings


def scan(root: Path) -> list[str]:
    problems = []
    for name in tracked_files(root):
        if (why := check_name(name)) is not None:
            problems.append(f"{name}: {why}")
        path = root / name
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue  # binary
        problems += [f"{name}:{n}: {finding}" for n, finding in check_text(name, text)]
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("."), help="git checkout to scan")
    args = parser.parse_args(argv)
    try:
        problems = scan(args.root)
    except NotADirectoryError as exc:
        print(f"leak guard: {exc}", file=sys.stderr)
        return 2
    for problem in problems:
        print(problem, file=sys.stderr)
    if problems:
        print(f"leak guard: {len(problems)} possible secret(s) in tracked files; nothing pushed",
              file=sys.stderr)
        return 1
    print("leak guard: clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
