#!/usr/bin/env python3
"""Fail if committed content looks like a secret or a local connections file.

Usage:
    python3 scripts/leak_guard.py [--root DIR]             # files in the HEAD commit
    python3 scripts/leak_guard.py --history [--root DIR]   # every commit reachable from HEAD
    python3 scripts/leak_guard.py --pre-push [--root DIR]  # commits being pushed (git hook)

Only committed content is scanned, never the working tree: what matters is
what a push would publish, and a key added in one commit and deleted in the
next is still in the pushed history. ``--pre-push`` reads git's pre-push
input (``<local ref> <local sha> <remote ref> <remote sha>`` per line) and
scans every commit in each pushed range: ``<remote sha>..<local sha>``, or
everything not yet on any remote for a new branch. Deleting a remote branch
scans nothing.

For every file a scanned commit adds or changes, it checks:

- the name: a ``.env`` or ``.env.*`` other than ``.env.example``, or a
  ``connections.local*``, is for your machine only;
- an API-key-shaped string (OpenRouter ``sk-or-v1-...``, other ``sk-...``
  keys, Langfuse ``pk-lf-``/``sk-lf-`` keys) anywhere;
- ``api_key = "<value>"`` style assignments (also ``secret``, ``token``,
  ``password``) with a long value that is not an obvious placeholder;
- in ``.env.example`` and ``connections.example.toml``: any non-empty key
  value that is not a placeholder, and any URL outside the allowed hosts.

Placeholders are values containing one of ``ALLOWED_MARKERS``. Findings are
printed as ``[commit] file:line: what`` with the match redacted. Exit 0 when
clean, 1 on findings, 2 when DIR is not a git checkout or git fails.
Standard library only; run by ``.githooks/pre-push`` (enable with ``git
config core.hooksPath .githooks``) and by CI with ``--history``.
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
ZERO_SHA = re.compile(r"0+")


class GitError(RuntimeError):
    """Not a git checkout, or a git command failed."""


def _placeholder(value: str) -> bool:
    return any(marker in value.lower() for marker in ALLOWED_MARKERS)


def _redact(text: str) -> str:
    return f"{text[:6]}...[redacted]"


def git(root: Path, *args: str, stdin: bytes | None = None) -> bytes:
    done = subprocess.run(["git", "-C", str(root), *args], input=stdin, capture_output=True,
                          timeout=120)
    if done.returncode != 0:
        detail = done.stderr.decode("utf-8", "replace").strip()[:200]
        raise GitError(f"git {args[0]} failed in {root}: {detail}")
    return done.stdout


def head_files(root: Path) -> list[tuple[str, str]]:
    """``(path, blob sha)`` for every file in the HEAD commit."""
    files = []
    for entry in git(root, "ls-tree", "-r", "-z", "HEAD").split(b"\0"):
        meta, _, path = entry.partition(b"\t")
        fields = meta.decode().split()  # mode type sha
        if len(fields) == 3 and fields[1] == "blob":
            files.append((path.decode("utf-8", "replace"), fields[2]))
    return files


def changed_files(root: Path, commit: str) -> list[tuple[str, str]]:
    """``(path, blob sha)`` for every file ``commit`` adds or changes (vs. each parent)."""
    out = git(root, "diff-tree", "--root", "-m", "-r", "-z", "--no-renames",
              "--diff-filter=d", "--no-commit-id", commit).split(b"\0")
    files = []
    for meta, path in zip(out[0::2], out[1::2]):
        fields = meta.decode().split()  # :old-mode new-mode old-sha new-sha status
        if len(fields) == 5 and fields[1] != "160000":  # 160000 is a submodule
            files.append((path.decode("utf-8", "replace"), fields[3]))
    return files


def _is_commit(root: Path, sha: str) -> bool:
    done = subprocess.run(["git", "-C", str(root), "cat-file", "-e", f"{sha}^{{commit}}"],
                          capture_output=True, timeout=60)
    return done.returncode == 0


def pushed_commits(root: Path, lines: list[str]) -> list[str]:
    """Commits in the ranges a pre-push hook is given (one input line per ref)."""
    commits: dict[str, None] = {}
    for line in lines:
        parts = line.split()
        if len(parts) != 4 or ZERO_SHA.fullmatch(parts[1]):  # malformed, or a ref deletion
            continue
        local, remote = parts[1], parts[3]
        if not ZERO_SHA.fullmatch(remote) and _is_commit(root, remote):
            spec = [f"{remote}..{local}"]
        else:  # a new branch, or a remote tip we don't have: all not yet on a remote
            spec = [local, "--not", "--remotes"]
        commits.update(dict.fromkeys(git(root, "rev-list", *spec).decode().split()))
    return list(commits)


def read_blobs(root: Path, shas: list[str]) -> dict[str, bytes]:
    """Blob contents by sha, from one ``git cat-file --batch``."""
    unique = list(dict.fromkeys(shas))
    if not unique:
        return {}
    out = git(root, "cat-file", "--batch", stdin="".join(f"{s}\n" for s in unique).encode())
    blobs, pos = {}, 0
    for sha in unique:
        header_end = out.index(b"\n", pos)
        size = int(out[pos:header_end].split()[2])
        blobs[sha] = out[header_end + 1:header_end + 1 + size]
        pos = header_end + 1 + size + 1
    return blobs


def check_name(name: str) -> str | None:
    base = PurePosixPath(name).name
    if (base == ".env" or base.startswith(".env.")) and base != ".env.example":
        return "local .env file is committed"
    if base.startswith("connections.local"):
        return "local connections file is committed"
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


def files_to_scan(root: Path, mode: str, stdin_lines: list[str]) -> list[tuple[str, str, str]]:
    """``(label, path, blob sha)`` for the mode: ``head``, ``history`` or ``pre-push``."""
    if mode == "head":
        return [("", path, sha) for path, sha in head_files(root)]
    commits = (git(root, "rev-list", "HEAD").decode().split() if mode == "history"
               else pushed_commits(root, stdin_lines))
    return [(f"{c[:12]} ", path, sha) for c in commits for path, sha in changed_files(root, c)]


def scan(root: Path, files: list[tuple[str, str, str]]) -> list[str]:
    """Findings for ``(label, path, blob sha)`` triples; each path and blob is checked once."""
    blobs = read_blobs(root, [sha for _, _, sha in files])
    problems, seen = [], set()
    for label, name, sha in files:
        if (name, sha) in seen:
            continue
        seen.add((name, sha))
        if (why := check_name(name)) is not None:
            problems.append(f"{label}{name}: {why}")
        try:
            text = blobs[sha].decode("utf-8")
        except UnicodeDecodeError:
            continue  # binary
        problems += [f"{label}{name}:{n}: {finding}" for n, finding in check_text(name, text)]
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("."), help="git checkout to scan")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--history", action="store_const", const="history", dest="mode",
                      help="scan every commit reachable from HEAD")
    mode.add_argument("--pre-push", action="store_const", const="pre-push", dest="mode",
                      help="scan the commits named on stdin in git's pre-push format")
    args = parser.parse_args(argv)
    lines = sys.stdin.read().splitlines() if args.mode == "pre-push" else []
    try:
        problems = scan(args.root, files_to_scan(args.root, args.mode or "head", lines))
    except GitError as exc:
        print(f"leak guard: {exc}", file=sys.stderr)
        return 2
    for problem in problems:
        print(problem, file=sys.stderr)
    if problems:
        print(f"leak guard: {len(problems)} possible secret(s) in committed files; remove them "
              "from the commits before pushing", file=sys.stderr)
        return 1
    print("leak guard: clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
