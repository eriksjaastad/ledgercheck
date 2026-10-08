#!/usr/bin/env python3
"""Deploy helper for Ledger Check. It never deploys.

The Dockerfile and the Azure Container Apps stack in ``terraform/`` are
a write-only reference: nothing in this repo applies them. This script only
checks them.

Commands
--------
``validate``
    If ``terraform`` is on PATH, runs ``terraform init -backend=false
    -input=false`` and ``terraform validate`` in ``terraform/``. Init downloads
    the azurerm provider; neither step logs in to Azure, touches state or
    creates anything. Without the binary it prints these instructions and exits
    0, so it is safe on a machine without Terraform. Each step is killed after
    ``STEP_TIMEOUT_SECONDS`` (600s); a timeout prints a message and exits 124,
    so a stalled provider download fails instead of hanging.
``apply``, ``destroy``
    Refused: prints that this stack is a write-only reference and exits 2.
    Nothing runs. Run terraform yourself if you mean to deploy.

Image
-----
Build and run the image with Docker on your own machine (see the Dockerfile
header)::

    docker build -t ledgercheck:dev .
    docker run --rm ledgercheck:dev

Usage: ``python3 scripts/deploy.py {validate,apply,destroy}``.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, Sequence

TERRAFORM_DIR = Path(__file__).resolve().parent.parent / "terraform"
VALIDATE_STEPS = (
    ("terraform", "init", "-backend=false", "-input=false"),
    ("terraform", "validate"),
)
STEP_TIMEOUT_SECONDS = 600
TIMEOUT_EXIT = 124  # same code as coreutils timeout(1)
REFUSED = ("apply", "destroy")
FORBIDDEN = (
    "terraform {cmd} is refused: this stack is a write-only reference;"
    " run terraform yourself if you mean to deploy."
)

Runner = Callable[[Sequence[str], Path], int]


def _run(cmd: Sequence[str], cwd: Path) -> int:
    try:
        return subprocess.run(
            list(cmd), cwd=cwd, check=False, timeout=STEP_TIMEOUT_SECONDS
        ).returncode
    except subprocess.TimeoutExpired:
        print(
            f"{' '.join(cmd)} timed out after {STEP_TIMEOUT_SECONDS}s; giving up.",
            file=sys.stderr,
        )
        return TIMEOUT_EXIT


def validate(
    runner: Runner = _run, which: Callable[[str], str | None] = shutil.which
) -> int:
    """Run the validate steps, or print them and return 0 without terraform."""
    if which("terraform") is None:
        print("terraform not found; skipping. To validate, install Terraform and run in terraform/:")
        for step in VALIDATE_STEPS:
            print("    " + " ".join(step))
        return 0
    for step in VALIDATE_STEPS:
        code = runner(step, TERRAFORM_DIR)
        if code != 0:
            return code
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="deploy.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("command", choices=("validate", *REFUSED))
    return parser


def main(argv: list[str] | None = None, runner: Runner = _run,
         which: Callable[[str], str | None] = shutil.which) -> int:
    args = build_parser().parse_args(argv)
    if args.command in REFUSED:
        print(FORBIDDEN.format(cmd=args.command), file=sys.stderr)
        return 2
    return validate(runner, which)


if __name__ == "__main__":
    sys.exit(main())
