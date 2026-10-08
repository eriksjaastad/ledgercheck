# Ledger Check

[![CI](https://github.com/eriksjaastad/ledgercheck/actions/workflows/ci.yml/badge.svg)](https://github.com/eriksjaastad/ledgercheck/actions/workflows/ci.yml)
[![Image](https://github.com/eriksjaastad/ledgercheck/actions/workflows/image.yml/badge.svg)](https://github.com/eriksjaastad/ledgercheck/pkgs/container/ledgercheck)

An invoice-reconciliation pipeline (intake, policy check, approval flags) with a golden-dataset eval gate. By default it runs offline and deterministically on the bundled sample invoices; an LLM judge is opt-in with your own key and a hard spend cap.

Not a full ERP. Not a payments processor. Not a hosted SaaS you sign up for here.

## Install

Requires **Python 3.11+**.

```bash
git clone https://github.com/eriksjaastad/ledgercheck.git
cd ledgercheck
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -U pip
pip install -e ".[dev]"
ledgercheck --help
pytest
```

## Connecting a model (optional)

To connect a model with your own key, or to enable the leak-guard pre-push hook, see `ledgercheck connections --help`.

## Docker

Each push to `main` publishes an image to `ghcr.io/eriksjaastad/ledgercheck`. Its default command is `ledgercheck judge`, the offline golden suite, which exits 0 when the suite passes:

```bash
docker run --rm ghcr.io/eriksjaastad/ledgercheck
docker run --rm ghcr.io/eriksjaastad/ledgercheck --help
```

The `Dockerfile` packages the app; `terraform/` is an unapplied Azure reference that nothing provisions.

Behavior lives in `--help` and module docstrings. Current limits: [Known issues](ISSUES.md). License: [MIT](LICENSE) — Copyright (c) 2026 Erik Sjaastad.
