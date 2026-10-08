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

Nothing is sent anywhere unless you opt in. Supply an OpenRouter key in any one of these ways (the first one found wins):

- an environment variable: `export OPENROUTER_API_KEY=...`, or a secrets manager that injects it, e.g. `doppler run -- ledgercheck judge --live --case <case_id>`;
- a `.env` file in the working directory (copy `.env.example`);
- `connections.local.toml` (copy `connections.example.toml`), which also holds the provider, model ids, prices and spend cap.

Both local files are gitignored. Then `ledgercheck connections` shows what was picked up, and `LEDGERCHECK_LLM=1 ledgercheck judge --live --case <case_id>` runs one case live. `ledgercheck connections --help` lists every setting.

Before pushing changes, enable the leak guard once per clone: `git config core.hooksPath .githooks`.

## Docker

Each push to `main` publishes an image to `ghcr.io/eriksjaastad/ledgercheck`. Its default command is `ledgercheck judge`, the offline golden suite, which exits 0 when the suite passes:

```bash
docker run --rm ghcr.io/eriksjaastad/ledgercheck
docker run --rm ghcr.io/eriksjaastad/ledgercheck --help
```

The `Dockerfile` packages the app; `terraform/` is an unapplied Azure reference that nothing provisions.

Behavior lives in `--help` and module docstrings. Current limits: [Known issues](ISSUES.md). License: [MIT](LICENSE) — Copyright (c) 2026 Erik Sjaastad.
