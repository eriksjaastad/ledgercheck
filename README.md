# Ledger Check

**AI-driven B2B invoice and payment reconciliation.** Multi-agent intake, policy check, and approval flags — with golden-dataset evals and observability hooks.

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

Behavior lives in `--help` and module docstrings. Current limits: [Known issues](ISSUES.md). License: [MIT](LICENSE) — Copyright (c) 2026 Erik Sjaastad.
