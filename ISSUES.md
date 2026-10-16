# Known issues

Current limits of Ledger Check. Each one is described next to its code, in the docstring or `--help` named.

- **Live LLM mode covers the judge and intake text extraction**, which was never run live. Policy rerank is not implemented; setup and limits: `ledgercheck connections --help`.
- **Approval checks are incomplete.** See "Not checked yet" in `ledgercheck.agents.approval`.
- **`ledgercheck serve` is local only.** Loopback addresses only, no login (`ledgercheck serve --help`).
- **Langfuse tracing is untested against a real server** (`ledgercheck.observability`).
- **The Azure Terraform stack has never been applied.** CI only runs `terraform validate` on it (`python3 scripts/deploy.py --help`).
