# Known issues

Current limits of Ledger Check. Each one is described next to its code, in the docstring or `--help` named.

- **Runs on the bundled synthetic invoices only.** No command reads your own invoices, and there is no PDF input (`ledgercheck.agents.intake`).
- **Live LLM mode covers the judge only.** Live invoice extraction and policy rerank are not implemented; setup and limits: `ledgercheck connections --help`.
- **Approval checks are incomplete.** See "Not checked yet" in `ledgercheck.agents.approval`.
- **Padded vendor ids are not matched.** See `PolicyCorpus.find_vendor` in `ledgercheck.agents.policy`.
- **`ledgercheck serve` is local only.** Loopback addresses only, no login (`ledgercheck serve --help`).
- **Langfuse tracing is untested against a real server** (`ledgercheck.observability`).
- **The Azure Terraform stack is an untested reference** (`python3 scripts/deploy.py --help`).
