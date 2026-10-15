"""Ledger Check — invoice-reconciliation pipeline with a golden-dataset eval gate.

Intake, policy check and approval flags run offline on the bundled sample
invoices. The live LLM paths are opt-in: the judge (``ledgercheck judge --live``)
and intake text extraction (``IntakeAgent.extract_text``, no command calls it).
"""

__version__ = "0.1.0"
