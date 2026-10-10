"""Ledger Check — invoice-reconciliation pipeline with a golden-dataset eval gate.

Intake, policy check and approval flags run offline on the bundled sample
invoices; the only live LLM path is the opt-in judge (``ledgercheck judge --live``).
"""

__version__ = "0.1.0"
