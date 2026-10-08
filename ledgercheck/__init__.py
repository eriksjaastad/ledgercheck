"""Ledger Check — invoice-reconciliation pipeline with a golden-dataset eval gate.

Intake, policy check and approval flags run offline on the bundled sample
invoices; the live LLM path is not built yet.
"""

__version__ = "0.1.0"
