"""Read the text layer of a PDF invoice, locally, with pypdf.

``read_pdf_text(path)`` returns the text pypdf extracts from every page, pages
joined by a newline. Only a text layer is read: there is no OCR, so a PDF with
any page that has no text (a scan) is refused, and so is an encrypted PDF (it
is never decrypted). Every refusal is a ``PdfTextError`` whose message names
the file by its name only, never its directory or any of its contents.
"""

from __future__ import annotations

import os
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError


class PdfTextError(ValueError):
    """A PDF whose text layer cannot be read; the message names the file only."""


def read_pdf_text(path: str | os.PathLike[str]) -> str:
    """The text layer of every page of the PDF at ``path``, pages joined by a newline."""
    pdf = Path(path)
    name = pdf.name
    try:
        reader = PdfReader(pdf)
        if reader.is_encrypted:
            raise PdfTextError(f"{name}: encrypted PDF; it is not decrypted, remove the password")
        pages = [page.extract_text() for page in reader.pages]
    except OSError as exc:
        raise PdfTextError(f"{name}: cannot read the file ({exc.strerror})") from None
    except PdfReadError:
        raise PdfTextError(f"{name}: not a readable PDF") from None
    for number, text in enumerate(pages, 1):
        if not text.strip():
            raise PdfTextError(
                f"{name}: page {number} has no text layer (probably a scan); OCR is not supported")
    return "\n".join(pages)
