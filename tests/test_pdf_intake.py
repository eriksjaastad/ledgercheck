"""PDF intake: the text layer read by pypdf and ``ledgercheck run-pdf``, offline."""

import io
import json
import re

import pytest
from pypdf import PdfWriter
from test_intake_agent import FakeClient

from ledgercheck import run_pdf
from ledgercheck.agents import IntakeAgent
from ledgercheck.fixtures_loader import FIXTURES_DIR, load_case
from ledgercheck.models import to_jsonable
from ledgercheck.pdf_text import PdfTextError, read_pdf_text

CASE = load_case(FIXTURES_DIR / "european_format_text.json")
TEXT = CASE.raw_text or ""  # non-ASCII text: €, ä, ·


def text_pdf(lines: list[str]) -> bytes:
    """A one-page PDF showing ``lines`` in Helvetica (WinAnsi), with a correct xref."""
    shown = (line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)") for line in lines)
    body = b"BT /F1 9 Tf 10 TL 36 800 Td " + b" ".join(
        b"(" + line.encode("cp1252") + b") '" for line in shown) + b" ET"
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 842] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
        b"<< /Length %d >>\nstream\n%s\nendstream" % (len(body), body),
    ]
    out, offsets = bytearray(b"%PDF-1.4\n"), []
    for number, obj in enumerate(objects, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n%s\nendobj\n" % (number, obj)
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % offset for offset in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1, xref)
    return bytes(out)


def blank_pdf(password: str | None = None) -> bytes:
    """A one-page PDF with no text, encrypted (RC4) when ``password`` is given."""
    writer = PdfWriter()
    writer.add_blank_page(612, 842)
    if password is not None:
        writer.encrypt(password)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def words(text: str) -> list[list[str]]:
    return [line.split() for line in text.splitlines() if line.strip()]


@pytest.fixture
def invoice_pdf(tmp_path):
    path = tmp_path / "docs" / "invoice.pdf"
    path.parent.mkdir()
    path.write_bytes(text_pdf(TEXT.splitlines()))
    return path


def run(argv, intake=None):
    return run_pdf.run(run_pdf.build_parser().parse_args(argv), intake=intake)


def test_read_pdf_text_round_trips_a_text_fixture(invoice_pdf):
    assert words(read_pdf_text(invoice_pdf)) == words(TEXT)


@pytest.mark.parametrize("content, message", [
    (blank_pdf(), "page 1 has no text layer (probably a scan); OCR is not supported"),
    (blank_pdf(password="secret"), "encrypted PDF"),
    (b"Invoice INV-1 total 10.00", "not a readable PDF"),
    (None, "cannot read the file"),
], ids=["blank", "encrypted", "not-a-pdf", "missing"])
def test_read_pdf_text_refuses_with_the_file_name_only(tmp_path, content, message):
    path = tmp_path / "invoice.pdf"
    if content is not None:
        path.write_bytes(content)
    with pytest.raises(PdfTextError, match=re.escape(message)) as refused:
        read_pdf_text(path)
    assert str(refused.value).startswith("invoice.pdf: ") and str(tmp_path) not in str(refused.value)


def test_read_pdf_text_refuses_one_scanned_page_among_text_pages(tmp_path):
    writer = PdfWriter(clone_from=io.BytesIO(text_pdf(TEXT.splitlines())))
    writer.add_blank_page(612, 842)
    path = tmp_path / "invoice.pdf"
    writer.write(path)
    with pytest.raises(PdfTextError, match="invoice.pdf: page 2 has no text layer"):
        read_pdf_text(path)


def test_text_only_prints_the_text_layer_and_stores_nothing(invoice_pdf, tmp_path, capsys):
    assert run(["--text-only", "--runs-dir", str(tmp_path / "runs"), str(invoice_pdf)]) == 0
    assert words(capsys.readouterr().out) == words(TEXT)
    assert not (tmp_path / "runs").exists()


def test_default_run_with_the_gate_off_refuses_in_one_line(invoice_pdf, tmp_path, capsys):
    assert run(["--runs-dir", str(tmp_path / "runs"), str(invoice_pdf)]) == 2
    out, err = capsys.readouterr()
    assert out == "" and err.count("\n") == 1
    assert err.startswith("ledgercheck run-pdf: live LLM calls are off") and "--text-only" in err
    assert not (tmp_path / "runs").exists()


def test_run_stores_the_decision_and_flags_a_second_run_as_duplicate(invoice_pdf, tmp_path,
                                                                     capsys):
    runs = tmp_path / "runs"
    client = FakeClient(to_jsonable(CASE.invoice))
    argv = ["--runs-dir", str(runs), str(invoice_pdf)]
    assert run(argv, IntakeAgent(llm_client=client)) == 0
    first = capsys.readouterr().out
    assert first.startswith("run run-") and "APR-DUPLICATE" not in first
    [stored] = [json.loads(p.read_text()) for p in runs.glob("*.json")]
    assert stored["source"] == "invoice.pdf" and words(client.texts[0]) == words(TEXT)

    assert run(argv, IntakeAgent(llm_client=client)) == 0
    second = capsys.readouterr().out.splitlines()
    assert second[0].endswith(": needs_human") and any(
        line.startswith("APR-DUPLICATE error: ") for line in second[1:])


def test_help_documents_the_text_layer_ocr_and_the_offline_option():
    help_text = " ".join(run_pdf.build_parser().format_help().split())
    for fact in ("text layer", "OCR is not supported", "--text-only", "LEDGERCHECK_LLM=1",
                 "--runs-dir", "APR-DUPLICATE", "Exit codes"):
        assert fact in help_text, fact
