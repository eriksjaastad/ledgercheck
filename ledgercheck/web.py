"""ledgercheck serve — a small local web UI for mock runs and corrections.

Pick a fixture case, run the pipeline on it (intake → policy → approval, see
``ledgercheck.agents.approval``) and see the extraction, the policy hits and
the approval outcome with its flags. A run that needs fixing takes a human
correction: one invoice field, a new value, who corrected it and why. The
correction is saved with ``RunStore.apply_correction`` and the run resumes
from policy on the corrected invoice (``resume_run``).

Launch
------
``ledgercheck serve`` and open the address it prints; Ctrl-C stops it.

Options
-------
``--host HOST``
    Bind address, default ``127.0.0.1``. Loopback only: anything outside
    ``127.0.0.0/8`` (``0.0.0.0``, a LAN address, a host name) is refused with
    exit code 2. There is no login, so the UI must not be reachable from
    another machine.
``--port PORT``
    TCP port 0-65535, default 8025; ``0`` picks a free one. A port outside
    that range is a usage error (exit code 2).
``--runs-dir DIR``
    Run-store directory, default ``.scratch/runs`` (gitignored). Every run
    and correction is a JSON file there (``ledgercheck.run_store``).

Offline
-------
Runs use the fixture path: no LLM call and no network. Tracing is
``NullTracer`` unless ``LANGFUSE_PUBLIC_KEY`` and ``LANGFUSE_SECRET_KEY`` are
set (``ledgercheck.connections``); keys set without the SDK fail at start.

Routes
------
``GET /``
    Fixture cases to run, and the stored runs.
``POST /runs``
    Form field ``case_id``; runs that case and redirects to its page.
``GET /runs/<run_id>``
    The run: extraction, policy hits, approval outcome and flags,
    corrections, and the correction form.
``POST /runs/<run_id>/corrections``
    Form fields ``field``, ``value``, ``corrected_by``, ``reason``
    (optional). An empty ``value`` means none (clears an optional field).
    ``line_items`` cannot be corrected here. ``subtotal``, ``tax_amount``,
    ``total`` and ``tax_rate`` must be under 10^12 in magnitude, so the
    pipeline can compute tax to the cent. Redirects to the run page.

Unknown routes are 404. Bad input (unknown case, invalid correction) is a
400 page and nothing is saved. If resuming still fails after a valid
correction is saved, the page is a 500 that says so: the correction is kept,
the run stays ``running`` at its next stage, and submitting another valid
correction resumes it. Form bodies over 16 KiB are refused (413).
Requests are handled in threads, and one lock serializes every store write.
"""

from __future__ import annotations

import argparse
import dataclasses
import html
import ipaddress
import re
import socketserver
import sys
import threading
import traceback
from decimal import Decimal, InvalidOperation
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Mapping, cast
from urllib.parse import parse_qs, urlsplit

from ledgercheck.agents.approval import resume_run, run_pipeline
from ledgercheck.fixtures_loader import FixtureCase, load_cases
from ledgercheck.models import Invoice
from ledgercheck import connections
from ledgercheck.observability import LangfuseUnavailable, Tracer
from ledgercheck.run_store import DEFAULT_ROOT, RunNotFound, RunRecord, RunStore, Stage

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8025
MAX_BODY = 16 * 1024
ROUTES = (
    ("GET", "/"),
    ("POST", "/runs"),
    ("GET", "/runs/<run_id>"),
    ("POST", "/runs/<run_id>/corrections"),
)
CORRECTABLE = tuple(f.name for f in dataclasses.fields(Invoice) if f.name != "line_items")
DECIMAL_FIELDS = tuple(
    f.name for f in dataclasses.fields(Invoice) if str(f.type).startswith("Decimal")
)
# subtotal × tax_rate stays under 10^24, so quantizing it to the cent fits the
# default 28-digit decimal context; larger values raise InvalidOperation.
MAX_AMOUNT = Decimal(10) ** 12
_RUN_PATH = re.compile(r"/runs/([^/]+)(/corrections)?")
_STYLE = (
    "body{font-family:sans-serif;max-width:60em;margin:2em auto}"
    "table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:.2em .5em;text-align:left}"
    ".error{color:#a00}.warning{color:#a60}"
)


class BadRequest(ValueError):
    """An error page with this message: 400 by default, or the given status."""

    def __init__(self, message: str, status: HTTPStatus = HTTPStatus.BAD_REQUEST) -> None:
        super().__init__(message)
        self.status = status


def check_loopback(host: str) -> str:
    """Return ``host`` if it is an IPv4 loopback address, else raise ``ValueError``."""
    try:
        loopback = ipaddress.IPv4Address(host).is_loopback
    except ValueError:
        loopback = False
    if not loopback:
        raise ValueError(f"refusing to bind {host!r}: only 127.0.0.0/8 loopback addresses")
    return host


class LedgerServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], store: RunStore, tracer: Tracer) -> None:
        check_loopback(address[0])
        self.store, self.tracer, self.lock = store, tracer, threading.Lock()
        self.cases: dict[str, FixtureCase] = {c.case_id: c for c in load_cases()}
        super().__init__(address, Handler)

    def server_bind(self) -> None:
        # HTTPServer.server_bind calls socket.getfqdn (a resolver lookup); skip it.
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address[:2]


def make_server(
    store: RunStore, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, tracer: Tracer | None = None
) -> LedgerServer:
    """A bound, not yet serving, server; a non-loopback ``host`` raises ``ValueError``."""
    return LedgerServer((host, port), store, connections.tracer() if tracer is None else tracer)


def check_magnitude(field: str, value: str) -> None:
    """Raise ``BadRequest`` if decimal ``field`` gets a non-decimal or a finite ``value``
    of 10^12 or more. An empty value, NaN and infinity are left to
    ``RunStore.apply_correction``.
    """
    if field not in DECIMAL_FIELDS or not value:
        return
    try:
        number = Decimal(value)
    except InvalidOperation:
        raise BadRequest(f"{field}: not a decimal number: {value!r}") from None
    if number.is_finite() and number.copy_abs() >= MAX_AMOUNT:  # copy_abs cannot overflow
        raise BadRequest(f"{field} must be under 10^12 in magnitude, got {value!r}")


def e(value: Any) -> str:
    return "—" if value is None else html.escape(str(value), quote=True)


def _page(title: str, body: str) -> bytes:
    return (
        f"<!doctype html><html><head><meta charset='utf-8'><title>{e(title)}</title>"
        f"<style>{_STYLE}</style></head><body><p><a href='/'>ledgercheck</a></p>"
        f"<h1>{e(title)}</h1>{body}</body></html>"
    ).encode("utf-8")


def _table(rows: list[Mapping[str, Any]], cols: tuple[str, ...]) -> str:
    if not rows:
        return "<p>none</p>"
    head = "".join(f"<th>{e(c)}</th>" for c in cols)
    body = "".join(
        f"<tr class='{e(r.get('severity', ''))}'>" + "".join(f"<td>{e(r.get(c))}</td>" for c in cols)
        + "</tr>" for r in rows
    )
    return f"<table><tr>{head}</tr>{body}</table>"


def render_index(cases: Mapping[str, FixtureCase], runs: list[RunRecord]) -> bytes:
    options = "".join(
        f"<option value='{e(c.case_id)}'>{e(c.case_id)}: {e(c.description)}</option>"
        for c in cases.values()
    )
    run_rows = "".join(
        f"<tr><td><a href='/runs/{e(r.run_id)}'>{e(r.run_id)}</a></td><td>{e(r.source)}</td>"
        f"<td>{e(r.status.value)}</td><td>{e(r.updated_at)}</td></tr>" for r in reversed(runs)
    )
    return _page("Runs", (
        "<h2>Run a fixture case</h2><form method='post' action='/runs'>"
        f"<select name='case_id'>{options}</select> <button>Run</button></form>"
        "<h2>Stored runs</h2><table><tr><th>run</th><th>case</th><th>status</th><th>updated</th>"
        f"</tr>{run_rows}</table>"
    ))


def render_run(record: RunRecord) -> bytes:
    latest = {s.stage: s.output for s in record.active_steps()}
    inv = dict(record.extracted or {})
    lines = inv.pop("line_items", [])
    fields = "".join(f"<tr><th>{e(k)}</th><td>{e(v)}</td></tr>" for k, v in inv.items())
    hit_cols = ("rule_id", "severity", "field", "expected", "observed", "message")
    policy, decision = latest.get(Stage.POLICY), latest.get(Stage.APPROVAL)
    approval = "<p>not run</p>" if decision is None else (
        f"<p>Outcome: <strong>{e(decision['outcome'])}</strong> "
        f"(decided by {e(decision.get('decided_by'))})</p>"
        + "<ul>" + "".join(f"<li>{e(r)}</li>" for r in decision.get("reasons", ())) + "</ul>"
        + _table(decision.get("hits", []), hit_cols)
    )
    options = "".join(f"<option>{e(f)}</option>" for f in CORRECTABLE)
    return _page(f"Run {record.run_id}", (
        f"<p>Case {e(record.source)} · status <strong>{e(record.status.value)}</strong>"
        f" · next stage {e(record.next_stage)}</p>"
        f"<h2>Extraction</h2><table>{fields}</table>"
        + _table(lines, ("description", "sku", "quantity", "unit_price", "amount"))
        + "<h2>Policy hits</h2>"
        + ("<p>not run</p>" if policy is None else _table(policy.get("hits", []), hit_cols))
        + "<h2>Approval</h2>" + approval
        + "<h2>Corrections</h2>"
        + _table([dataclasses.asdict(c) for c in record.corrections],
                 ("field", "old_value", "new_value", "corrected_by", "reason", "resume_from"))
        + f"<h2>Correct a field</h2><form method='post' action='/runs/{e(record.run_id)}/corrections'>"
        f"<select name='field'>{options}</select> <input name='value' placeholder='new value'> "
        "<input name='corrected_by' placeholder='your name' required> "
        "<input name='reason' placeholder='reason (optional)'> <button>Correct and resume</button></form>"
    ))


class Handler(BaseHTTPRequestHandler):
    timeout = 10  # seconds; a client that stalls mid-body cannot hold a thread forever

    @property
    def app(self) -> LedgerServer:
        return cast(LedgerServer, self.server)

    def do_GET(self) -> None:
        self._handle(self._get)

    def do_POST(self) -> None:
        self._handle(self._post)

    def _handle(self, route) -> None:
        try:
            route()
        except BadRequest as exc:
            self._send(exc.status, _page(f"{exc.status.value} {exc.status.phrase}",
                                         f"<p class='error'>{e(exc)}</p>"))
        except Exception:  # never show a traceback to the client
            self.log_error("%s", connections.mask_lines(traceback.format_exc()))
            self._send(HTTPStatus.INTERNAL_SERVER_ERROR, _page("500 Internal Server Error", ""))

    def _send(self, status: HTTPStatus, body: bytes, location: str | None = None) -> None:
        self.send_response(status)
        if location is not None:
            self.send_header("Location", location)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _not_found(self) -> BadRequest:
        return BadRequest(f"no such page: {self.path}", HTTPStatus.NOT_FOUND)

    def _record(self, run_id: str) -> RunRecord:
        try:
            return self.app.store.get_run(run_id)
        except (RunNotFound, ValueError):  # ValueError: not a valid run id
            raise self._not_found() from None

    def _get(self) -> None:
        path = urlsplit(self.path).path
        match = _RUN_PATH.fullmatch(path)
        if path == "/":
            body = render_index(self.app.cases, self.app.store.list_runs())
        elif match and not match[2]:
            body = render_run(self._record(match[1]))
        else:
            raise self._not_found()
        self._send(HTTPStatus.OK, body)

    def _form(self, *names: str) -> dict[str, str]:
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            raise BadRequest("Content-Length required", HTTPStatus.LENGTH_REQUIRED) from None
        if length < 0 or length > MAX_BODY:
            self.close_connection = True  # the unread body must not be parsed as a request
            raise BadRequest(f"form body over {MAX_BODY} bytes", HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
        try:
            form = parse_qs(self.rfile.read(length).decode("utf-8"), keep_blank_values=True)
        except UnicodeDecodeError:
            raise BadRequest("form body is not UTF-8") from None
        if any(len(v) > 1 for v in form.values()):
            raise BadRequest("repeated form field")
        return {n: form.get(n, [""])[0] for n in names}

    def _post(self) -> None:
        path = urlsplit(self.path).path
        match = _RUN_PATH.fullmatch(path)
        if path == "/runs":
            case = self.app.cases.get(self._form("case_id")["case_id"])
            if case is None:
                raise BadRequest("unknown fixture case")
            with self.app.lock:
                run = run_pipeline(case, store=self.app.store, tracer=self.app.tracer)
            run_id = run.extraction.run_id
        elif match and match[2]:
            run_id = self._record(match[1]).run_id
            form = self._form("field", "value", "corrected_by", "reason")
            if form["field"] not in CORRECTABLE:
                raise BadRequest(f"field must be one of: {', '.join(CORRECTABLE)}")
            check_magnitude(form["field"], form["value"])
            with self.app.lock:
                try:
                    self.app.store.apply_correction(
                        run_id, form["field"], form["value"] or None,
                        corrected_by=form["corrected_by"].strip(), reason=form["reason"],
                    )
                except ValueError as exc:
                    raise BadRequest(str(exc)) from None
                try:
                    resume_run(self.app.store, run_id, tracer=self.app.tracer)
                except Exception as exc:  # the correction is saved; another one retries
                    self.log_error("resume of %s failed: %r", run_id, exc)
                    raise BadRequest(
                        f"correction saved, but resuming the run failed ({type(exc).__name__}); "
                        "the run is left to resume, so submit another correction to retry",
                        HTTPStatus.INTERNAL_SERVER_ERROR,
                    ) from None
        else:
            raise self._not_found()
        self._send(HTTPStatus.SEE_OTHER, b"", location=f"/runs/{run_id}")


def _port(text: str) -> int:
    try:
        port = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not an integer: {text!r}") from None
    if not 0 <= port <= 65535:
        raise argparse.ArgumentTypeError(f"must be 0-65535, got {port}")
    return port


def build_parser(parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    """Add the serve options to ``parser`` (a new one by default); --help shows the docstring."""
    if parser is None:
        parser = argparse.ArgumentParser(prog="python -m ledgercheck.web")
    parser.description = __doc__
    parser.formatter_class = argparse.RawDescriptionHelpFormatter
    parser.add_argument("--host", default=DEFAULT_HOST, help="loopback bind address (127.0.0.0/8)")
    parser.add_argument("--port", type=_port, default=DEFAULT_PORT,
                        help="TCP port 0-65535; 0 picks a free one")
    parser.add_argument("--runs-dir", default=str(DEFAULT_ROOT), metavar="DIR",
                        help="run-store directory")
    return parser


def run(args: argparse.Namespace) -> int:
    """Serve until Ctrl-C (exit 0); a refused host or a bind error exits 2."""
    try:
        server = make_server(connections.run_store(args.runs_dir), args.host, args.port)
    except (ValueError, OverflowError, OSError, LangfuseUnavailable,
            connections.ConnectionConfigError) as exc:
        print(connections.mask(f"serve: {exc}"), file=sys.stderr)
        return 2
    host, port = server.server_address[:2]
    print(f"ledgercheck serving on http://{host}:{port}/ (Ctrl-C to stop)", flush=True)
    with server:
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("ledgercheck serve stopped", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    connections.install_masked_excepthook()
    sys.exit(main())
