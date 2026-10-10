"""Local web UI: a real server on 127.0.0.1 port 0, driven with urllib (offline)."""

import argparse
import inspect
import re
import threading
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pytest

from ledgercheck import cli, web
from ledgercheck.observability import NullTracer
from ledgercheck.run_store import RunStatus, RunStore

# No proxy handler: an http_proxy in the environment must not route the test's requests.
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


@pytest.fixture
def served(tmp_path):
    store = RunStore(tmp_path / "runs")
    server = web.make_server(store, "127.0.0.1", 0, tracer=NullTracer())
    thread = threading.Thread(target=server.serve_forever, args=(0.05,), daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}", store
    server.shutdown()
    server.server_close()
    thread.join(timeout=10)


def request(url, form=None, *, data=None):
    """(status, final url, body); ``form`` is urlencoded into a POST."""
    if form is not None:
        data = urllib.parse.urlencode(form).encode()
    try:
        with OPENER.open(urllib.request.Request(url, data=data), timeout=10) as resp:
            return resp.status, resp.url, resp.read().decode()
    except urllib.error.HTTPError as err:
        with err:
            return err.code, url, err.read().decode()


def test_mock_run_flags_correction_and_resume(served):
    base, store = served
    status, _, page = request(base + "/")
    assert status == 200 and "tax_mismatch" in page and "clean_baseline" in page
    status, url, page = request(base + "/runs", {"case_id": "tax_mismatch"})
    assert status == 200 and re.fullmatch(re.escape(base) + r"/runs/run-[0-9a-f]+", url)
    run_id = url.rsplit("/", 1)[1]
    assert "needs_human" in page and "APR-TAX-AMOUNT" in page and "POL-TAX-AMOUNT" in page
    assert run_id in request(base + "/")[2]

    form = {"field": "tax_amount", "value": "435.90", "corrected_by": "reviewer", "reason": "typo"}
    status, url, page = request(f"{url}/corrections", form)
    assert status == 200 and url == f"{base}/runs/{run_id}"
    assert "approve" in page and "435.90" in page and "typo" in page
    record = store.get_run(run_id)
    assert record.status is RunStatus.COMPLETED
    (correction,) = record.corrections
    assert (correction.field, correction.new_value, correction.corrected_by) == (
        "tax_amount", "435.90", "reviewer"
    )
    assert record.steps[-1].output["outcome"] == "approve"


def test_hostile_values_are_escaped(served):
    base, store = served
    url = request(base + "/runs", {"case_id": "clean_baseline"})[1]
    evil = "<script>alert(1)</script>"
    form = {"field": "vendor_name", "value": evil, "corrected_by": f"'\"{evil}", "reason": evil}
    status, _, page = request(f"{url}/corrections", form)
    assert status == 200
    assert "<script>" not in page and "&lt;script&gt;alert(1)&lt;/script&gt;" in page
    assert "&#x27;&quot;&lt;script" in page
    assert store.get_run(url.rsplit("/", 1)[1]).corrections[0].reason == evil  # stored raw


@pytest.mark.parametrize("path", ["/nope", "/runs/run-none", "/runs/..%2Fx", "/runs/a/b"])
def test_unknown_get_routes_404(served, path):
    status, _, page = request(served[0] + path)
    assert status == 404 and "Traceback" not in page


def test_unknown_post_routes_404(served):
    base, _ = served
    assert request(base + "/", {"x": "1"})[0] == 404
    assert request(base + "/runs/run-none/corrections", {"field": "total"})[0] == 404


@pytest.mark.parametrize(
    "form, message",
    [
        ({"field": "subtotal", "value": "abc"}, "not a decimal number"),
        ({"field": "invoice_date", "value": ""}, "invalid"),
        ({"field": "line_items", "value": "[]"}, "field must be one of"),
        ({"field": "total", "value": "1", "corrected_by": "  "}, "corrected_by"),
        # Valid decimals the policy tax quantize cannot handle (InvalidOperation).
        ({"field": "tax_rate", "value": "-1E+999999999"}, "under 10^12"),
        ({"field": "total", "value": "1000000000000"}, "under 10^12"),
    ],
)
def test_invalid_correction_is_400_and_saves_nothing(served, form, message):
    base, store = served
    url = request(base + "/runs", {"case_id": "missing_po"})[1]
    before = store.get_run(url.rsplit("/", 1)[1])
    status, _, page = request(f"{url}/corrections", {"corrected_by": "reviewer", **form})
    assert status == 400 and message in page
    assert store.get_run(before.run_id) == before


def test_magnitude_bound_matches_decimal_fields():
    assert web.DECIMAL_FIELDS == ("subtotal", "tax_amount", "total", "tax_rate")
    web.check_magnitude("subtotal", "999999999999.99")  # just under the bound
    web.check_magnitude("vendor_name", "1e30")  # not a decimal field
    with pytest.raises(web.BadRequest, match="under 10"):
        web.check_magnitude("tax_amount", "-1e12")


def test_failed_resume_keeps_correction_and_another_recovers(served, monkeypatch):
    base, store = served
    url = request(base + "/runs", {"case_id": "tax_mismatch"})[1]
    run_id = url.rsplit("/", 1)[1]

    def broken(*args, **kwargs):
        raise ArithmeticError("boom")

    monkeypatch.setattr(web, "resume_run", broken)
    form = {"field": "tax_amount", "value": "435.90", "corrected_by": "reviewer"}
    status, _, page = request(f"{url}/corrections", form)
    assert status == 500 and "correction saved" in page and "Traceback" not in page
    record = store.get_run(run_id)
    assert (record.status, record.next_stage) == (RunStatus.RUNNING, "policy")
    assert len(record.corrections) == 1
    monkeypatch.undo()
    status, _, page = request(f"{url}/corrections", form)
    assert status == 200 and "approve" in page
    assert store.get_run(run_id).status is RunStatus.COMPLETED
    assert "the run stays ``running`` at its next stage" in web.__doc__


def test_failed_resume_log_masks_a_key(served, monkeypatch, capsys):
    base, _ = served
    url = request(base + "/runs", {"case_id": "tax_mismatch"})[1]
    fake = "sk-or-v1-" + "f" * 32
    monkeypatch.setenv("OPENROUTER_API_KEY", fake)

    def broken(*args, **kwargs):
        raise RuntimeError(f"auth failed with {fake}")

    monkeypatch.setattr(web, "resume_run", broken)
    form = {"field": "tax_amount", "value": "435.90", "corrected_by": "reviewer"}
    assert request(f"{url}/corrections", form)[0] == 500
    err = capsys.readouterr().err
    assert "resume of" in err and fake not in err


def test_bad_run_requests_are_4xx(served):
    base, store = served
    assert request(base + "/runs", {"case_id": "<b>nope</b>"})[0] == 400
    assert request(base + "/runs", data=b"case_id=a&case_id=b")[0] == 400
    assert request(base + "/runs", data=b"case_id=\xff")[0] == 400
    assert request(base + "/runs", data=b"x" * (web.MAX_BODY + 1))[0] == 413
    assert store.list_runs() == []


@pytest.mark.parametrize("host", ["0.0.0.0", "::1"])  # not loopback; not an IPv4 address
def test_non_loopback_bind_is_refused(tmp_path, host, capsys):
    with pytest.raises(ValueError, match="loopback"):
        web.make_server(RunStore(tmp_path), host, 0, tracer=NullTracer())
    assert cli.main(["serve", "--host", host, "--port", "0"]) == 2
    assert "refusing to bind" in capsys.readouterr().err


@pytest.mark.parametrize("port", ["65536", "x"])
def test_out_of_range_port_exits_2(tmp_path, port, capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["serve", "--port", port, "--runs-dir", str(tmp_path)])
    err = capsys.readouterr().err
    assert exit_info.value.code == 2 and "--port" in err and "Traceback" not in err


def test_bind_overflow_exits_2(tmp_path, capsys):
    # Past the argparse check (a direct run() call), socket bind's OverflowError still exits 2.
    args = argparse.Namespace(host="127.0.0.1", port=70000, runs_dir=str(tmp_path))
    assert web.run(args) == 2
    assert "serve:" in capsys.readouterr().err


def test_docs_match_routes_and_options(capsys):
    doc = web.__doc__
    assert re.findall(r"^``(GET|POST) (\S+)``$", doc, re.M) == list(web.ROUTES)
    # The handler dispatches on exactly the documented fixed paths and run-id pattern.
    source = inspect.getsource(web.Handler)
    assert 'path == "/"' in source and 'path == "/runs"' in source
    assert web._RUN_PATH.pattern == r"/runs/([^/]+)(/corrections)?"
    options = {o for a in web.build_parser()._actions for o in a.option_strings}
    assert options == {"-h", "--help", "--host", "--port", "--runs-dir"}
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["serve", "--help"])
    help_text = capsys.readouterr().out
    assert exit_info.value.code == 0 and help_text.startswith("usage: ledgercheck serve")
    for option in options - {"-h", "--help"}:
        assert f"``{option} " in doc and option in help_text
    for fact in ("127.0.0.1", "loopback", "NullTracer", "--runs-dir", f"default {web.DEFAULT_PORT}",
                 f"{web.MAX_BODY // 1024} KiB", "0-65535", "under 10^12"):
        assert fact in help_text
    assert "``serve``" in cli.__doc__ and "ledgercheck serve --help" in cli.__doc__


def test_readme_has_no_local_urls():
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
    assert "localhost" not in readme and "127.0.0.1" not in readme
