"""One canary test: fake secrets in every config source never reach stdout, stderr, logs or a
written report, on any command's success or error path (all offline)."""

import http.client
import io
import json
import sys
import threading
import urllib.error
import urllib.parse

import pytest

from ledgercheck import beta_notes, cli, connections, web
from ledgercheck.golden import load_golden_cases

ENV_KEY = "sk-or-v1-CANARY-env-q8W3zT5mL1xR9pVb"  # OPENROUTER_API_KEY in the environment
DOTENV_KEY = "sk-lf-CANARY-dotenv-Hj4Nc7Yw2Ks6Gd"  # LANGFUSE_SECRET_KEY in .env
FILE_KEY = "sk-or-v1-CANARY-file-Pz5Qe8Rt1Uy3Io"  # api_key in the connections file
SHORT_PW = "Kq7Zx"  # a short password in LANGFUSE_HOST in .env
CANARIES = (ENV_KEY, DOTENV_KEY, FILE_KEY)
MODEL = "vendor/large-y"
CASE = load_golden_cases()[0].case_id
LIVE = {"LEDGERCHECK_LLM": "1", "LEDGERCHECK_LLM_PROVIDER": "openrouter",
        "LEDGERCHECK_MODEL_LARGE": MODEL, "LEDGERCHECK_SPEND_CAP_USD": "0.05"}
ECHO = f"keys {ENV_KEY} {DOTENV_KEY} {FILE_KEY} {SHORT_PW}"  # a provider or model quoting them
REPLY = {"choices": [{"message": {"content": json.dumps(
    {"accuracy": 5, "hallucination": 5, "formatting": 5, "notes": [ECHO]})}}],
    "usage": {"prompt_tokens": 9, "completion_tokens": 9, "cost": 0.001}}
DENIED = urllib.error.HTTPError("https://x", 401, "denied", {}, io.BytesIO(ECHO.encode()))


def _opener(reply):
    """A stand-in for ``urlopen`` that answers every request with ``reply``."""
    def urlopen(request, timeout):
        if isinstance(reply, Exception):
            raise reply
        return io.BytesIO(json.dumps(reply).encode())
    return lambda: urlopen


def _fail(*args, **kwargs):
    raise RuntimeError(f"unexpected failure: {ECHO}")


def _request(server, method, path, form=None):
    """One request to ``server``; the response's Location header."""
    conn = http.client.HTTPConnection(*server.server_address[:2], timeout=10)
    body = urllib.parse.urlencode(form) if form else None
    conn.request(method, path, body, {"Content-Type": "application/x-www-form-urlencoded"})
    response = conn.getresponse()
    response.read()
    conn.close()
    return response.getheader("Location")


def _serve_three_requests(server):
    """Start a run, correct it (its resume fails, which is logged), and GET a path holding a
    key, then stop as Ctrl-C would."""
    def client():
        run = _request(server, "POST", "/runs", {"case_id": "tax_mismatch"})
        _request(server, "POST", f"{run}/corrections",
                 {"field": "tax_amount", "value": "435.90", "corrected_by": "reviewer"})
        _request(server, "GET", f"/runs/{ENV_KEY}")
    thread = threading.Thread(target=client)
    thread.start()
    for _ in range(3):
        server.handle_request()
    thread.join(timeout=10)
    raise KeyboardInterrupt


# id: (argv, extra environment, patches, expected exit code)
CASES = {
    "connections ok": (["connections"], {}, (), 0),
    "connections pasted keys": (["connections"], {
        **LIVE, "LEDGERCHECK_MODEL_LARGE": DOTENV_KEY, "LEDGERCHECK_MODEL_SMALL": FILE_KEY,
        "LEDGERCHECK_LLM_BASE_URL": f"https://{ENV_KEY}/v/{SHORT_PW}"}, (), 2),
    "judge ok": (["judge", "--case", CASE, "--json", "{tmp}/r.json"], {}, (), 0),
    "judge unknown case": (["judge", "--case", ENV_KEY], {}, (), 2),
    "judge live ok": (["judge", "--live", "--case", CASE, "--json", "{tmp}/r.json"], LIVE,
                      ((connections, "_default_opener", _opener(REPLY)),), 0),
    "judge live provider error": (["judge", "--live", "--case", CASE], LIVE,
                                  ((connections, "_default_opener", _opener(DENIED)),), 2),
    "serve with a failing resume": (["serve", "--port", "0", "--runs-dir", "{tmp}/runs"], {},
                                    ((web.LedgerServer, "serve_forever", _serve_three_requests),
                                     (web, "resume_run", _fail)), 0),
    "serve refused host": (["serve", "--host", ENV_KEY, "--runs-dir", "{tmp}/runs"], {}, (), 2),
    "beta-notes ok": (["beta-notes"], {}, (), 0),
    "beta-notes bad argument": (["beta-notes", f"--key={ENV_KEY}"], {}, (), 2),
    "unexpected crash": (["beta-notes"], {}, ((beta_notes, "template", _fail),), 1),
}


@pytest.mark.parametrize("argv, env, patches, code", CASES.values(), ids=CASES.keys())
def test_no_secret_reaches_any_output(argv, env, patches, code, tmp_path, monkeypatch, capsys,
                                      caplog):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(f"LANGFUSE_SECRET_KEY={DOTENV_KEY}\n"
                                   f"LANGFUSE_HOST=https://lf:{SHORT_PW}@lf.invalid\n")
    (tmp_path / "c.toml").write_text(
        f'api_key = "{FILE_KEY}"\n[prices."{MODEL}"]\nprompt = 1.0\ncompletion = 2.0\n')
    monkeypatch.setattr(connections, "DOTENV_PATH", tmp_path / ".env")
    for name, value in {"OPENROUTER_API_KEY": ENV_KEY,
                        "LEDGERCHECK_CONNECTIONS_FILE": str(tmp_path / "c.toml"), **env}.items():
        monkeypatch.setenv(name, value)
    for patch in patches:
        monkeypatch.setattr(*patch)
    monkeypatch.setattr(sys, "excepthook", sys.excepthook)  # the interpreter's own hook
    try:
        exit_code = cli.main([arg.format(tmp=tmp_path) for arg in argv])
    except SystemExit as stop:  # argparse errors
        exit_code = stop.code
    except RuntimeError as crash:  # what the interpreter does with an uncaught crash
        sys.excepthook(type(crash), crash, crash.__traceback__)
        exit_code = 1
    out, err = capsys.readouterr()
    report = tmp_path / "r.json"
    shown = [out, err, caplog.text, report.read_text() if report.is_file() else ""]
    assert exit_code == code, err
    for text in shown:  # no 6-character piece of a key, nor the short password as a word
        pieces = {c[i:i + 6] for c in CANARIES for i in range(len(c) - 5)} | {SHORT_PW}
        assert not [p for p in pieces if p in text], text
