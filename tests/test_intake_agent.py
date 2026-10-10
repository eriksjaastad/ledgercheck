"""IntakeAgent: fixture extraction, the live-LLM gate, and RunStore recording."""

import socket
from decimal import Decimal

import pytest

from ledgercheck.agents import IntakeAgent, record_intake
from ledgercheck.agents.llm_client import (
    API_KEY_ENV,
    ENV_FLAG,
    LiveLLMDisabled,
    LLMClient,
)
from ledgercheck.fixtures_loader import FIXTURES_DIR, load_case, load_cases
from ledgercheck.models import ExtractionResult, to_jsonable
from ledgercheck.run_store import RunStatus, RunStore, Stage


@pytest.fixture
def agent():
    return IntakeAgent()


@pytest.fixture
def store(tmp_path):
    return RunStore(tmp_path / "runs")


@pytest.mark.parametrize("case_id", ["clean_baseline", "gbp_cloud_services", "ocr_noise_text"])
def test_extract_returns_fixture_invoice(agent, case_id):
    case = load_case(FIXTURES_DIR / f"{case_id}.json")
    result = agent.extract(case_id, run_id="run-a")
    assert isinstance(result, ExtractionResult)
    assert result.run_id == "run-a"
    assert result.source == case_id
    assert result.extractor == "fixture"
    assert result.invoice == case.invoice
    assert dict(result.field_confidence) == {}
    assert result.warnings == ()


def test_text_case_uses_paired_json_as_extraction(agent):
    text_cases = [c for c in load_cases() if c.source_format == "text"]
    assert len(text_cases) >= 2
    for case in text_cases:
        result = agent.extract(case)
        assert result.invoice == case.invoice
        assert result.source == case.case_id


def test_extract_accepts_fixture_case_and_generates_unique_run_ids(agent):
    case = load_case(FIXTURES_DIR / "credit_note.json")
    first, second = agent.extract(case), agent.extract(case)
    assert first.run_id != second.run_id
    assert first.invoice.total < Decimal("0")


def test_extract_rejects_empty_run_id(agent):
    with pytest.raises(ValueError, match="run_id"):
        agent.extract("clean_baseline", run_id="")


def test_extract_unknown_case_raises(agent):
    with pytest.raises(FileNotFoundError, match="no_such_case"):
        agent.extract("no_such_case")


@pytest.mark.parametrize("case_id", ["CLEAN_BASELINE", "Clean_Baseline"])
def test_extract_mis_cased_id_raises_file_not_found(agent, case_id):
    with pytest.raises(FileNotFoundError, match=case_id):
        agent.extract(case_id)


def test_extract_missing_fixtures_root_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError, match="clean_baseline"):
        IntakeAgent(fixtures_root=tmp_path / "absent").extract("clean_baseline")


@pytest.mark.parametrize(
    "case_id",
    [
        "",
        ".",
        "..",
        "./clean_baseline",
        ".//clean_baseline",
        "../clean_baseline",
        "a/b",
        "invoices/clean_baseline",
        "/abs",
    ],
)
def test_extract_rejects_non_stem_case_id(agent, case_id):
    with pytest.raises(ValueError, match="not a fixture case id"):
        agent.extract(case_id)


def test_extract_accepts_bare_stem_case_id(agent):
    result = agent.extract("clean_baseline")
    assert result.source == "clean_baseline"
    assert result.invoice == load_case(FIXTURES_DIR / "clean_baseline.json").invoice


def test_extract_uses_custom_fixtures_root(tmp_path):
    src = FIXTURES_DIR / "rounding.json"
    (tmp_path / "rounding.json").write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    result = IntakeAgent(fixtures_root=tmp_path).extract("rounding")
    assert result.invoice == load_case(src).invoice
    with pytest.raises(FileNotFoundError):
        IntakeAgent(fixtures_root=tmp_path).extract("clean_baseline")


# --- live path: blocked unless explicitly enabled -------------------------


def test_live_client_off_by_default(monkeypatch):
    monkeypatch.delenv(ENV_FLAG, raising=False)
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    with pytest.raises(LiveLLMDisabled, match=ENV_FLAG):
        LLMClient()


@pytest.mark.parametrize(
    "env",
    [
        {},
        {API_KEY_ENV: "sk-test"},  # key alone is not consent
        {ENV_FLAG: "true", API_KEY_ENV: "sk-test"},  # only "1" counts
        {ENV_FLAG: "0", API_KEY_ENV: "sk-test"},
        {ENV_FLAG: "1"},  # flag without a key
        {ENV_FLAG: "1", API_KEY_ENV: "  "},
    ],
)
def test_live_client_refuses_without_flag_and_key(env):
    with pytest.raises(LiveLLMDisabled) as exc:
        LLMClient(env)
    assert "sk-test" not in str(exc.value)


def test_live_client_unset_key_message():
    with pytest.raises(LiveLLMDisabled, match=f"{API_KEY_ENV} is not set"):
        LLMClient({ENV_FLAG: "1"})


@pytest.mark.parametrize("key", ["", "  ", "\t\n"])
def test_live_client_blank_key_message(key):
    with pytest.raises(LiveLLMDisabled, match=f"{API_KEY_ENV} is blank"):
        LLMClient({ENV_FLAG: "1", API_KEY_ENV: key})


@pytest.fixture
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        pytest.fail("network call attempted")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)


def test_enabled_client_constructs_and_makes_no_request(no_network):
    env = {ENV_FLAG: "1", API_KEY_ENV: "fake-not-a-real-key"}
    client = LLMClient(env)
    assert client.name == "openrouter"
    with pytest.raises(NotImplementedError):
        client.extract_invoice("Invoice INV-1 total 10.00")


def test_extract_text_with_enabled_client_makes_no_request(no_network):
    client = LLMClient({ENV_FLAG: "1", API_KEY_ENV: "fake-not-a-real-key"})
    with pytest.raises(NotImplementedError):
        IntakeAgent(llm_client=client).extract_text("Invoice INV-1", source="doc")


def test_extract_text_blocked_by_default(agent, monkeypatch):
    monkeypatch.delenv(ENV_FLAG, raising=False)
    case = load_case(FIXTURES_DIR / "ocr_noise_text.json")
    with pytest.raises(LiveLLMDisabled):
        agent.extract_text(case.raw_text, source=case.case_id)


class FakeClient:
    name = "fake-llm"

    def __init__(self, fields):
        self.fields = fields
        self.texts = []

    def extract_invoice(self, text):
        self.texts.append(text)
        return self.fields


def test_extract_text_with_injected_client():
    case = load_case(FIXTURES_DIR / "european_format_text.json")
    client = FakeClient(to_jsonable(case.invoice))
    result = IntakeAgent(llm_client=client).extract_text(
        case.raw_text, source=case.case_id, run_id="run-t"
    )
    assert client.texts == [case.raw_text]
    assert result.extractor == "fake-llm"
    assert result.invoice == case.invoice
    assert result.run_id == "run-t"


def test_extract_text_rejects_invalid_client_output():
    client = FakeClient({"invoice_number": "X"})
    with pytest.raises(ValueError, match="missing keys"):
        IntakeAgent(llm_client=client).extract_text("text", source="doc")


# --- RunStore integration -------------------------------------------------


def test_record_intake_starts_run_and_sets_extracted(agent, store):
    result = agent.extract("missing_po", run_id="run-1")
    record = record_intake(store, result)
    assert record.next_stage is Stage.POLICY
    loaded = store.get_run("run-1")
    assert loaded.source == "missing_po"
    assert loaded.status is RunStatus.RUNNING
    assert loaded.invoice() == result.invoice
    assert loaded.extracted == to_jsonable(result.invoice)
    (step,) = loaded.steps
    assert step.stage is Stage.INTAKE
    assert step.output["extractor"] == "fixture"
    assert step.output["invoice"] == to_jsonable(result.invoice)


def test_record_intake_uses_existing_run(agent, store):
    store.start_run("uploaded.txt", run_id="run-2")
    record = record_intake(store, agent.extract("vendor_alias", run_id="run-2"))
    assert record.source == "uploaded.txt"
    assert [s.stage for s in record.steps] == [Stage.INTAKE]


def test_record_intake_twice_is_rejected(agent, store):
    result = agent.extract("clean_baseline", run_id="run-3")
    record_intake(store, result)
    with pytest.raises(ValueError, match="expected stage policy"):
        record_intake(store, result)
    assert len(store.get_run("run-3").steps) == 1
